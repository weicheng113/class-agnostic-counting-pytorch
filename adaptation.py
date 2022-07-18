from collections import OrderedDict, defaultdict
from typing import Callable, Iterable, Dict, Tuple

import torch
from catalyst import dl
from catalyst.callbacks import SchedulerCallback, FunctionalBatchMetricCallback, LoaderMetricCallback, \
    BatchMetricCallback
from catalyst.callbacks.metric import _IMetricCallback
from catalyst.metrics import FunctionalBatchMetric, FunctionalLoaderMetric, ICallbackLoaderMetric, AccumulativeMetric
from catalyst.runners import SupervisedRunner
from torch.utils.data import RandomSampler, BatchSampler, DataLoader

from center_weighted_mse_loss import CenterWeightedMSELoss
from data import CarPKDataset, collate_fn
from lr_logger import LRLogger
from model import Generic_Matching_Net, config
from train_catalyst import get_optimizer, get_lr_schedule, count_parameters


class MyAccumulativeMetric:
    def __init__(self):
        self.sum_absolute_error = 0
        self.sum_square_error = 0
        self.n_samples = 0
        self.sum_true_count = 0
        self.sum_pred_count = 0

    def reset(self):
        self.sum_absolute_error = 0
        self.sum_square_error = 0
        self.n_samples = 0
        self.sum_true_count = 0
        self.sum_pred_count = 0

    def update_metric(self, new_inputs):
        prediction, label = new_inputs
        logits = prediction["logits"].detach().cpu()
        label = label.detach().cpu()
        batch_size = logits.shape[0]
        true_counts = label.sum(dim=(1, 2, 3))/100.0
        # pred_counts = (logits > 2).sum(dim=(1, 2, 3))
        pred_counts = logits.sum(dim=(1, 2, 3))/100.0

        errors = (true_counts - pred_counts).abs()
        self.sum_absolute_error += errors.sum()
        self.sum_square_error += (errors**2).sum()
        self.n_samples += batch_size
        self.sum_true_count += true_counts.sum()
        self.sum_pred_count += pred_counts.sum()

        mae = self.sum_absolute_error/self.n_samples
        rmse = torch.sqrt(self.sum_square_error/self.n_samples)
        avg_true_count = self.sum_true_count/self.n_samples
        avg_pred_count = self.sum_pred_count/self.n_samples

        return {
            "MAE": mae,
            "RMSE": rmse,
            "avg_true_count": avg_true_count,
            "avg_pred_count": avg_pred_count,
            "true_count": true_counts[0],
            "pred_count": pred_counts[0]
        }


class MyMetrics(_IMetricCallback):
    def __init__(self):
        super(MyMetrics, self).__init__()
        self.metric = MyAccumulativeMetric()
        self.input_key = "logits"
        self.target_key = "targets"

    def on_loader_start(self, runner: "IRunner") -> None:
        self.metric.reset()

    def on_batch_end(self, runner: "IRunner") -> None:
        """On batch end action: update metric with new batch data
        and log it's value if necessary

        Args:
            runner: current runner
        """
        metrics_inputs = self.get_inputs(runner=runner)
        metrics = self.metric.update_metric(metrics_inputs)
        runner.batch_metrics.update(metrics)

    def get_inputs(self, runner: "IRunner") -> Tuple[torch.Tensor, torch.Tensor]:
        return runner.batch[self.input_key], runner.batch[self.target_key]

    def on_loader_end(self, runner: "IRunner") -> None:
        pass


def main():
    BS = 64
    EPOCHS = 25
    # nw = 8
    nw = 0
    data_root = "/media/cwei/WD_BLACK/datasets/CARPK/CARPK_devkit/data/"
    trn_ds = CarPKDataset(data_root=data_root, data_meta_dir="./datasets/meta/", mode="train")
    trn_sampler = RandomSampler(
        data_source=trn_ds,
        # num_samples=len(trn_ds),
        num_samples=BS*2,
        replacement=True)
    trn_batch_sampler = BatchSampler(trn_sampler, batch_size=BS, drop_last=False)
    trn_dl = DataLoader(trn_ds, batch_sampler=trn_batch_sampler, collate_fn=collate_fn, num_workers=nw)

    val_ds = CarPKDataset(data_root=data_root, data_meta_dir="./datasets/meta/", mode='valid')
    # val_sampler = ValidSamplerSubset(
    #     data_source=val_ds,
    #     num_samples=100
    # )
    # val_sampler = RandomSampler(
    #     data_source=val_ds,
    #     num_samples=BS*2,
    #     replacement=True)
    val_sampler = RandomSampler(
        data_source=val_ds,
        # num_samples=len(val_ds),
        num_samples=BS*2,
        replacement=True)
    # val_sampler = ValidSampler(val_ds)
    val_batch_sampler = BatchSampler(val_sampler, batch_size=BS, drop_last=False)
    val_dl = DataLoader(val_ds, batch_sampler=val_batch_sampler, collate_fn=collate_fn, num_workers=nw)

    dataloaders = OrderedDict(
        train=trn_dl,
        valid=val_dl)

    optimizer_config = OrderedDict(
        # name='adam',
        name='adamw',
        # adapt_lr=0.0,
        adapt_lr=5e-5,
        main_lr=5e-5,
        weight_decay=0.0,
        # adapt_lr=2.5e-4,
        # main_lr=2.5e-4,
        # weight_decay=1e-4,
        momentum=0.9)

    #this will depend on the lr scheduler
    base_lr_scheduler_config = lambda x: defaultdict(schedule=x)

    #onecycle
    onecycle_lr_config = base_lr_scheduler_config('onecycle')
    onecycle_lr_config['pct_start'] = 0.2
    onecycle_lr_config['epochs'] = EPOCHS
    onecycle_lr_config['steps_per_epoch'] = len(trn_batch_sampler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Generic_Matching_Net(config=config)
    print(f"{count_parameters(model)}")
    model = model.to_adapting_mode()
    optimizer = get_optimizer(optimizer_config, model)
    lr_scheduler = get_lr_schedule(onecycle_lr_config, optimizer)

    print(f"{count_parameters(model)}")
    my_metrics = MyMetrics()

    runner = SupervisedRunner(
        # device=device,
        input_key='images',
        output_key='logits',
        target_key='targets'
    )
    runner.train(model=model,
                 optimizer=optimizer,
                 criterion=CenterWeightedMSELoss(),
                 loaders=dataloaders,
                 logdir='./car_adapt/',
                 loggers={"tensorboard": dl.TensorboardLogger(logdir="./car_adapt/tensorboard")},
                 callbacks=[SchedulerCallback(mode='batch'), LRLogger(), my_metrics],
                 scheduler=lr_scheduler,
                 num_epochs=EPOCHS,
                 verbose=True)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU
    main()