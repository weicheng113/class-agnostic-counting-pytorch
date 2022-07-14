from collections import OrderedDict, defaultdict

import torch
from catalyst import dl
from catalyst.callbacks import SchedulerCallback, FunctionalBatchMetricCallback, LoaderMetricCallback
from catalyst.metrics import FunctionalBatchMetric, FunctionalLoaderMetric
from catalyst.runners import SupervisedRunner
from torch.utils.data import RandomSampler, BatchSampler, DataLoader

from adaptation import MyMetrics
from center_weighted_mse_loss import CenterWeightedMSELoss
from data import CarPKDataset, collate_fn
from lr_logger import LRLogger
from model import Generic_Matching_Net, config
from train_catalyst import get_optimizer, get_lr_schedule, count_parameters


def main():
    BS = 32
    EPOCHS = 25
    nw = 4
    # nw = 0
    data_root = "/home/ubuntu/datasets/carpk/"
    trn_ds = CarPKDataset(data_root=data_root, data_meta_dir="./datasets/meta/", mode="train")
    trn_sampler = RandomSampler(
        data_source=trn_ds,
        num_samples=len(trn_ds),
        # num_samples=BS*2,
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
        num_samples=len(val_ds),
        # num_samples=BS*2,
        replacement=True)
    # val_sampler = ValidSampler(val_ds)
    val_batch_sampler = BatchSampler(val_sampler, batch_size=BS, drop_last=False)
    val_dl = DataLoader(val_ds, batch_sampler=val_batch_sampler, collate_fn=collate_fn, num_workers=nw)

    dataloaders = OrderedDict(
        train=trn_dl,
        valid=val_dl)

    optimizer_config = OrderedDict(
        name='adam',
        # adapt_lr=0.0,
        adapt_lr=2.5e-4,
        main_lr=2.5e-4,
        weight_decay=1e-4,
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