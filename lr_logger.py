import torch
from catalyst.core import IRunner
from catalyst.dl import Callback, CallbackOrder

LRs = {0: 'adapter',
       1: 'main'}
class LRLogger(Callback):
    def __init__(self,):
        super().__init__(order=CallbackOrder.External)
    # override
    def _throw_on_tensorboard(self, runner: IRunner, step, name):
        tensorboard_logger = runner.loggers['tensorboard']
        tensorboard_logger._check_loader_key(loader_key=runner.loader_key)
        tensorboard = tensorboard_logger.loggers[runner.loader_key]
        if isinstance(runner.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            for ll, slr in enumerate(runner.scheduler.optimizer.param_groups):
                tensorboard.add_scalar(f"LR/{LRs[ll]}/{name}", slr['lr'], step)
        else:
            for ll, slr in enumerate(runner.scheduler.get_last_lr()):
                tensorboard.add_scalar(f"LR/{LRs[ll]}/{name}", slr, step)

    def on_batch_end(self, runner: IRunner):
        # print("on_batch_end")
        # every train loader
        if not runner.is_train_loader:
            return
        self._throw_on_tensorboard(runner, runner.epoch_step, 'batch')

    def on_epoch_end(self, runner: IRunner):
        # print("on_epoch_end")
        if not runner.is_train_loader:
            return

        self._throw_on_tensorboard(runner,  runner.global_epoch, 'epoch')
