from typing import Any, List

import torch
from torchmetrics import MaxMetric

from lightning_torch.core.model_base import MyLightningModule


# TODO revisit the importance of this class because only net is need it

class MNISTLitModule(MyLightningModule):
    """Example of LightningModule for MNIST classification.
    Overwrite the function if needed
    """

    def __init__(
            self,
            net: torch.nn.Module,
            lr: float = 0.001,
            weight_decay: float = 0.0005):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    # def forward(self, x: torch.Tensor):
    #     return self.net(x)

    # def step(self, batch: Any):
    #
    #     x, y = batch
    #     logits = self.forward(x)
    #     loss = self.criterion(logits, y)
    #     preds = torch.argmax(logits, dim=1)
    #     return loss, preds, y

    # def training_step(self, batch: Any, batch_idx: int):
    #     loss, preds, targets = self.step(batch)
    #
    #     # log train torch_metrics
    #     acc = self.train_accuracy(preds, targets)
    #     self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
    #     self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
    #
    #     # we can return here dict with any tensors
    #     # and then read it in some callback or in `training_epoch_end()`` below
    #     # remember to always return loss from `training_step()` or else backpropagation will fail!
    #     return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    # def validation_step(self, batch: Any, batch_idx: int):
    #     loss, preds, targets = self.step(batch)
    #
    #     # log val torch_metrics
    #     acc = self.val_accuracy(preds, targets)
    #     self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
    #     self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
    #
    #     return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_accuracy.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    # def test_step(self, batch: Any, batch_idx: int):
    #     loss, preds, targets = self.step(batch)
    #
    #     # log test torch_metrics
    #     acc = self.test_accuracy(preds, targets)
    #     self.log("test/loss", loss, on_step=False, on_epoch=True)
    #     self.log("test/acc", acc, on_step=False, on_epoch=True)
    #
    #     return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset torch_metrics at the end of every epoch
        self.train_accuracy.reset()
        self.test_accuracy.reset()
        self.val_accuracy.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
