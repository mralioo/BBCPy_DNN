import logging
from typing import Any, Mapping, Optional

import pytorch_lightning as pl
import torch
import torchmetrics

from lightning_torch.core.datamodule_base import BaseDataset
from lightning_torch.utils.file_mgmt import get_dir_by_indicator

# from nn_core.model_logging import NNLogger


PROJECT_ROOT = get_dir_by_indicator(indicator="ROOT")

pylogger = logging.getLogger(__name__)


class MyLightningModule(pl.LightningModule):

    def __init__(self,
                 net: torch.nn.Module,
                 metadata: Optional[BaseDataset] = None,
                 *args, **kwargs) -> None:
        super().__init__()

        """ MyLightningModule is a wrapper class from the pytorch_lightning library 
        
            A LightningModule organizes your PyTorch code into 5 sections:
                - Computations (init).
                - Train loop (training_step)
                - Validation loop (validation_step)
                - Test loop (test_step)
                - Optimizers (configure_optimizers)
            Read the docs:
                https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
        """

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata

        # example
        metric = torchmetrics.Accuracy()
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        # example
        return self.net(x)

    def step(self, x, y) -> Mapping[str, Any]:
        # example
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return {"logits": logits.detach(), "loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x, y = batch
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/train": self.train_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x, y = batch
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.val_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/val": self.val_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x, y = batch
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/test": step_out["loss"].cpu().detach()},
        )

        self.test_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/test": self.test_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

    # def configure_optimizers(
    #         self,
    # ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
    #     """Choose what optimizers and learning-rate schedulers to use in your optimization.
    #
    #     Normally you'd need one. But in the case of GANs or similar you might have multiple.
    #
    #     Return:
    #         Any of these 6 options.
    #         - Single optimizer.
    #         - List or Tuple - List of optimizers.
    #         - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
    #         - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
    #           key whose value is a single LR scheduler or lr_dict.
    #         - Tuple of dictionaries as described, with an optional 'frequency' key.
    #         - None - Fit will run without any optimizer.
    #     """
    #     opt = hydra.utils.instantiate(self.hparams.model.optimizer, params=self.parameters(), _convert_="partial")
    #     if "lr_scheduler" not in self.hparams:
    #         return [opt]
    #     scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
    #     return [opt], [scheduler]

# @hydra.main(config_path=os.path.join(PROJECT_ROOT, "configs", "debug"), config_name="test_only")
# def main(cfg: omegaconf.DictConfig) -> None:
#     """Debug main to quickly develop the Lightning Module.
#
#     Args:
#         cfg: the hydra configuration
#     """
#     _: pl.LightningModule = hydra.utils.instantiate(
#         cfg.model,
#         optim=cfg.optim,
#         _recursive_=False,
#     )
#
#
# if __name__ == "__main__":
#     main()
