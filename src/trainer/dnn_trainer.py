import os.path
from typing import Any

import numpy as np
import pyrootutils
import torch
from lightning import LightningModule
from sklearn.metrics import confusion_matrix
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.vis import calculate_cm_stats
import matplotlib.pyplot as plt
import itertools


class EEGNetLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn,
            scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)

        self.net = net

        # loss function
        self.criterion = self.hparams.criterion
        #
        # # scheduler
        # self.scheduler = scheduler

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.training_step_outputs = {"preds": [], "targets": []}
        self.validation_step_outputs = {"preds": [], "targets": []}

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

        self.class_names = self.trainer.datamodule.classes
        self.num_classes = len(self.class_names)

        self.mlflow_client = self.logger.experiment
        self.run_id = self.logger.run_id

        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x).double()
        loss = self.criterion()(logits[0], y)
        preds_ie = torch.argmax(logits, dim=1)
        preds = torch.nn.functional.one_hot(preds_ie, num_classes=2)[0]
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        self.training_step_outputs["preds"].append(preds)
        self.training_step_outputs["targets"].append(targets)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass
        # outputs = self.training_step_outputs
        #
        # if self.current_epoch % 2 == 0:
        #     y_true = torch.stack(outputs["targets"]).cpu().numpy()  # True labels
        #     y_pred = torch.stack(outputs["preds"]).cpu().numpy()  # Predicted labels
        #
        #     # Calculate the confusion matrix and log it to mlflow
        #     cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        #     title = f"Training Confusion matrix, epoch {self.current_epoch}"
        #     file_name = f"Training_confusion_matrix_epoch_{self.current_epoch}.png"
        #     self.confusion_matrix_to_png(cm, title, file_name)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        self.validation_step_outputs["preds"].append(preds)
        self.validation_step_outputs["targets"].append(targets)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        outputs = self.validation_step_outputs
        if self.current_epoch % 2 == 0:
            y_true = torch.stack(outputs["targets"]).cpu().numpy()  # True labels
            y_pred = torch.stack(outputs["preds"]).cpu().numpy()  # Predicted labels

            # Calculate the confusion matrix and log it to mlflow
            cm = confusion_matrix(y_true, y_pred)
            title = f"Validation Confusion matrix, epoch {self.current_epoch}"
            file_name = f"Vali_confusion_matrix_epoch_{self.current_epoch}.png"
            self.confusion_matrix_to_png(cm, title, file_name)

        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def confusion_matrix_to_png(self, conf_mat, title, figure_file_name=None, type='standard'):
        if type == 'standard':
            plt.rcParams["font.family"] = 'DejaVu Sans'
            figure = plt.figure(figsize=(9, 9))

            plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()
            tick_marks = np.arange(self.num_classes)
            plt.xticks(tick_marks, self.class_names)
            plt.yticks(tick_marks, self.class_names)
            # set title
            plt.title(title)

            # render the confusion matrix with percentage and ratio.
            group_counts = []
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    group_counts.append("{}/{}".format(conf_mat[i, j], conf_mat.sum(axis=1)[i]))
            group_percentages = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals=2)
            labels = [f"{v1} \n {v2 * 100: .4}%" for v1, v2 in zip(group_counts, group_percentages.flatten())]
            labels = np.asarray(labels).reshape(self.num_classes, self.num_classes)

            # set the font size of the text in the confusion matrix
            for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
                color = "red"
                plt.text(j, i, labels[i, j], horizontalalignment="center", color=color, fontsize=15)

            plt.tight_layout()
            plt.ylabel('True label', fontsize=10)
            plt.xlabel('Predicted label', fontsize=10)

            if figure_file_name is None:
                fig_file_path = f'{title}.png'
            else:
                fig_file_path = f'{figure_file_name}.png'

            plt.savefig(fig_file_path)
            self.mlflow_client.log_artifact(self.run_id,
                                            local_path=fig_file_path,
                                            artifact_path=os.path.join(self.mlflow_client.tracking_uri,"plots"))
            plt.close(figure)

        elif type == 'mean':

            # Fixme here I pass list of cm (better way)
            plt.rcParams["font.family"] = 'DejaVu Sans'
            figure = plt.figure(figsize=(9, 9))

            # Add values to the plot
            mean_cm_val, std_cm_val = calculate_cm_stats(cm_list=conf_mat, num_classes=self.num_classes)

            plt.imshow(mean_cm_val, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()
            tick_marks = np.arange(self.num_classes)
            plt.xticks(tick_marks, self.class_names)
            plt.yticks(tick_marks, self.class_names)
            # set title
            plt.title(title)

            labels = [f"{v1 * 100:.4}%  ±{v2 * 100: .4}%" for v1, v2 in
                      zip(mean_cm_val.flatten(), std_cm_val.flatten())]
            labels = np.asarray(labels).reshape(self.num_classes, self.num_classes)

            # thresh = mean_cm_val.max() / 2.0
            for i, j in itertools.product(range(mean_cm_val.shape[0]), range(mean_cm_val.shape[1])):
                color = "red"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

            plt.tight_layout()
            plt.ylabel('True label', fontsize=10)
            plt.xlabel('Predicted label', fontsize=10)

            if figure_file_name is None:
                fig_file_path = f'{title}.png'
            else:
                fig_file_path = f'{figure_file_name}.png'

            plt.savefig(fig_file_path)
            self.mlflow_client.log_artifact(self.run_id,
                                            local_path=fig_file_path,
                                            artifact_path="plots")
            plt.close(figure)
