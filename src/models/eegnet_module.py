import os.path
from typing import Any

import numpy as np
import pyrootutils
import torch
from lightning import LightningModule
# from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from sklearn.metrics import f1_score

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
            net,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn,
            scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)

        # set net
        self.net = net

        # loss function
        self.criterion = self.hparams.criterion

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

        # --> HERE STEP 1 <--
        # ATTRIBUTES TO SAVE BATCH OUTPUTS
        self.training_step_outputs = []  # save outputs in each batch to compute metric overall epoch
        self.training_step_targets = []  # save targets in each batch to compute metric overall epoch
        self.val_step_outputs = []  # save outputs in each batch to compute metric overall epoch
        self.val_step_targets = []  # save targets in each batch to compute metric overall epoch


    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.class_names = self.trainer.datamodule.classes
        self.num_classes = self.trainer.datamodule.num_classes

        self.mlflow_client = self.logger.experiment
        self.run_id = self.logger.run_id

        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y = batch

        # fixme
        # x = torch.squeeze(x)

        logits = self.forward(x).double()
        loss = self.criterion()(logits, y)
        preds_ie = torch.argmax(logits, dim=1)
        preds = torch.nn.functional.one_hot(preds_ie, num_classes=self.num_classes)
        return loss, preds, y

    def training_step(self, batch: Any):
        loss, preds, targets = self.model_step(batch)

        # GET AND SAVE OUTPUTS AND TARGETS PER BATCH
        y_pred = preds.argmax(axis=1).cpu().numpy()
        y_true = targets.argmax(axis=1).cpu().numpy()

        # --> HERE STEP 2 <--
        self.training_step_outputs.extend(y_pred)
        self.training_step_targets.extend(y_true)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        ## F1 Macro all epoch saving outputs and target per batch
        train_all_outputs = self.training_step_outputs
        train_all_targets = self.training_step_targets
        f1_macro_epoch = f1_score(train_all_outputs, train_all_targets, average='macro')
        self.log("train_f1_epoch", f1_macro_epoch, on_step=False, on_epoch=True, prog_bar=True)

        if self.current_epoch % 2 == 0:
            # Calculate the confusion matrix and log it to mlflow
            cm = confusion_matrix(train_all_targets, train_all_outputs)
            title = f"Training Confusion matrix, epoch {self.current_epoch}"
            file_name = f"Training_confusion_matrix_epoch_{self.current_epoch}.png"
            self.confusion_matrix_to_png(cm, title, file_name)

        # free up the memory
        # --> HERE STEP 3 <--
        self.training_step_outputs.clear()
        self.training_step_targets.clear()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        # GET AND SAVE OUTPUTS AND TARGETS PER BATCH
        y_pred = preds.argmax().cpu().numpy()
        y_true = targets.argmax().cpu().numpy()

        # --> HERE STEP 2 <--
        self.val_step_outputs.extend(y_pred)
        self.val_step_targets.extend(y_true)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):

        ## F1 Macro all epoch saving outputs and target per batch
        val_all_outputs = self.val_step_outputs
        val_all_targets = self.val_step_targets
        val_f1_macro_epoch = f1_score(val_all_outputs, val_all_targets, average='macro')
        self.log("val_f1_epoch", val_f1_macro_epoch, on_step=False, on_epoch=True, prog_bar=True)

        if self.current_epoch % 2 == 0:
            # Calculate the confusion matrix and log it to mlflow
            cm = confusion_matrix(val_all_targets, val_all_outputs)
            title = f"Validation Confusion matrix, epoch {self.current_epoch}"
            file_name = f"Vali_confusion_matrix_epoch_{self.current_epoch}.png"
            self.confusion_matrix_to_png(cm, title, file_name)

        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

        # free up the memory
        # --> HERE STEP 3 <--
        self.val_step_outputs.clear()
        self.val_step_targets.clear()

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
                    "monitor": "train/loss",
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
                                            artifact_path="plots")
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
