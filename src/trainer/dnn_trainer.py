import os.path
import tempfile
from typing import Any

import numpy as np
import pyrootutils
import sklearn
import torch
from lightning import LightningModule
# from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.vis import calculate_cm_stats
import matplotlib.pyplot as plt
import itertools
import mlflow


class DnnLitModule(LightningModule):
    """Pytorch Lightning module for deep neural network model.
    This class implements training, validation and test steps for EEGNet model.
    """

    def __init__(
            self,
            net,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn,
            scheduler: torch.optim.lr_scheduler,
            plots_settings: dict,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)

        # set net
        self.net = net
        self.num_classes = self.net.num_classes

        # loss function
        self.criterion = self.hparams.criterion

        # plots settings
        self.plots_settings = plots_settings

        # Training metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_loss = MeanMetric()
        # --> HERE STEP 1 <--
        # ATTRIBUTES TO SAVE BATCH OUTPUTS
        self.training_step_outputs = []  # save outputs in each batch to compute metric overall epoch
        self.training_step_targets = []  # save targets in each batch to compute metric overall epoch

        # Validation metric objects for calculating and averaging accuracy across batches
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_loss = MeanMetric()
        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # Testing metric objects for calculating and averaging accuracy across batches
        self.test_loss = MeanMetric()
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)




    def forward(self, x: torch.Tensor):
        return self.net(x)

    def model_step(self, batch: Any):

        x, y = batch
        logits = self.forward(x).double()
        classes_weights_tensor = torch.tensor(self.calculate_sample_weights(y)).to(self.device)
        loss = self.criterion(weight=classes_weights_tensor)(logits, y)
        preds_ie = torch.argmax(logits, dim=1)
        preds = torch.nn.functional.one_hot(preds_ie, num_classes=self.num_classes)

        return loss, preds, y

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.class_names = self.trainer.datamodule.classes

        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, batch: Any):
        loss, preds, targets = self.model_step(batch)

        # GET AND SAVE OUTPUTS AND TARGETS PER BATCH
        # FIXME: only support cpu
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

        if (self.current_epoch + 1) % self.plots_settings["plot_every_n_epoch"] == 0:
            # Calculate the confusion matrix and log it to mlflow
            cm = confusion_matrix(train_all_targets, train_all_outputs)
            title = f"Training Confusion matrix epoch_{self.current_epoch}"
            self.confusion_matrix_to_png(cm, title, f"train_cm_epo_{self.current_epoch}")

        # free up the memory
        # --> HERE STEP 3 <--
        self.training_step_outputs.clear()
        self.training_step_targets.clear()

    def on_validation_start(self):

        self.class_names = self.trainer.datamodule.classes

        # mlflow autologging
        self.mlflow_client = self.logger.experiment
        self.run_id = self.logger.run_id

        mlflow.set_tracking_uri(self.mlflow_client.tracking_uri)
        mlflow.pytorch.autolog()

        # --> HERE STEP 1 <--
        # ATTRIBUTES TO SAVE BATCH OUTPUTS
        self.val_step_outputs = []  # save outputs in each batch to compute metric overall epoch
        self.val_step_targets = []  # save targets in each batch to compute metric overall epoch

    def validation_step(self, batch: Any, batch_idx: int):

        loss, preds, targets = self.model_step(batch)
        # GET AND SAVE OUTPUTS AND TARGETS PER BATCH
        y_pred = preds.argmax(axis=1).cpu().numpy()
        y_true = targets.argmax(axis=1).cpu().numpy()

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

        if (self.current_epoch + 1) % self.plots_settings["plot_every_n_epoch"] == 0:
            # Calculate the confusion matrix and log it to mlflow
            cm = confusion_matrix(val_all_targets, val_all_outputs)
            title = f"Validation Confusion matrix epoch_{self.current_epoch}"
            self.confusion_matrix_to_png(cm, title, f"vali_cm_epo_{self.current_epoch}")

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

        y_pred = preds.argmax(axis=1).cpu().numpy()
        y_true = targets.argmax(axis=1).cpu().numpy()

        cm = confusion_matrix(y_true, y_pred)
        title = f"Testing Confusion matrix, epoch {self.current_epoch}"
        self.confusion_matrix_to_png(cm, title, f"test_cm_epo_{self.current_epoch}")

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
            # FIXME : check if this is correct order of classes names
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

            # Use a temporary directory to save the plot
            with tempfile.TemporaryDirectory() as tmpdirname:
                plot_path = os.path.join(tmpdirname, fig_file_path)

                plt.savefig(plot_path)  # Save the plot to the temporary directory
                # self.mlflow_client.log_artifact(tmpdirname)
                self.mlflow_client.log_artifacts(self.run_id,
                                                 local_dir=tmpdirname)
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

            # Use a temporary directory to save the plot
            with tempfile.TemporaryDirectory() as tmpdirname:
                plot_path = os.path.join(tmpdirname, fig_file_path)

                plt.savefig(plot_path)  # Save the plot to the temporary directory
                self.mlflow_client.log_artifacts(self.run_id,
                                                 local_dir=tmpdirname)
                plt.close(figure)

    def roc_curve_to_png(self, fpr, tpr, roc_auc, title, figure_file_name=None):
        """Compute and plot the roc curve. It calculates a standard roc curve or the mean and std of a list of roc curve
                    used for cross validation folds. """

        plt.rcParams["font.family"] = 'DejaVu Sans'
        plt.figure(figsize=(10, 7))

        # TODO: remove if not needed
        if set_name == "vali":
            for i, (fpr, tpr) in enumerate(self.val_roc_curve_list):
                auc_val = sklearn.metrics.auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, auc_val))
        elif set_name == "test":
            for i, (fpr, tpr) in enumerate(self.test_roc_curve_list):
                auc_val = sklearn.metrics.auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, auc_val))

        # TODO : walkaround check shape of tprs
        new_tprs = []
        mean_fpr_len = len(mean_fpr)
        for tpr in tprs:
            if len(tpr) != mean_fpr_len:
                tpr = tpr[:mean_fpr_len]
            new_tprs.append(tpr)

        mean_tpr = np.mean(new_tprs, axis=0)
        mean_tpr[-1] = 1.0  # set last value to 1.0 to have a complete curve
        mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.title(title)
        plt.legend(loc='lower right')

        with tempfile.TemporaryDirectory() as tmpdirname:
            plot_path = os.path.join(tmpdirname, f"{title}.png")

            plt.savefig(plot_path)  # Save the plot to the temporary directory
            mlflow.log_artifacts(tmpdirname)

    def log_haprams(self):
        """Log all hyperparameters to mlflow"""
        for k, v in self.hparams.items():
            self.mlflow_client.log_param(self.run_id, k, v)

    def calculate_sample_weights(self, y):
        """Calculate sample weights for unbalanced dataset"""
        y_np = np.argmax(y.cpu().numpy(), axis=1)
        class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                        classes=np.unique(y_np),
                                                                        y=y_np)
        return class_weights
