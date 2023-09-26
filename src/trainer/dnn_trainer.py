import itertools
import json
import os.path
import sys
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyrootutils
import sklearn
import torch
import torchsummary
from lightning import LightningModule
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from torchmetrics import MaxMetric, MeanMetric, F1Score
from torchmetrics.classification.accuracy import Accuracy

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.file_mgmt import default
from src.utils.vis import calculate_cm_stats

from src.utils.device import print_memory_usage, print_cpu_cores, print_gpu_memory
from src import utils
from src.models.components.layers_init import xavier_initialize_weights

logging = utils.get_pylogger(__name__)


class DnnLitModule(LightningModule):
    """Pytorch Lightning module for deep neural network model.
    This class implements training, validation and test steps for EEGNet model.
    """

    def __init__(
            self,
            net,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            criterion: torch.nn.Module,
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

        # weight initialization
        self.net.apply(xavier_initialize_weights)

        # ATTRIBUTES TO SAVE BATCH OUTPUTS
        self.training_step_outputs = []  # save outputs in each batch to compute metric overall epoch
        self.training_step_targets = []  # save targets in each batch to compute metric overall epoch

        self.train_loss = MeanMetric()
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=self.num_classes)

        # Validation metric objects for calculating and averaging accuracy across batches
        self.val_loss = MeanMetric()
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=self.num_classes)

        # for tracking best so far validation accuracy (used for checkpointing/ early stopping / HPO)
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

        # Testing metric objects for calculating and averaging accuracy across batches
        self.test_loss = MeanMetric()
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=self.num_classes)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def model_step(self, batch: Any):

        x, y = batch

        # TODO Runs the forward pass in mixed precision
        # with autocast():

        logits = self.forward(x)
        # FIXME : add class weights
        # classes_weights_tensor = torch.tensor(self.calculate_sample_weights(y)).to(self.device)
        # loss = self.criterion(weight=classes_weights_tensor)(logits, y)
        loss = self.criterion()(logits, y)

        # transform logits to one hot encoding
        preds_ie = torch.argmax(logits, dim=1)
        preds = torch.nn.functional.one_hot(preds_ie, num_classes=self.num_classes)

        return loss, preds, y

    def on_train_start(self):

        # print resources info
        print_memory_usage()
        print_cpu_cores()
        print_gpu_memory()

        # log hyperparameters
        self.save_hparams_to_mlflow()

        # reset metrics
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_f1.reset()
        self.val_f1_best.reset()

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
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):

        train_all_outputs = self.training_step_outputs
        train_all_targets = self.training_step_targets

        # F1 Macro all epoch saving outputs and target per batch
        f1_macro_epoch = f1_score(train_all_outputs, train_all_targets, average='macro')
        self.log("train/f1_epoch", f1_macro_epoch, on_step=False, on_epoch=True, prog_bar=True)
        # Calculate the confusion matrix and log it to mlflow
        if (self.current_epoch + 1) % self.plots_settings["plot_every_n_epoch"] == 0:
            # Calculate the confusion matrix and log it to mlflow
            cm = confusion_matrix(train_all_targets, train_all_outputs)
            title = f"Training Confusion matrix epoch_{self.current_epoch}"
            self.confusion_matrix_to_png(cm, title, f"train_cm_epo_{self.current_epoch}")

        # Log gradient for each parameter FIXME
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f"grad_{name}_norm", param.grad.norm().item(), on_step=False, on_epoch=True, prog_bar=True)

        # free up the memory
        # --> HERE STEP 3 <--
        self.training_step_outputs.clear()
        self.training_step_targets.clear()

    def on_validation_start(self):

        self.class_names = self.trainer.datamodule.classes

        # mlflow autologging
        self.mlflow_client = self.logger.experiment
        self.run_id = self.logger.run_id

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
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):

        val_all_outputs = self.val_step_outputs
        val_all_targets = self.val_step_targets

        # F1 Macro all epoch saving outputs and target per batch
        val_f1_macro_epoch = f1_score(val_all_outputs, val_all_targets, average='macro')
        self.log("val/f1_epoch", val_f1_macro_epoch, on_step=False, on_epoch=True, prog_bar=True)
        # Calculate the confusion matrix and log it to mlflow
        if (self.current_epoch + 1) % self.plots_settings["plot_every_n_epoch"] == 0:
            # Calculate the confusion matrix and log it to mlflow
            cm = confusion_matrix(val_all_targets, val_all_outputs)
            title = f"Validation Confusion matrix epoch_{self.current_epoch}"
            self.confusion_matrix_to_png(cm, title, f"vali_cm_epo_{self.current_epoch}")

        # Accuracy is a metric object, so we need to call `.compute()` to get the value
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

        # F1 is a metric object, so we need to call `.compute()` to get the value
        f1 = self.val_f1.compute()  # get current val f1
        self.val_f1_best(f1)  # update best so far val f1
        # log `val_f1_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)

        # free up the memory
        # --> HERE STEP 3 <--
        self.val_step_outputs.clear()
        self.val_step_targets.clear()

    def on_test_start(self):
        # log hyperparameters
        hparams = {}
        state_dict = self.trainer.datamodule.state_dict(stage="test")
        # FIXME
        hparams["data_type"] = "forced_trials"
        hparams["test_data_shape"] = state_dict["test_data_shape"]
        hparams["test_classes_weights"] = state_dict["test_classes_weights"]
        hparams["test_stats"] = state_dict["test_stats"]

        # Use a temporary directory to save
        with tempfile.TemporaryDirectory() as tmpdirname:
            hparam_path = os.path.join(tmpdirname, "test_dataset_info.json")
            with open(hparam_path, "w") as f:
                json.dump(hparams, f, default=default)
            self.mlflow_client.log_artifacts(self.run_id,
                                             local_dir=tmpdirname)

        # --> HERE STEP 1 <--
        # ATTRIBUTES TO SAVE BATCH OUTPUTS
        self.test_step_outputs = []  # save outputs in each batch to compute metric overall epoch
        self.test_step_targets = []  # save targets in each batch to compute metric overall epoch

    def test_step(self, batch: Any, batch_idx: int):

        loss, preds, targets = self.model_step(batch)

        y_pred = preds.argmax(axis=1).cpu().numpy()
        y_true = targets.argmax(axis=1).cpu().numpy()
        # --> HERE STEP 2 <--
        self.test_step_outputs.extend(y_pred)
        self.test_step_targets.extend(y_true)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):

        test_all_outputs = self.test_step_outputs
        test_all_targets = self.test_step_targets

        ## F1 Macro all epoch saving outputs and target per batch
        test_f1_macro_epoch = f1_score(test_all_targets, test_all_outputs, average='macro')
        self.log("test/f1_epoch", test_f1_macro_epoch, on_step=False, on_epoch=True, prog_bar=True)

        cm = confusion_matrix(test_all_outputs, test_all_targets)
        title = f"Testing Confusion matrix, forced trials"
        self.confusion_matrix_to_png(cm, title, f"test_cm_forced_trials")

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

    def log_haprams(self):
        """Log all hyperparameters to mlflow"""
        for k, v in self.hparams.items():
            self.mlflow_client.log_param(self.run_id, k, v)
    def calculate_sample_weights(self, y, NUM_MIN_SAMPLES=10):
        """Calculate sample weights for unbalanced dataset"""
        y_np = np.argmax(y.cpu().numpy(), axis=1)

        num_classes = y.shape[1]
        # If the number of samples is small, just return equal weights for all classes
        if len(y_np) < NUM_MIN_SAMPLES:  # Define SOME_THRESHOLD as per your needs
            return np.ones(num_classes)

        # Classes present in y
        present_classes = np.unique(y_np)
        # All possible classes based on the shape of y

        if len(present_classes) < num_classes:
            # Compute weights for present classes
            present_class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                                    classes=present_classes,
                                                                                    y=y_np)

            # Initialize weights for all classes with a high value
            all_class_weights = np.max(present_class_weights) * np.ones(num_classes)

            # Set the computed weights for the classes present in y
            for cls, weight in zip(present_classes, present_class_weights):
                all_class_weights[cls] = weight

            return all_class_weights

        else:
            total_classes = np.arange(num_classes)
            class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                            classes=total_classes,
                                                                            y=y_np)

            return class_weights

    def save_hparams_to_mlflow(self):
        """Log all hyperparameters to mlflow"""

        model_name = self.hparams.net.__class__.__name__
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.class_names = self.trainer.datamodule.classes

        # mlflow autologging
        self.mlflow_client = self.logger.experiment
        self.run_id = self.logger.run_id

        # log hyperparameters
        # self.log_haprams()
        hparams = {}
        # load successfull loaded data sessions dict
        # stage = self.trainer.state.fn.value
        state_dict = self.trainer.datamodule.state_dict(stage="fit")
        self.mlflow_client.log_param(self.run_id, "task", state_dict["task_name"])

        hparams["train_data_shape"] = state_dict["train_data_shape"]
        hparams["valid_data_shape"] = state_dict["valid_data_shape"]
        hparams["train_classes_weights"] = state_dict["train_classes_weights"]
        hparams["valid_classes_weights"] = state_dict["valid_classes_weights"]
        hparams["train_stats"] = state_dict["train_stats"]
        hparams["valid_stats"] = state_dict["valid_stats"]

        with tempfile.TemporaryDirectory() as tmpdirname:
            hparam_path = os.path.join(tmpdirname, "dataset_info.json")
            with open(hparam_path, "w") as f:
                json.dump(hparams, f, default=default)

            self.mlflow_client.log_artifacts(self.run_id,
                                             local_dir=tmpdirname)

        for subject_name, subject_info_dict in state_dict["subjects_info_dict"].items():
            if self.trainer.datamodule.loading_data_mode != "cross_subject_hpo":
                self.mlflow_client.log_param(self.run_id, "pvc", subject_info_dict["pvc"])
            # Use a temporary directory to save
            with tempfile.TemporaryDirectory() as tmpdirname:
                hparam_path = os.path.join(tmpdirname, f"{subject_name}_info.json")
                with open(hparam_path, "w") as f:
                    json.dump(subject_info_dict, f, default=default)

                self.mlflow_client.log_artifacts(self.run_id,
                                                 local_dir=tmpdirname)

        # save model summary as a text file
        with tempfile.TemporaryDirectory() as tmpdirname:
            x_shape = (1, self.net.num_electrodes, self.net.chunk_size)
            summary_path = os.path.join(tmpdirname, f"{model_name}_summary.txt")
            with open(summary_path, "w") as f:
                sys.stdout = f
                torchsummary.summary(self.net, x_shape, device="cuda")
                sys.stdout = sys.__stdout__

            self.mlflow_client.log_artifacts(self.run_id,
                                             local_dir=tmpdirname)
        # save to onnx
        with tempfile.TemporaryDirectory() as tmpdirname:
            x_tmp = torch.randn(1, 1, self.net.num_electrodes, self.net.chunk_size).cuda()
            onnx_path = os.path.join(tmpdirname, f"{model_name}.onnx")
            torch.onnx.export(self.net, x_tmp, onnx_path, verbose=True, input_names=["input"],
                              output_names=["output"])
            self.mlflow_client.log_artifacts(self.run_id,
                                             local_dir=tmpdirname)
