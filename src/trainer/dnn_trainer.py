import itertools
import json
import os.path
import sys
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyrootutils
import sklearn
import torch
import torch.nn.functional as F
import torchmetrics
import torchsummary
from lightning import LightningModule
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torchmetrics import MaxMetric, MeanMetric, F1Score
from torchmetrics.classification.accuracy import Accuracy

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from src.utils.file_mgmt import default
from src.utils.vis import calculate_cm_stats, plot_roc_curve
from src.utils.device import print_memory_usage, print_cpu_cores, print_gpu_memory
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

        self.classes_names_dict = {}
        if self.num_classes == 2:
            self.classes_names_dict = {0: "R", 1: "L"}

        if self.num_classes == 4:
            self.classes_names_dict = {0: "R", 1: "L", 2: "U", 3: "D"}

        self.classes_names = [v for k, v in self.classes_names_dict.items()]

        # set loss function
        self.criterion = self.hparams.criterion

        # plots settings
        self.plots_settings = plots_settings

        # weight initialization
        self.net.apply(xavier_initialize_weights)

        # metrics
        self.init_metrics()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def model_step(self, batch: Any):

        x, y = batch

        logits = self.forward(x)

        if self.num_classes == 2:
            probs = F.softmax(logits, dim=1)
        if self.num_classes > 2:
            probs = F.softmax(logits, dim=1)

        # FIXME : add class weights
        # classes_weights_tensor = torch.tensor(self.calculate_sample_weights(y)).to(self.device)
        # loss = self.criterion(weight=classes_weights_tensor)(logits, y)

        loss = self.criterion()(probs, y)

        # transform logits to one hot encoding
        # preds_ie = torch.argmax(probs, dim=1)
        # preds = torch.nn.functional.one_hot(preds_ie, num_classes=self.num_classes)

        return loss, probs, y

    def model_step_logits(self, batch: Any):
        x, y = batch

        logits = self.forward(x)

        if self.num_classes == 2:
            probs = F.softmax(logits, dim=1)
        if self.num_classes > 2:
            probs = F.softmax(logits, dim=1)

        loss = self.criterion()(probs, y)

        return loss, probs, torch.argmax(y, dim=1)

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
        self.train_f1(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):

        train_all_outputs = self.training_step_outputs
        train_all_targets = self.training_step_targets

        # Calculate the confusion matrix and log it to mlflow
        if (self.current_epoch + 1) % self.plots_settings["plot_every_n_epoch"] == 0:
            # Calculate the confusion matrix and log it to mlflow
            cm = confusion_matrix(train_all_targets, train_all_outputs)
            title = f"Training Confusion matrix epoch_{self.current_epoch}"
            self.confusion_matrix_to_png(cm, title, f"train_cm_epo_{self.current_epoch}")

        # F1 is a metric object, so we need to call `.compute()` to get the value
        f1 = self.train_f1.compute()  # get current val f1
        self.train_f1_best(f1)  # update best so far val f1
        # log `val_f1_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("train/f1_best", self.train_f1_best.compute(), sync_dist=True, prog_bar=True)

        # FIXME : Log gradient for each parameter
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f"grad_{name}_norm", param.grad.norm().item(), on_step=False, on_epoch=True, prog_bar=True)

        # free up the memory
        # --> HERE STEP 3 <--
        self.training_step_outputs.clear()
        self.training_step_targets.clear()

    def on_validation_start(self):

        # mlflow autologging
        self.mlflow_client = self.logger.experiment
        self.run_id = self.logger.run_id

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
        self.val_f1(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        # Log metrics from the collection
        if self.num_classes == 2:
            self.roc(preds, targets.long())

            # --> HERE STEP 2 <--
            _, logits, targets_ie = self.model_step_logits(batch)
            self.val_step_outputs_logits.extend(logits.cpu().tolist())
            self.val_step_targets_ie.extend(targets_ie.cpu().tolist())

        if self.num_classes > 2:
            _, logits, targets_ie = self.model_step_logits(batch)
            self.roc(logits, targets_ie)

            # --> HERE STEP 2 <--
            self.val_step_outputs_logits.extend(logits.cpu().tolist())
            self.val_step_targets_ie.extend(targets_ie.cpu().tolist())

        self.tracker.increment()  # Notify the tracker of a new step
        self.tracker.update(preds, targets)

        return loss

    def on_validation_epoch_end(self):

        val_all_outputs = self.val_step_outputs
        val_all_targets = self.val_step_targets

        all_results = self.tracker.compute_all()

        # Calculate the confusion matrix and log it to mlflow
        if (self.current_epoch + 1) % self.plots_settings["plot_every_n_epoch"] == 0:
            # Calculate the confusion matrix and log it to mlflow
            cm = confusion_matrix(val_all_targets, val_all_outputs)
            title = f"Validation Confusion matrix epoch_{self.current_epoch}"
            self.confusion_matrix_to_png(cm, title, f"vali_cm_epo_{self.current_epoch}")

            # Plot ROC curve  Summary FIXME : do we need it ?
            self.roc.compute()
            self.plot_summary_advanced(all_results, image_name=f"val_summary_epoch_{self.current_epoch}")

            # Plot ROC curve ovr and ovo FIXME : test on LR
            self.plot_roc_curve_ovr(Y_ie=self.val_step_targets_ie, Y_pred_logits=self.val_step_outputs_logits,
                                    filename=f"val_roc_ovr_epoch_{self.current_epoch}")
            self.plot_roc_curve_ovo(Y_ie=self.val_step_targets_ie, Y_pred_logits=self.val_step_outputs_logits,
                                    filename=f"val_roc_ovo_epoch_{self.current_epoch}")

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

        # rest tracker
        self.tracker.reset()

        # free up the memory
        # --> HERE STEP 3 <--
        self.val_step_outputs.clear()
        self.val_step_targets.clear()

        self.val_step_outputs_logits.clear()
        self.val_step_targets_ie.clear()

    def on_test_start(self):
        # log hyperparameters
        hparams = {}
        state_dict = self.trainer.datamodule.state_dict(stage="test")
        # FIXME
        hparams["data_type"] = "Run 3 & 6"
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

        self.test_step_logits = []  # save outputs in each batch to compute metric overall epoch
        self.test_step_targets_ie = []  # save targets in each batch to compute metric overall epoch

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
        self.test_f1(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):

        test_all_outputs = self.test_step_outputs
        test_all_targets = self.test_step_targets

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
        # FIXME: add class names
        target_map_dict = {0: "R", 1: "L", 2: "U", 3: "D"}
        class_names = [target_map_dict[i] for i in range(self.num_classes)]

        if type == 'standard':
            plt.rcParams["font.family"] = 'DejaVu Sans'
            figure = plt.figure(figsize=(9, 9))

            plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()
            # FIXME : check if this is correct order of classes names
            tick_marks = np.arange(self.num_classes)
            plt.xticks(tick_marks, class_names)
            plt.yticks(tick_marks, class_names)
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

        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

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

        model_name = self.hparams.net.__class__.__name__
        # save model summary as a text file
        # FIXME : add support for other models
        if model_name == "EEGNet":
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

    def plot_roc_curve_ovr(self, Y_ie, Y_pred_logits, filename=None):

        if isinstance(Y_pred_logits, list):
            Y_pred_logits = np.array(Y_pred_logits)

        fig = plt.figure(figsize=(16, 10))
        bins = [i / 20 for i in range(20)] + [1]

        for i, (k, v) in enumerate(self.classes_names_dict.items()):
            # Gets the class
            # c = k
            # Prepares an auxiliar dataframe to help with the plots
            df_aux = pd.DataFrame(Y_ie)

            df_aux['class'] = [1 if y == k else 0 for y in Y_ie]
            df_aux['prob'] = Y_pred_logits[:, i]
            df_aux = df_aux.reset_index(drop=True)

            # Plots the probability distribution for the class and the rest
            ax = plt.subplot(2, self.num_classes, i + 1)

            # Assuming df_aux has columns 'prob' and 'class'
            classes = df_aux['class'].unique()

            for cls in classes:
                subset = df_aux[df_aux['class'] == cls]
                ax.hist(subset['prob'], bins=bins, label=cls, alpha=0.6)  # alpha is for transparency

            # sns.histplot(x="prob", data=df_aux, hue='class', color='b', ax=ax, bins=bins)
            ax.set_title(v)
            ax.legend([f"Class: {v}", "Rest"])
            ax.set_xlabel(f"P(x = {v})")

            # Calculates the ROC Coordinates and plots the ROC Curves
            ax_bottom = plt.subplot(2, self.num_classes, i + (self.num_classes + 1))
            # tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
            tpr, fpr, _ = roc_curve(y_true=df_aux['class'], y_score=df_aux['prob'])
            plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
            ax_bottom.set_title("ROC Curve OvR")

        # plt.tight_layout()

        # Use a temporary directory to save the plot
        with tempfile.TemporaryDirectory() as tmpdirname:
            plot_path = os.path.join(tmpdirname, f"{filename}.png")

            plt.savefig(plot_path)  # Save the plot to the temporary directory
            # self.mlflow_client.log_artifact(tmpdirname)
            self.mlflow_client.log_artifacts(self.run_id,
                                             local_dir=tmpdirname)
        plt.close(fig)

    def plot_roc_curve_ovo(self, Y_ie, Y_pred_logits, filename=None):

        if isinstance(Y_pred_logits, list):
            Y_pred_logits = np.array(Y_pred_logits)

        classes_combinations = []

        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                classes_combinations.append([self.classes_names[i], self.classes_names[j]])
                classes_combinations.append([self.classes_names[j], self.classes_names[i]])

        # Plots the Probability Distributions and the ROC Curves One vs ONe
        fig = plt.figure(figsize=(30, 30))
        bins = [i / 20 for i in range(20)] + [1]

        roc_auc_ovo = {}

        for i in range(len(classes_combinations)):
            # Gets the class
            comb = classes_combinations[i]
            c1 = comb[0]
            c2 = comb[1]
            c1_index = self.classes_names.index(c1)
            title = c1 + " vs " + c2

            # Prepares an auxiliar dataframe to help with the plots
            df_aux = pd.DataFrame(Y_ie)

            df_aux['class'] = [self.classes_names_dict[y] for y in Y_ie]
            df_aux['prob'] = Y_pred_logits[:, c1_index]

            # Slices only the subset with both classes
            df_aux = df_aux[(df_aux['class'] == c1) | (df_aux['class'] == c2)]
            df_aux['class'] = [1 if y == c1 else 0 for y in df_aux['class']]
            df_aux = df_aux.reset_index(drop=True)

            if i < 6:
                # Plots the probability distribution for the class and the rest
                ax = plt.subplot(4, 6, i + 1)

                # Assuming df_aux has columns 'prob' and 'class'
                classes = df_aux['class'].unique()

                for cls in classes:
                    subset = df_aux[df_aux['class'] == cls]
                    ax.hist(subset['prob'], bins=bins, label=cls, alpha=0.6)  # alpha is for transparency

                ax.set_title(title)

                # c1_name = self.classes_names_dict[c1]
                # c2_name = self.classes_names_dict[c2]

                ax.legend([f"Class 1: {c1}", f"Class 0: {c2}"])
                ax.set_xlabel(f"P(x = {c1})")

                # Calculates the ROC Coordinates and plots the ROC Curves
                ax_bottom = plt.subplot(4, 6, i + 7)
                # tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
                tpr, fpr, _ = roc_curve(y_true=df_aux['class'], y_score=df_aux['prob'])
                plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
                ax_bottom.set_title("ROC Curve OvO")
            else:
                # Plots the probability distribution for the class and the rest
                ax = plt.subplot(4, 6, i + 7)

                # Assuming df_aux has columns 'prob' and 'class'
                classes = df_aux['class'].unique()

                for cls in classes:
                    subset = df_aux[df_aux['class'] == cls]
                    ax.hist(subset['prob'], bins=bins, label=cls, alpha=0.6)  # alpha is for transparency

                ax.set_title(title)
                ax.legend([f"Class 1: {c1}", f"Class 0: {c2}"])
                ax.set_xlabel(f"P(x = {c1})")
                # Calculates the ROC Coordinates and plots the ROC Curves
                ax_bottom = plt.subplot(4, 6, i + 13)
                # tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
                tpr, fpr, _ = roc_curve(y_true=df_aux['class'], y_score=df_aux['prob'])

                plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
                ax_bottom.set_title("ROC Curve OvO")

            # Calculates the ROC AUC OvO
            roc_auc_ovo[title] = roc_auc_score(df_aux['class'], df_aux['prob'])
        # plt.tight_layout()

        # Use a temporary directory to save the plot
        with tempfile.TemporaryDirectory() as tmpdirname:
            plot_path = os.path.join(tmpdirname, f"{filename}.png")

            plt.savefig(plot_path)  # Save the plot to the temporary directory
            # self.mlflow_client.log_artifact(tmpdirname)
            self.mlflow_client.log_artifacts(self.run_id,
                                             local_dir=tmpdirname)
        plt.close(fig)

    def plot_summary_advanced(self, all_results, image_name=None):

        # Constuct a single figure with appropriate layout for all metrics
        fig = plt.figure(layout="constrained")
        # ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, (3, 4))

        if self.num_classes == 2:
            # ConfusionMatrix and ROC we just plot the last step, notice how we call the plot method of those metrics
            # self.confmat.plot(val=all_results['BinaryConfusionMatrix'], ax=ax1)
            self.roc.plot(ax=ax2)

            scalar_results = {k: v[-1] for k, v in all_results.items() if isinstance(v, torch.Tensor)}

        if self.num_classes == 4:
            self.tracker.reset()
            # self.confmat.plot(val=all_results['ConfusionMatrix'], ax=ax1)
            self.roc.plot(ax=ax2)

            # For the remainig we plot the full history, but we need to extract the scalar values from the results
            scalar_results = {k: v[-1] for k, v in all_results.items() if isinstance(v, torch.Tensor)}

        self.tracker.plot(val=scalar_results, ax=ax3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            plot_path = os.path.join(tmpdirname, f"{image_name}.png")

            plt.savefig(plot_path)  # Save the plot to the temporary directory
            self.mlflow_client.log_artifacts(self.run_id, local_dir=tmpdirname)
            plt.close(fig)

    def init_metrics(self):

        if self.num_classes == 4:
            self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.train_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

            # Validation metric objects for calculating and averaging accuracy across batches
            self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.val_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

            # Testing metric objects for calculating and averaging accuracy across batches
            self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.test_f1 = F1Score(task="multiclass", num_classes=self.num_classes)

            # Define collection that is a mix of metrics that return a scalar tensors and not
            # self.confmat = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=self.num_classes)
            self.roc = torchmetrics.classification.MulticlassROC(num_classes=self.num_classes)
            self.mean_roc = MeanMetric()

            self.collection = torchmetrics.MetricCollection(
                torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes),
                torchmetrics.Recall(task="multiclass", num_classes=self.num_classes),
                torchmetrics.Precision(task="multiclass", num_classes=self.num_classes),
                # self.confmat,
            )

        if self.num_classes == 2:
            self.train_acc = Accuracy(task="binary")
            self.train_f1 = F1Score(task="binary", average="macro")

            self.val_acc = Accuracy(task="binary")
            self.val_f1 = F1Score(task="binary", average="macro")

            self.test_acc = Accuracy(task="binary")
            self.test_f1 = F1Score(task="binary")

            # Define collection that is a mix of metrics that return a scalar tensors and not
            # self.confmat = torchmetrics.ConfusionMatrix(task="binary")
            self.roc = torchmetrics.ROC(task="binary")
            self.collection = torchmetrics.MetricCollection(
                torchmetrics.Accuracy(task="binary"),
                torchmetrics.Recall(task="binary"),
                torchmetrics.Precision(task="binary"),
                # self.confmat,
            )

        # ATTRIBUTES TO SAVE BATCH OUTPUTS
        self.training_step_outputs = []  # save outputs in each batch to compute metric overall epoch
        self.training_step_targets = []  # save targets in each batch to compute metric overall epoch

        # ATTRIBUTES TO SAVE BATCH OUTPUTS
        self.val_step_outputs = []  # save outputs in each batch to compute metric overall epoch
        self.val_step_targets = []  # save targets in each batch to compute metric overall epoch

        # ATTRIBUTES TO SAVE BATCH OUTPUTS
        self.val_step_outputs_logits = []  # save outputs in each batch to compute metric overall epoch
        self.val_step_targets_ie = []  # save targets in each batch to compute metric overall epoch

        self.train_loss = MeanMetric()
        self.train_f1_best = MaxMetric()
        # Validation metric objects for calculating and averaging accuracy across batches
        self.val_loss = MeanMetric()
        # for tracking best so far validation accuracy (used for checkpointing/ early stopping / HPO)
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

        self.test_loss = MeanMetric()
        # Define tracker over the collection to easy keep track of the metrics over multiple steps
        self.tracker = torchmetrics.wrappers.MetricTracker(self.collection)
