import itertools
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pyrootutils
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from src.utils.srm_utils import train_valid_split
from src.utils.hyperparam_opt import optimize_hyperparams
from src.utils.mlflow import fetch_logged_data
from src.utils.vis import compute_percentages_cm, calculate_cm_stats
from src.utils.file_mgmt import default
from src.utils.device import print_memory_usage, print_cpu_cores

log = utils.get_pylogger(__name__)


class SklearnTrainer(object):
    """ sklearn trainer class
    """

    def __init__(self,
                 cv,
                 train_val_split,
                 datamodule,
                 logger=None,
                 hyperparameter_search=None):

        self.cv = cv

        if train_val_split:
            self.train_val_split = dict(train_val_split)
        else:
            self.train_val_split = None

        self.logger = logger
        self.datamodule = datamodule

        self.hyperparameter_search = hyperparameter_search
        self.best_params = None

        log.info("Loading data...")
        self.datamodule.prepare_dataloader()

        self.train_data_list = self.datamodule.train_data_list
        self.test_data = self.datamodule.test_data

        self.classes_names = self.datamodule.classes
        self.task_name = self.datamodule.task_name

        self.classes_names_dict = {}
        if self.task_name == "LR":
            self.classes_names_dict = {0: "R", 1: "L"}
        if self.task_name == "2D":
            self.classes_names_dict = {0: "R", 1: "L", 2: "U", 3: "D"}

        log.info("Resolving device...")
        print_memory_usage()
        print_cpu_cores()

    def search_hyperparams(self, pipeline, hparams):

        mlflow.set_tracking_uri(Path(self.logger.mlflow.tracking_uri).as_uri())

        experiment_name = self.logger.mlflow.experiment_name

        run_name = "HPO_{}_{}".format(self.logger.mlflow.run_name,
                                      datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        log.info("Create experiment: {}".format(experiment_name))
        try:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)

        mlflow.sklearn.autolog()

        log.info("Mlflow initialized! Logging to mlflow...")

        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as parent_run:

            self.opt = optimize_hyperparams(self.hyperparameter_search, pipeline)

            self.opt.fit(self.train_data, self.train_data.y)

            self.best_params = self.opt.best_params_

            log.info("Best params: {}".format(self.best_params))
            best_model = self.opt.best_estimator_

            # test on data "valid" / forced trials
            self.predict_test_data(best_model, self.test_data, test_data_type="valid")

            # log dataset dict to mlflow parent run
            _, metrics, _, _ = fetch_logged_data(parent_run.info.run_id)
            self.log_to_mlflow(hparams, parent_run, self.train_data, self.test_data)

            log.info("Hyperparameter search completed!")

            # clean up
            # gc.collect()

        return metrics
    
    def hpo_optimize_optuna(self, pipeline, hparams):
        """Train the model. using Optuna"""
        pass

    def train_baseline_cv(self, pipeline, hparams):
        """Train the model. using cross validation"""

        # # FIXME: New data split
        # if self.train_val_split is not None:
        #     train_data, val_data = train_valid_split(self.datamodule.train_data, self.train_val_split)

        # compute classes weights for imbalanced dataset
        # global_classes_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
        #                                                                          classes=np.unique(train_data.y),
        #                                                                          y=train_data.y)

        if self.best_params is not None:
            for param_name in self.best_params:
                step, param = param_name.split('__', 1)
                pipeline.named_steps[step].set_params(**{param: self.best_params[param_name]})

        self.clf = pipeline

        if "mlflow" in self.logger:

            num_folds = self.cv.n_splits
            log.info(f"Train with cross-validation with {num_folds} folds...")

            # initialize training lists for storing results of confusion matrix to calculate mean std
            train_cm_list = []
            # initialize validation lists for storing results of confusion matrix to calculate mean std
            val_cm_list = []
            val_f1_list = []
            val_acc_list = []
            val_recall_list = []
            # initialize test lists for storing results of confusion matrix to calculate mean std
            test_cm_list = []
            test_f1_list = []
            test_acc_list = []
            test_recall_list = []

            cv_metrics_dict = {}

            # roc curve over folds
            self.train_roc_curve_list = []
            self.val_roc_curve_list = []
            self.test_roc_curve_list = []

            # roc curve over folds for validation data
            val_tprs = []
            val_aucs = []

            # roc curve over folds for test data
            test_tprs = []
            test_aucs = []
            num_smaples_test = self.test_data.y.shape[0]
            test_mean_fpr = np.linspace(0, 1, num_smaples_test)

            mlflow.set_tracking_uri(Path(self.logger.mlflow.tracking_uri).as_uri())

            experiment_name = self.logger.mlflow.experiment_name

            run_name = "T_{}_{}".format(self.logger.mlflow.run_name,
                                        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

            log.info("Create experiment: {}".format(experiment_name))
            try:
                mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
                experiment = mlflow.get_experiment_by_name(experiment_name)
            except:
                experiment = mlflow.get_experiment_by_name(experiment_name)

            mlflow.sklearn.autolog()
            log.info("Mlflow initialized! Logging to mlflow...")
            with mlflow.start_run(experiment_id=experiment.experiment_id,
                                  run_name=run_name) as parent_run:

                cv_classes_weights_dict = {}
                model_name = self.logger.mlflow.run_name

                for fold, (train_index, vali_index) in enumerate(self.cv.split(self.train_data, self.train_data.y)):
                    foldNum = fold + 1

                    log.info("Starting fold {} of {}".format(foldNum, num_folds))

                    X_train, X_vali = self.train_data[train_index], self.train_data[vali_index]
                    y_train, y_vali = self.train_data.y[train_index], self.train_data.y[vali_index]

                    num_smaples_val = y_vali.shape[0]
                    val_mean_fpr = np.linspace(0, 1, num_smaples_val)

                    # compute classes weights for imbalanced dataset

                    # fold_classes_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                    #                                                                        classes=np.unique(y_train),
                    #                                                                        y=y_train)
                    # cv_classes_weights_dict[f"fold_{foldNum}"] = {self.classes_names[i]: fold_classes_weights[i] for i
                    #                                               in range(len(self.classes_names))}

                    mlflow_job_name = f"CV_{foldNum}_{experiment_name}_{run_name}"

                    with mlflow.start_run(experiment_id=experiment.experiment_id,
                                          run_name=mlflow_job_name,
                                          nested=True) as child_run:
                        # self.child_mlflow_client = mlflow.tracking.MlflowClient()
                        # self.child_run_id = child_run.info.run_id

                        self.clf.fit(X_train, y_train)

                        # train confusion matrix & metrics
                        y_pred = self.clf.predict(X_train)

                        cm_train = confusion_matrix(y_true=y_train, y_pred=y_pred,
                                                    labels=list(self.classes_names_dict.keys()))
                        train_cm_list.append(compute_percentages_cm(cm_train))

                        # validation confusion matrix & metrics
                        y_pred = self.clf.predict(X_vali)
                        cm_vali = confusion_matrix(y_true=y_vali, y_pred=y_pred,
                                                   labels=list(self.classes_names_dict.keys()))

                        self.plot_confusion_matrix(cm_vali, title="cm_validation_fold_{}".format(foldNum))
                        val_cm_list.append(compute_percentages_cm(cm_vali))

                        vali_metrics_dict = self.compute_metrics(y_true=y_vali, y_pred=y_pred, set_name="val")

                        val_f1_list.append(vali_metrics_dict["f1_score"])
                        val_acc_list.append(vali_metrics_dict["acc"])
                        val_recall_list.append(vali_metrics_dict["recall_score"])

                        # test confusion matrix & metrics
                        test_set_name = "test_valid"
                        y_pred = self.clf.predict(self.test_data)
                        cm_test = confusion_matrix(y_true=self.test_data.y, y_pred=y_pred,
                                                   labels=list(self.classes_names_dict.keys()))

                        self.plot_confusion_matrix(conf_mat=cm_test,
                                                   title="cm_{}_fold_{}".format(test_set_name, foldNum))
                        test_cm_list.append(compute_percentages_cm(cm_test))

                        test_metrics_dict = self.compute_metrics(y_true=self.test_data.y, y_pred=y_pred,
                                                                 set_name=test_set_name)

                        test_f1_list.append(test_metrics_dict["f1_score"])
                        test_acc_list.append(test_metrics_dict["acc"])
                        test_recall_list.append(test_metrics_dict["recall_score"])

                        # Roc curve / average Roc curve
                        if self.task_name == "LR":
                            # Train data metrics
                            y_pred = self.clf.predict(X_train)
                            self.compute_roc_curve(y_true=y_train, y_pred=y_pred, set_name="train")

                            # compute metrics on valid data
                            y_pred = self.clf.predict(X_vali)
                            self.compute_roc_curve(y_true=y_vali, y_pred=y_pred, set_name="val")
                            fpr, tpr, _ = roc_curve(y_true=y_vali, y_score=y_pred)
                            val_tprs.append(np.interp(val_mean_fpr, fpr, tpr))
                            val_tprs[-1][0] = 0.0  # set first value to 0 to have a better plot
                            roc_auc = sklearn.metrics.auc(fpr, tpr)
                            val_aucs.append(roc_auc)

                            # compute metrics on Test data
                            y_pred = self.clf.predict(self.test_data)
                            self.compute_roc_curve(y_true=self.test_data.y, y_pred=y_pred, set_name="test")
                            fpr, tpr, _ = roc_curve(y_true=self.test_data.y, y_score=y_pred)
                            test_tprs.append(np.interp(test_mean_fpr, fpr, tpr))
                            test_tprs[-1][0] = 0.0  # set first value to 0 to have a better plot
                            roc_auc = sklearn.metrics.auc(fpr, tpr)
                            test_aucs.append(roc_auc)

                        if self.task_name == "2D":
                            # ovo and ovr roc curve on validation data
                            self.predict_prob(X_vali, y_vali, foldNum=foldNum, test_data_type="vali")

                            # ovo and ovr roc curve on test data
                            self.predict_prob(self.test_data, self.test_data.y, foldNum=foldNum, test_data_type="test")

                        # fetch logged data from child run
                        child_run_id = child_run.info.run_id
                        _, metrics, _, _ = fetch_logged_data(child_run_id)
                        cv_metrics_dict[f"fold_{foldNum}"] = metrics

                        log.info(f"Train Fold {foldNum} with run {child_run_id} is completed!")

                log.info(f"Training completed!, Computing mean and std of metrics...")
                # log the train mean confusion matrix to mlflow parent run
                self.plot_confusion_matrix(train_cm_list,
                                           title=f"cv_train-avg-cm_{model_name}",
                                           type="mean")

                # log the validation mean confusion matrix to mlflow parent run
                mean_f1_score_vali = np.mean(val_f1_list)
                mean_acc_score_vali = np.mean(val_acc_list)
                mlflow.log_metric("mean_val/f1", mean_f1_score_vali)
                mlflow.log_metric("mean_val/acc", mean_acc_score_vali)
                self.plot_confusion_matrix(val_cm_list,
                                           title=f"cv_val-avg-cm_{model_name}",
                                           type="mean")

                # log the test mean confusion matrix to mlflow parent run
                mean_f1_score_test = np.mean(test_f1_list)
                mean_acc_score_test = np.mean(test_acc_list)
                mlflow.log_metric(f"mean_test/f1", mean_f1_score_test)
                mlflow.log_metric(f"mean_test/acc", mean_acc_score_test)
                self.plot_confusion_matrix(test_cm_list,
                                           title=f"cv_{test_set_name}-avg-cm_{model_name}",
                                           type="mean")

                if self.task_name == "LR":
                    self.plot_roc_curve(tprs=val_tprs,
                                        aucs=val_aucs,
                                        mean_fpr=val_mean_fpr,
                                        set_name="vali",
                                        title="Cross-validation_roc-curve_validation")

                    self.plot_roc_curve(tprs=test_tprs,
                                        aucs=test_aucs,
                                        mean_fpr=test_mean_fpr,
                                        set_name="test",
                                        title="Cross-validation_roc-curve_test")

                # fetch logged data from parent run
                # parent_run_id = parent_run.info.run_id

                _, mean_metrics, _, _ = fetch_logged_data(parent_run.info.run_id)

                metrics = {**mean_metrics, **cv_metrics_dict}

                # self.log_to_mlflow(hparams, child_run, train_data, test_data, global_classes_weights,
                #                    cv_classes_weights_dict)
                self.log_to_mlflow(hparams, child_run, self.train_data, self.test_data)

                log.info(f"Training completed!")

        return metrics

    def train_baseline(self, pipeline, hparams):
        """Train the model. using standard train test split"""

        if self.best_params is not None:
            for param_name in self.best_params:
                step, param = param_name.split('__', 1)
                pipeline.named_steps[step].set_params(**{param: self.best_params[param_name]})

        self.clf = pipeline

        train_runs_list, val_runs_list = train_valid_split(self.train_data_list,
                                                           val_ratio=self.train_val_split["val_ratio"],
                                                           random_seed=self.train_val_split["random_seed"])
        self.train_data = train_runs_list[0]
        self.val_data = val_runs_list[0]
        for i in range(1, 5):
            self.train_data = self.train_data.append(train_runs_list[i], axis=0)
            self.val_data = self.val_data.append(val_runs_list[i], axis=0)

        mlflow.set_tracking_uri(Path(self.logger.mlflow.tracking_uri).as_uri())

        experiment_name = self.logger.mlflow.experiment_name

        run_name = "T_{}_{}".format(self.logger.mlflow.run_name,
                                    datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        log.info("Create experiment: {}".format(experiment_name))
        try:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)

        mlflow.sklearn.autolog()

        log.info("Mlflow initialized! Logging to mlflow...")
        with mlflow.start_run(experiment_id=experiment.experiment_id,
                              run_name=run_name) as parent_run:

            model_name = self.logger.mlflow.run_name

            self.clf.fit(self.train_data, self.train_data.y)

            # compute metrics on valid data
            y_pred_vali = self.clf.predict(self.val_data)
            cm_vali = confusion_matrix(y_true=self.val_data.y, y_pred=y_pred_vali,
                                       labels=list(self.classes_names_dict.keys()))
            self.plot_confusion_matrix(cm_vali, title="cm_validation")
            self.compute_metrics(y_true=self.val_data.y, y_pred=y_pred_vali, set_name="val")

            # compute metrics on Test data
            y_pred_test = self.clf.predict(self.test_data)
            cm_test = confusion_matrix(y_true=self.test_data.y, y_pred=y_pred_test,
                                       labels=list(self.classes_names_dict.keys()))
            self.plot_confusion_matrix(conf_mat=cm_test, title="cm_test")
            self.compute_metrics(y_true=self.test_data.y, y_pred=y_pred_test, set_name="test")

            # ovo and ovr roc curve on validation data
            self.predict_prob(X=self.val_data, y=self.val_data.y, data_type="val")

            # ovo and ovr roc curve on test data
            self.predict_prob(self.test_data, self.test_data.y, data_type="test")

            _, metrics, _, _ = fetch_logged_data(parent_run.info.run_id)
            self.log_to_mlflow(hparams, parent_run, self.train_data, self.test_data)
            log.info(f"Training completed!")

        return metrics

    def predict_prob(self, X, y, foldNum=None, data_type="val"):
        """Predict probabilities for each class for each sample."""

        try:
            y_pred_proba = self.clf.predict_proba(X)
        except AttributeError:
            try:
                y_pred_proba = self.clf.predict_log_proba(X)
            except AttributeError:
                raise ValueError(
                    "Neither predict_proba nor predict_log_proba are available for the given classifier.")

        if y_pred_proba.shape[1] != len(self.classes_names_dict):
            # Number of samples
            n_samples = y_pred_proba.shape[0]
            # Create an empty array to hold the adjusted probabilities
            adjusted_probs = np.zeros((n_samples, len(self.classes_names_dict)))

            # Fill the adjusted predict_proba output
            for idx, class_label in self.classes_names_dict.items():
                if idx in self.clf.classes_:
                    # Find the index of the current class in the trained classes
                    trained_class_idx = list(self.clf.classes_).index(idx)
                    # Assign the corresponding probabilities
                    adjusted_probs[:, idx] = y_pred_proba[:, trained_class_idx]
        else:
            adjusted_probs = y_pred_proba

        if foldNum is not None:
            roc_curve_ovr_title = f"roc-ovr_{data_type}_fold-{foldNum}"
            roc_curve_ovo_title = f"roc-ovo_{data_type}_fold-{foldNum}"
        else:
            roc_curve_ovr_title = f"roc-ovr_{data_type}"
            roc_curve_ovo_title = f"roc-ovo_{data_type}"

        self.plot_roc_curve_ovr(Y_ie=y, Y_pred_logits=adjusted_probs,
                                filename=roc_curve_ovr_title)

        self.plot_roc_curve_ovo(Y_ie=y, Y_pred_logits=adjusted_probs,
                                filename=roc_curve_ovo_title)

    def compute_metrics(self, y_true, y_pred, set_name="vali"):
        """Compute metrics for classification task.
        :param y_true: true labels
        :param y_pred: predicted labels
        :return: dict with metrics
        """
        # compute metrics
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        mlflow.log_metric(f"{set_name}/acc", acc)

        f1_score = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
        mlflow.log_metric(f"{set_name}/f1", f1_score)

        recall_score = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average="weighted")
        mlflow.log_metric(f"{set_name}/recall", recall_score)

        precision_score = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average="weighted")
        mlflow.log_metric(f"{set_name}/precision", precision_score)

        return {"acc": acc,
                "f1_score": f1_score,
                "recall_score": recall_score,
                "precision_score": precision_score}

    def compute_roc_curve(self, y_true, y_pred, set_name="val"):
        """Compute roc curve for classification task.
        :param y_true: true labels
        :param y_pred: predicted labels
        :return: dict with metrics
        """
        # compute metrics
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_pred)
        if set_name == "train":
            self.train_roc_curve_list.append((fpr, tpr))
        elif set_name == "val":
            self.val_roc_curve_list.append((fpr, tpr))
        elif set_name == "test":
            self.test_roc_curve_list.append((fpr, tpr))
        else:
            raise ValueError(f"set_name {set_name} not valid")

    def plot_confusion_matrix(self, conf_mat, title, type='standard'):
        """Compute and plot the confusion matrix. It calculates a standard confusion matrix or
            the mean and std of a list of confusion matrix used for cross validation folds.
            Type can be 'standard' or 'mean' , default is 'standard'
        :param conf_mat: confusion matrix
        :param title: title of the plot
        :param type: type of the plot
        :return: None
        """

        if type == 'standard':
            plt.rcParams["font.family"] = 'DejaVu Sans'
            plt.figure(figsize=(9, 9))

            plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()
            tick_marks = np.arange(len(self.classes_names))
            plt.xticks(tick_marks, self.classes_names)
            plt.yticks(tick_marks, self.classes_names)
            # set title
            plt.title(title)

            # render the confusion matrix with percentage and ratio.
            group_counts = []
            for i in range(len(self.classes_names)):
                for j in range(len(self.classes_names)):
                    group_counts.append("{}/{}".format(conf_mat[i, j], conf_mat.sum(axis=1)[i]))

            group_percentages = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals=2)
            labels = [f"{v1} \n {v2 * 100: .4}%" for v1, v2 in zip(group_counts, group_percentages.flatten())]
            labels = np.asarray(labels).reshape(len(self.classes_names), len(self.classes_names))

            # set the font size of the text in the confusion matrix
            for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
                color = "red"
                plt.text(j, i, labels[i, j], horizontalalignment="center", color=color, fontsize=12)

            plt.tight_layout()
            plt.ylabel('True label', fontsize=10)
            plt.xlabel('Predicted label', fontsize=10)

            # Use a temporary directory to save the plot
            with tempfile.TemporaryDirectory() as tmpdirname:
                plot_path = os.path.join(tmpdirname, f"{title}.png")

                plt.savefig(plot_path)  # Save the plot to the temporary directory
                mlflow.log_artifacts(tmpdirname)

        if type == 'mean':

            # Fixme here I pass list of cm (better way)
            plt.rcParams["font.family"] = 'DejaVu Sans'
            plt.figure(figsize=(9, 9))

            # Add values to the plot
            mean_cm_val, std_cm_val = calculate_cm_stats(cm_list=conf_mat, num_classes=len(self.classes_names))

            plt.imshow(mean_cm_val, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()
            tick_marks = np.arange(len(self.classes_names))
            plt.xticks(tick_marks, self.classes_names, rotation=0)
            plt.yticks(tick_marks, self.classes_names)
            # set title
            plt.title(title)

            labels = [f"{v1 * 100:.4}%  ±{v2 * 100: .4}%" for v1, v2 in
                      zip(mean_cm_val.flatten(), std_cm_val.flatten())]
            labels = np.asarray(labels).reshape(len(self.classes_names), len(self.classes_names))

            # thresh = mean_cm_val.max() / 2.0
            for i, j in itertools.product(range(mean_cm_val.shape[0]), range(mean_cm_val.shape[1])):
                color = "red"
                plt.text(j, i, labels[i, j], horizontalalignment="center", color=color, fontsize=10)

            plt.tight_layout()
            plt.ylabel('True label', fontsize=10)
            plt.xlabel('Predicted label', fontsize=10)

            # Use a temporary directory to save the plot
            with tempfile.TemporaryDirectory() as tmpdirname:
                plot_path = os.path.join(tmpdirname, f"{title}.png")

                plt.savefig(plot_path)  # Save the plot to the temporary directory
                mlflow.log_artifacts(tmpdirname)
        plt.close()

    def plot_roc_curve(self, tprs, aucs, mean_fpr, set_name, title):
        """Compute and plot the roc curve. It calculates a standard roc curve or the mean and std of a list of roc curve
            used for cross validation folds. """

        plt.rcParams["font.family"] = 'DejaVu Sans'
        plt.figure(figsize=(10, 7))

        # TODO: remove if not needed
        if set_name == "val":
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
        plt.close()

    def plot_roc_curve_ovr(self, Y_ie, Y_pred_logits, filename=None):

        fig = plt.figure(figsize=(16, 10))
        bins = [i / 20 for i in range(20)] + [1]

        num_classes = len(self.classes_names)

        for i, (k, v) in enumerate(self.classes_names_dict.items()):
            # Gets the class
            # c = k
            # Prepares an auxiliar dataframe to help with the plots
            df_aux = pd.DataFrame(Y_ie)

            df_aux['class'] = [1 if y == k else 0 for y in Y_ie]
            df_aux['prob'] = Y_pred_logits[:, i]
            df_aux = df_aux.reset_index(drop=True)

            # Plots the probability distribution for the class and the rest
            ax = plt.subplot(2, num_classes, i + 1)

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
            ax_bottom = plt.subplot(2, num_classes, i + (num_classes + 1))
            # tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
            tpr, fpr, _ = roc_curve(y_true=df_aux['class'], y_score=df_aux['prob'])
            plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
            ax_bottom.set_title("ROC Curve OvR")

        # plt.tight_layout()

        # Use a temporary directory to save the plot
        with tempfile.TemporaryDirectory() as tmpdirname:
            plot_path = os.path.join(tmpdirname, f"{filename}.png")
            plt.savefig(plot_path)  # Save the plot to the temporary directory
            mlflow.log_artifacts(tmpdirname)
        plt.close()

    def plot_roc_curve_ovo(self, Y_ie, Y_pred_logits, filename=None):

        classes_combinations = []

        num_classes = len(self.classes_names)

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                classes_combinations.append([self.classes_names[i], self.classes_names[j]])
                classes_combinations.append([self.classes_names[j], self.classes_names[i]])

        # Plots the Probability Distributions and the ROC Curves One vs ONe
        fig = plt.figure(figsize=(30, 30))
        bins = [i / 20 for i in range(20)] + [1]

        roc_auc_ovo = {}

        # class_names = {v: k for k, v in self.classes_names_dict.items()}

        for i in range(len(classes_combinations)):
            # Gets the class
            comb = classes_combinations[i]
            c1 = comb[0]
            c2 = comb[1]
            c1_index = self.classes_names.index(c1)
            title = c1 + " vs " + c2

            # Prepares an auxiliar dataframe to help with the plots

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
            # roc_auc_ovo[title] = roc_auc_score(df_aux['class'], df_aux['prob'], multi_class='ovo')
            roc_auc_ovo[title] = roc_auc_score(df_aux['class'], df_aux['prob'])
            # plt.tight_layout()

        # Use a temporary directory to save the plot
        with tempfile.TemporaryDirectory() as tmpdirname:
            plot_path = os.path.join(tmpdirname, f"{filename}.png")
            plt.savefig(plot_path)  # Save the plot to the temporary directory
            mlflow.log_artifacts(tmpdirname)
        plt.close()

    def log_to_mlflow(self,
                      hparams,
                      mlflow_run,
                      train_data,
                      test_data,
                      global_classes_weights=None,
                      cv_classes_weights_dict=None):

        params, _, _, _ = fetch_logged_data(mlflow_run.info.run_id)
        with tempfile.TemporaryDirectory() as tmpdirname:
            hparam_path = os.path.join(tmpdirname, "pipeline_params.json")
            with open(hparam_path, "w") as f:
                json.dump(params, f, default=default)
            mlflow.log_artifacts(tmpdirname)

        if global_classes_weights is not None:
            hparams["global_classes_weights"] = {self.classes_names[i]: global_classes_weights[i] for i in
                                                 range(len(self.classes_names))}

        if cv_classes_weights_dict is not None:
            hparams["cv_classes_weights"] = cv_classes_weights_dict

        hparams["train_data_shape"] = train_data.shape
        hparams["test_data_shape"] = test_data.shape

        # load successfully loaded data sessions dict
        with tempfile.TemporaryDirectory() as tmpdirname:
            hparam_path = os.path.join(tmpdirname, "dataset_info.json")
            with open(hparam_path, "w") as f:
                json.dump(hparams, f, default=default)

            mlflow.log_artifacts(tmpdirname)

        for subject_name, subject_info_dict in self.datamodule.subjects_info_dict.items():
            if self.datamodule.loading_data_mode != "cross_subject_hpo":
                mlflow.log_param("pvc", subject_info_dict["pvc"])
            # Use a temporary directory to save
            with tempfile.TemporaryDirectory() as tmpdirname:
                hparam_path = os.path.join(tmpdirname, f"{subject_name}_info.json")
                with open(hparam_path, "w") as f:
                    json.dump(subject_info_dict, f, default=default)

                mlflow.log_artifacts(tmpdirname)


def plot_roc_curve(tpr, fpr, scatter=False, ax=None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).

    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
        ax: Matplotlib axis object. If None, a new figure and axis will be created.
    '''
    if ax == None:
        _, ax = plt.subplots(figsize=(5, 5))

    if scatter:
        ax.scatter(fpr, tpr)

    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], color='green', linestyle='--', label="Random classifier")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    # plt.show()


# TODO plot csp filter and csp patterns
def csp_filters(clf, datamodule, num_filters=4, title="csp_filters"):
    # get csp filters
    A = clf.csf["csp"].A
    W = clf.csf["csp"].W
    d = clf.csf["csp"].d
    selected_cmps = clf["csp"].selected_cmps

    pass
