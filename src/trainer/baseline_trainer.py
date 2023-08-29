import itertools
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pyrootutils
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from src.utils.hyperparam_opt import optimize_hyperparams
from src.utils.mlflow import fetch_logged_data
from src.utils.vis import compute_percentages_cm, calculate_cm_stats

log = utils.get_pylogger(__name__)


class SklearnTrainer(object):
    """ sklearn trainer class
    """

    def __init__(self,
                 cv,
                 datamodule,
                 logger=None,
                 hyperparameter_search=None):

        self.cv = cv

        self.logger = logger
        self.datamodule = datamodule

        self.hyperparameter_search = hyperparameter_search

    def search_hyperparams(self, pipeline, hparams):

        log.info("Loading data...")

        train_data, test_data = self.datamodule.prepare_dataloader()

        # compute classes weights for imbalanced dataset
        global_classes_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                                 classes=train_data.className,
                                                                                 y=train_data.y)

        self.classes_names = train_data.className

        # FIXME with multi run does it automatically?
        # self.optuna_search = optuna.integration.OptunaSearchCV(pipeline, self.hyperparameter_search)
        # self.optuna_search = pipeline

        mlflow.set_tracking_uri(Path(self.logger.mlflow.tracking_uri).as_uri())

        experiment_name = self.logger.mlflow.experiment_name

        run_name = "{}_{}".format(self.logger.mlflow.run_name,
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

            self.opt.fit(train_data, train_data.y)

            log.info("Best params: {}".format(self.opt.best_params_))
            best_model = self.opt.best_estimator_


            # test on forced trials data
            y_pred = best_model.predict(test_data)

            test_forced_acc = sklearn.metrics.accuracy_score(test_data.y, y_pred)
            test_forced_f1 = sklearn.metrics.f1_score(test_data.y, y_pred, average='weighted')
            test_forced_recall = sklearn.metrics.recall_score(test_data.y, y_pred, average='weighted')
            test_forced_precision = sklearn.metrics.precision_score(test_data.y, y_pred, average='weighted')

            mlflow.log_metric("test_forced_acc", test_forced_acc)
            mlflow.log_metric("test_forced_f1", test_forced_f1)
            mlflow.log_metric("test_forced_recall", test_forced_recall)
            mlflow.log_metric("test_forced_precision", test_forced_precision)

            cm_vali = confusion_matrix(test_data.y, y_pred)
            test_forced_cm_title = "cm_forced_trials"

            self.compute_confusion_matrix(conf_mat=cm_vali,
                                          title=test_forced_cm_title)

            # log dataset dict to mlflow parent run
            train_sessions_dict = self.datamodule.train_subjects_sessions_dict
            tmp_dict = {}
            for key, value in train_sessions_dict.items():
                tmp_dict[key] = list(value)

            parent_run_id = parent_run.info.run_id
            params, metrics, tags, artifacts = fetch_logged_data(parent_run_id)

            hparams.update(params)
            hparams.update(tags)
            hparams["global_classes_weights"] = {self.classes_names[i]: global_classes_weights[i] for i in
                                                 range(len(self.classes_names))}

            # Use a temporary directory to save
            with tempfile.TemporaryDirectory() as tmpdirname:
                description_path = os.path.join(tmpdirname, "data_description.json")
                with open(description_path, "w") as f:
                    json.dump(tmp_dict, f)

                hparam_path = os.path.join(tmpdirname, "Hparams.json")
                with open(hparam_path, "w") as f:
                    json.dump(hparams, f)

                mlflow.log_artifacts(tmpdirname)

        return metrics

    def train(self, pipeline, hparams):

        log.info("Loading data...")
        train_data, test_data = self.datamodule.prepare_dataloader()

        # compute classes weights for imbalanced dataset
        global_classes_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                                 classes=np.unique(train_data.y),
                                                                                 y=train_data.y)

        self.classes_names = train_data.className

        metrics = None

        if "mlflow" in self.logger:
            self.clf = pipeline
            num_folds = self.cv.n_splits

            log.info(f"Train with cross-validation with {num_folds} folds...")

            # initialize lists for storing results of confusion matrix to calculate mean std
            val_cm_list = []
            val_f1_list = []
            val_recall_list = []

            mlflow.set_tracking_uri(Path(self.logger.mlflow.tracking_uri).as_uri())

            experiment_name = self.logger.mlflow.experiment_name

            run_name = "{}_{}".format(self.logger.mlflow.run_name,
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

                for fold, (train_index, test_index) in enumerate(self.cv.split(train_data, train_data.y)):

                    foldNum = fold + 1

                    log.info("Starting fold {} of {}".format(foldNum, num_folds))

                    X_train, X_test = train_data[train_index], train_data[test_index]
                    y_train, y_test = train_data.y[train_index], train_data.y[test_index]

                    # compute classes weights for imbalanced dataset

                    fold_classes_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                                           classes=np.unique(y_train),
                                                                                           y=y_train)
                    cv_classes_weights_dict[f"fold_{foldNum}"] = {self.classes_names[i]: fold_classes_weights[i] for i
                                                                  in range(len(self.classes_names))}

                    mlflow_job_name = f"CV_{foldNum}_{experiment_name}_{run_name}"

                    with mlflow.start_run(experiment_id=experiment.experiment_id,
                                          run_name=mlflow_job_name,
                                          nested=True) as child_run:

                        # self.child_mlflow_client = mlflow.tracking.MlflowClient()
                        # self.child_run_id = child_run.info.run_id

                        self.clf.fit(X_train, y_train)

                        # test on validation data
                        y_pred = self.clf.predict(X_test)

                        f1_score_vali = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred, average="weighted")
                        mlflow.log_metric("vali_f1_score", f1_score_vali)
                        val_f1_list.append(f1_score_vali)

                        recall_score_vali = sklearn.metrics.recall_score(y_true=y_test, y_pred=y_pred,
                                                                         average="weighted")
                        mlflow.log_metric("vali_recall_score", recall_score_vali)
                        val_recall_list.append(recall_score_vali)

                        cm_vali = confusion_matrix(y_test, y_pred)
                        val_cm_list.append(compute_percentages_cm(cm_vali))
                        val_cm_title = "cm_validation_fold_{}".format(foldNum)
                        self.compute_confusion_matrix(cm_vali, title=val_cm_title)

                        # test on forced trials data
                        y_pred = self.clf.predict(test_data)
                        test_forced_acc = sklearn.metrics.accuracy_score(test_data.y, y_pred)
                        test_forced_f1 = sklearn.metrics.f1_score(test_data.y, y_pred, average='weighted')
                        test_forced_recall = sklearn.metrics.recall_score(test_data.y, y_pred, average='weighted')
                        test_forced_precision = sklearn.metrics.precision_score(test_data.y, y_pred, average='weighted')

                        mlflow.log_metric("test_forced_acc", test_forced_acc)
                        mlflow.log_metric("test_forced_f1", test_forced_f1)
                        mlflow.log_metric("test_forced_recall", test_forced_recall)
                        mlflow.log_metric("test_forced_precision", test_forced_precision)

                        cm_vali = confusion_matrix(test_data.y, y_pred)
                        test_forced_cm_title = "cm_forced_trials"

                        self.compute_confusion_matrix(conf_mat=cm_vali,
                                                      title=test_forced_cm_title)

                        # fetch logged data from child run
                        child_run_id = child_run.info.run_id

                        log.info(f"Train Fold {foldNum} with run {child_run_id} is completed!")

                    # log the mean confusion matrix to mlflow parent run
                    mean_f1_score = np.mean(val_f1_list)
                    mlflow.log_metric("vali_mean_f1_score", mean_f1_score)

                    self.compute_confusion_matrix(val_cm_list,
                                                  title="mean_cm_val",
                                                  type="mean")
                    # log the mean confusion matrix to mlflow parent run

                    # fetch logged data from parent run
                    parent_run_id = parent_run.info.run_id
                    params, metrics, tags, artifacts = fetch_logged_data(parent_run_id)

                    hparams.update(params)
                    hparams.update(tags)
                    hparams["global_classes_weights"] = {self.classes_names[i]: global_classes_weights[i] for i in
                                                         range(len(self.classes_names))}

                    hparams["cv_classes_weights"] = cv_classes_weights_dict

                    # load successfull loaded data sessions dict
                    hparams["loaded_subject_sessions_dict"] = self.datamodule.loaded_subjects_sessions

                    # log dataset dict to mlflow parent run
                    train_sessions_dict = self.datamodule.train_subjects_sessions_dict
                    tmp_dict = {}
                    for key, value in train_sessions_dict.items():
                        tmp_dict[key] = list(value)

                    # Use a temporary directory to save
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        description_path = os.path.join(tmpdirname, "data_description.json")
                        with open(description_path, "w") as f:
                            json.dump(tmp_dict, f)

                        hparam_path = os.path.join(tmpdirname, "Hparams.json")
                        with open(hparam_path, "w") as f:
                            json.dump(hparams, f)

                        mlflow.log_artifacts(tmpdirname)

                log.info(f"Training completed!")

        else:
            log.warning("Logger not found! Skipping hyperparameter logging...")
            score = self.clf.fit(train_data, train_data.y)
            return score

        return metrics

    def test(self, pipeline, datamodule):
        # test_data = datamodule.load_data(self.test_sessions, self.concatenate_subjects)
        # return pipeline.predict(test_data)
        return NotImplementedError

    def compute_confusion_matrix(self, conf_mat, title, type='standard'):
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
            figure = plt.figure(figsize=(9, 9))

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
                plt.text(j, i, labels[i, j], horizontalalignment="center", color=color, fontsize=15)

            plt.tight_layout()
            plt.ylabel('True label', fontsize=10)
            plt.xlabel('Predicted label', fontsize=10)

            # Use a temporary directory to save the plot
            with tempfile.TemporaryDirectory() as tmpdirname:
                plot_path = os.path.join(tmpdirname, f"{title}.png")

                plt.savefig(plot_path)  # Save the plot to the temporary directory
                mlflow.log_artifacts(tmpdirname)



        elif type == 'mean':

            # Fixme here I pass list of cm (better way)
            plt.rcParams["font.family"] = 'DejaVu Sans'
            figure = plt.figure(figsize=(9, 9))

            # Add values to the plot
            mean_cm_val, std_cm_val = calculate_cm_stats(cm_list=conf_mat, num_classes=len(self.classes_names))

            plt.imshow(mean_cm_val, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()
            tick_marks = np.arange(len(self.classes_names))
            plt.xticks(tick_marks, self.classes_names, rotation=45)
            plt.yticks(tick_marks, self.classes_names)
            # set title
            plt.title(title)

            labels = [f"{v1 * 100:.4}%  ±{v2 * 100: .4}%" for v1, v2 in
                      zip(mean_cm_val.flatten(), std_cm_val.flatten())]
            labels = np.asarray(labels).reshape(len(self.classes_names), len(self.classes_names))

            # thresh = mean_cm_val.max() / 2.0
            for i, j in itertools.product(range(mean_cm_val.shape[0]), range(mean_cm_val.shape[1])):
                color = "red"
                plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

            plt.tight_layout()
            plt.ylabel('True label', fontsize=10)
            plt.xlabel('Predicted label', fontsize=10)

            # Use a temporary directory to save the plot
            with tempfile.TemporaryDirectory() as tmpdirname:
                plot_path = os.path.join(tmpdirname, f"{title}.png")

                plt.savefig(plot_path)  # Save the plot to the temporary directory
                mlflow.log_artifacts(tmpdirname)
