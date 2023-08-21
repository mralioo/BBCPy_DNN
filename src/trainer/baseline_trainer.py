import json
from datetime import datetime
from pathlib import Path
import os

import mlflow
import numpy as np
import pyrootutils
import sklearn
from sklearn.metrics import confusion_matrix

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from src.utils.hyperparam_opt import optimize_hyperparams
from src.utils.mlflow import fetch_logged_data
from src.utils.vis import compute_percentages_cm, confusion_matrix_to_png

log = utils.get_pylogger(__name__)


class SklearnTrainer(object):
    def __init__(self,
                 cv,
                 datamodule,
                 logger=None,
                 hyperparameter_search=None):

        self.cv = cv

        self.logger = logger
        self.datamodule = datamodule

        self.hyperparameter_search = hyperparameter_search

        # self.train_sessions = self.datamodule.train_subjects_sessions_dict
        # if self.datamodule.test_subjects_sessions_dict is not None:
        #     self.test_sessions = self.datamodule.test_subjects_sessions_dict
        # self.concatenate_subjects = self.datamodule.concatenate_subjects

    def search_hyperparams(self, pipeline, hparams):

        log.info("Loading data...")

        train_sessions_dict = self.datamodule.train_subjects_sessions_dict
        train_data = self.datamodule.train_dataloader()

        vali_data = self.datamodule.train_dataloader()

        classes_names = train_data.className

        # FIXME with multi run does it automatically?
        # self.optuna_search = optuna.integration.OptunaSearchCV(pipeline, self.hyperparameter_search)
        # self.optuna_search = pipeline

        log.info("Logging to mlflow...")
        mlflow.set_tracking_uri(self.logger.mlflow.tracking_uri.as_uri())

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

        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as parent_run:

            self.opt = optimize_hyperparams(self.hyperparameter_search, pipeline)

            self.opt.fit(train_data, train_data.y)

            log.info("Best params: {}".format(self.opt.best_params_))
            best_model = self.opt.best_estimator_
            y_pred = best_model.predict(vali_data)

            val_acc = sklearn.metrics.accuracy_score(vali_data.y, y_pred)
            val_f1 = sklearn.metrics.f1_score(vali_data.y, y_pred, average='weighted')
            val_recall = sklearn.metrics.recall_score(vali_data.y, y_pred, average='weighted')
            val_precision = sklearn.metrics.precision_score(vali_data.y, y_pred, average='weighted')

            mlflow.log_metric("val_acc", val_acc)
            mlflow.log_metric("val_f1", val_f1)
            mlflow.log_metric("val_recall", val_recall)
            mlflow.log_metric("val_precision", val_precision)

            cm_vali = confusion_matrix(vali_data.y, y_pred)
            val_cm_title = f"best_model_val-f1_score-{val_f1}"
            confusion_matrix_to_png(cm_vali, classes_names, val_cm_title, figure_file_name="best_model_vali_cm")

            with open("Hparams.json", "w") as f:
                json.dump(hparams, f)
            mlflow.log_artifact("Hparams.json", artifact_path="best_model")

            parent_run_id = parent_run.info.run_id
            params, metrics, tags, artifacts = fetch_logged_data(parent_run_id)

        return metrics

    def train(self, pipeline, hparams):

        log.info("Loading data...")
        train_data = self.datamodule.train_dataloader()
        classes_names = train_data.className

        metrics = None

        if "mlflow" in self.logger:
            self.clf = pipeline
            num_folds = self.cv.n_splits

            log.info(f"Train with cross-validation with {num_folds} folds...")

            # initialize lists for storing results of confusion matrix to calculate mean std
            val_cm_list = []
            val_f1_list = []
            val_recall_list = []

            log.info("Logging to mlflow...")
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

            with mlflow.start_run(experiment_id=experiment.experiment_id,
                                  run_name=run_name) as parent_run:

                for fold, (train_index, test_index) in enumerate(self.cv.split(train_data, train_data.y)):

                    foldNum = fold + 1

                    log.info("Starting fold {} of {}".format(foldNum, num_folds))

                    X_train, X_test = train_data[train_index], train_data[test_index]
                    y_train, y_test = train_data.y[train_index], train_data.y[test_index]

                    parent_run_id = parent_run.info.run_id
                    mlflow_job_name = f"CV_{foldNum}_{experiment_name}_{run_name}"

                    with mlflow.start_run(experiment_id=experiment.experiment_id,
                                          run_name=mlflow_job_name,
                                          nested=True) as child_run:


                        self.clf.fit(X_train, y_train)
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
                        val_cm_title = f"vali-foldNum-{foldNum}-f1_score-{f1_score_vali}"
                        confusion_matrix_to_png(cm_vali,
                                                classes_names,
                                                val_cm_title,
                                                figure_file_name="vali_cm")

                        # fetch logged data from child run
                        child_run_id = child_run.info.run_id

                        log.info(f"Train Fold {foldNum} with run {child_run_id} is completed!")

                    # log the mean confusion matrix to mlflow parent run
                    mean_f1_score = np.mean(val_f1_list)
                    mlflow.log_metric("vali_mean_f1_score", mean_f1_score)
                    confusion_matrix_to_png(val_cm_list,
                                            classes_names,
                                            "mean_cm_val",
                                            type="mean")
                    # log the mean confusion matrix to mlflow parent run

                    # fetch logged data from parent run
                    parent_run_id = parent_run.info.run_id
                    params, metrics, tags, artifacts = fetch_logged_data(parent_run_id)

                    hparams.update(params)
                    hparams.update(tags)

                    mlflow_artifact_path = mlflow.get_artifact_uri().replace("file:///", "")
                    hparams_file_path = os.path.join(mlflow_artifact_path, "Hparams.json")
                    hparams["mlflow_uri"] = mlflow_artifact_path
                    with open(hparams_file_path, "w") as f:
                        json.dump(hparams, f)
                    mlflow.log_artifact(hparams_file_path, artifact_path="best_model")

                    # log dataset dict to mlflow parent run
                    train_sessions_dict = self.datamodule.train_subjects_sessions_dict
                    tmp_dict = {}
                    for key, value in train_sessions_dict.items():
                        tmp_dict[key] = list(value)

                    mlflow_artifact_path = mlflow.get_artifact_uri().replace("file:///", "")
                    data_description_file_path = os.path.join(mlflow_artifact_path, "data_description.json")

                    with open(data_description_file_path, "w") as f:
                        json.dump(tmp_dict, f)

                    mlflow.log_artifact(data_description_file_path)

                log.info(f"Training completed!")

        else:
            log.warning("Logger not found! Skipping hyperparameter logging...")
            score = self.clf.fit(train_data, train_data.y)
            return score

        return metrics

    def test(self, pipeline, datamodule):
        return NotImplementedError

        # test_data = datamodule.load_data(self.test_sessions, self.concatenate_subjects)
        #
        # return pipeline.predict(test_data)
