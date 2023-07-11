import json
from datetime import datetime
import numpy as np
import itertools

import mlflow
import sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import confusion_matrix
from pprint import pprint

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src import utils
from src.utils.vis import calculate_cm_stats, compute_percentages_cm
import matplotlib.pyplot as plt

log = utils.get_pylogger(__name__)


def confusion_matrix_to_png(conf_mat, class_names, title, figure_file_name=None, type='standard'):
    if type == 'standard':
        plt.rcParams["font.family"] = 'DejaVu Sans'
        figure = plt.figure(figsize=(9, 9))

        plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        # set title
        plt.title(title)

        # render the confusion matrix with percentage and ratio.
        group_counts = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                group_counts.append("{}/{}".format(conf_mat[i, j], conf_mat.sum(axis=1)[i]))
        group_percentages = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        labels = [f"{v1} \n {v2 * 100: .4}%" for v1, v2 in zip(group_counts, group_percentages.flatten())]
        labels = np.asarray(labels).reshape(len(class_names), len(class_names))

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
        mlflow.log_artifact(fig_file_path)
        plt.close(figure)

    elif type == 'mean':

        # Fixme here I pass list of cm (better way)
        plt.rcParams["font.family"] = 'DejaVu Sans'
        figure = plt.figure(figsize=(9, 9))

        # Add values to the plot
        mean_cm_val, std_cm_val = calculate_cm_stats(cm_list=conf_mat, num_classes=len(class_names))

        plt.imshow(mean_cm_val, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        # set title
        plt.title(title)

        labels = [f"{v1 * 100:.4}%  ±{v2 * 100: .4}%" for v1, v2 in zip(mean_cm_val.flatten(), std_cm_val.flatten())]
        labels = np.asarray(labels).reshape(len(class_names), len(class_names))

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
        mlflow.log_artifact(fig_file_path, artifact_path="Validation_cm")
        plt.close(figure)


def fetch_logged_data(run_id):
    client = MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


class SklearnTrainer(object):
    def __init__(self, train_subjects_sessions_dict,
                 test_subjects_sessions_dict,
                 logger=None, cv=None, concatenate_subjects=True):

        self.logger = logger
        self.train_sessions = dict(train_subjects_sessions_dict)
        self.test_sessions = dict(test_subjects_sessions_dict)
        self.concatenate_subjects = concatenate_subjects
        self.cv = cv

    def train(self, model, datamodule, hparams):
        self.model = model
        train_data = datamodule.load_data(self.train_sessions, self.concatenate_subjects)
        classes_names = train_data.className
        # initialize lists for storing results of confusion matrix to calculate mean std
        val_cm_list = []
        val_f1_list = []
        val_recall_list = []


        num_folds = self.cv.n_splits

        if "mlflow" in self.logger:
            log.info("Logging to mlflow...")
            mlflow.set_tracking_uri('file://' + self.logger.mlflow.tracking_uri)

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

                for fold, (train_index, test_index) in enumerate(self.cv.split(train_data, train_data.y)):
                    foldNum = fold + 1

                    log.info("Starting fold {} of {}".format(foldNum, num_folds))

                    X_train, X_test = train_data[train_index], train_data[test_index]
                    y_train, y_test = train_data.y[train_index], train_data.y[test_index]

                    parent_run_id = parent_run.info.run_id
                    mlflow_job_name = f"Fold-{foldNum}"

                    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=mlflow_job_name,
                                          nested=True) as child_run:
                        self.model.fit(X_train, y_train)
                        y_pred = self.model.predict(X_test)


                        f1_score_vali = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred, average="weighted")
                        mlflow.log_metric("vali_f1_score", f1_score_vali)
                        val_f1_list.append(f1_score_vali)

                        recall_score_vali = sklearn.metrics.recall_score(y_true=y_test, y_pred=y_pred, average="weighted")
                        mlflow.log_metric("vali_recall_score", recall_score_vali)
                        val_recall_list.append(recall_score_vali)

                        cm_vali = confusion_matrix(y_test, y_pred)
                        val_cm_list.append(compute_percentages_cm(cm_vali))
                        val_cm_title = f"vali-foldNum-{foldNum}-f1_score-{f1_score_vali}"
                        confusion_matrix_to_png(cm_vali, classes_names, val_cm_title, figure_file_name="vali_cm")



                        # fetch logged data from child run
                        child_run_id = child_run.info.run_id
                        # params, metrics, tags, artifacts = fetch_logged_data(child_run_id)
                        # print("run_id: {}".format(child_run_id))
                        # pprint(params)
                        # pprint(metrics)
                        # pprint(tags)
                        # pprint(artifacts)

                        log.info(f"Train Fold {foldNum} with run-i {child_run_id} is completed!")

                    # log the mean confusion matrix to mlflow parent run
                    mean_f1_score = np.mean(val_f1_list)
                    mlflow.log_metric("vali_mean_f1_score", mean_f1_score)
                    confusion_matrix_to_png(val_cm_list, classes_names, "mean_cm_val", type="mean")
                    # log the mean confusion matrix to mlflow parent run
                    with open("Hparams.json", "w") as f:
                        json.dump(hparams, f)
                    mlflow.log_artifact("Hparams.json", artifact_path="model")
                    # log dataset dict to mlflow parent run

                    for key, value in self.train_sessions.items():
                        self.train_sessions[key] = list(value)

                    with open("train_sessions.json", "w") as f:
                        json.dump(self.train_sessions, f)
                    mlflow.log_artifact("train_sessions.json", artifact_path="model")

                # fetch logged data from parent run
                params, metrics, tags, artifacts = fetch_logged_data(parent_run_id)
                # print("run_id: {}".format(parent_run_id))
                # pprint(params)
                # pprint(metrics)
                # pprint(tags)
                # pprint(artifacts)
                log.info(f"Training completed!")

            return metrics

        else:
            log.warning("Logger not found! Skipping hyperparameter logging...")
            score = self.model.fit(train_data, train_data.y)
            return score

    def test(self, model, datamodule):
        test_data = datamodule.load_data(self.test_sessions, self.concatenate_subjects)

        return model.predict(test_data)
