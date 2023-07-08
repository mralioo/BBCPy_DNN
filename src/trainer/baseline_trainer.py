import json
import os.path
from pprint import pprint
from src.utils.file_mgmt import pjoin
import datetime
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from src import utils

log = utils.get_pylogger(__name__)


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
        train_data = datamodule.prepare_data(self.train_sessions, self.concatenate_subjects)
        if "mlflow" in self.logger:
            log.info("Logging to mlflow...")
            mlflow.set_tracking_uri('file://' + self.logger.mlflow.tracking_uri)

            experiment_name = self.logger.mlflow.experiment_name
            run_name = "{}_{}".format(self.logger.mlflow.run_name,
                                      datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


            log.info("experiment_name: {}, run_name: {}".format(experiment_name, run_name))
            try:
                mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
                experiment = mlflow.get_experiment_by_name(experiment_name)
            except:
                experiment = mlflow.get_experiment_by_name(experiment_name)

            mlflow.sklearn.autolog()
            with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as run:
                mlflow.set_tag("Run-Id", run.info.run_uuid)
                scores = cross_val_score(self.model, train_data, train_data.y, cv=self.cv, error_score='raise')
                print(f"mean score: {scores.mean():.3f}")
                print(f"score std: {scores.std():.3f}")

                # fetch logged data
                params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)

                pprint(params)
                pprint(metrics)
                pprint(tags)
                pprint(artifacts)

                self.active_run = mlflow.active_run()

                # Get the artifact URI
                artifact_uri = self.active_run.info.artifact_uri
                print("Artifact URI:", artifact_uri)

                with open("Hparams.json", "w") as f:
                    json.dump(hparams, f)
                mlflow.log_artifact("Hparams.json", "Hparams")
                log.info(
                    f"Training completed!, close mlflow run: {run.info.run_uuid} of the experiment:{experiment_name}")

            return metrics

        else:
            log.warning("Logger not found! Skipping hyperparameter logging...")
            self.model.fit(train_data, train_data.y)

    def test(self, model, datamodule):
        test_data = datamodule.prepare_data(self.test_sessions, self.concatenate_subjects)

        return model.predict(test_data)
