from typing import List, Optional, Tuple

import hydra
import lightning as L
import optuna.trial
import pyrootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from omegaconf import OmegaConf

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils
from utils.device import print_gpu_info, print_memory_usage, print_cpu_cores, print_gpu_memory

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    avg_metrics = {}
    object_dict = {}

    run_name = cfg.get("logger").mlflow.run_name

    # load data
    datamodule.load_raw_data()

    log.info("Instantiating Optuna...")
    optuna_hpo = cfg.get("hparams_search")
    sampler = hydra.utils.instantiate(optuna_hpo.sampler)
    pruner = hydra.utils.instantiate(optuna_hpo.pruner)

    study = optuna.create_study(direction=optuna_hpo.direction,
                                sampler=sampler,
                                pruner=pruner,
                                study_name=optuna_hpo.study_name,
                                storage="sqlite:///./optuna.db",
                                load_if_exists=False)

    def objective(trial):
        # Suggest values for hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        F1 = trial.suggest_int('F1', 8, 12)
        F2 = trial.suggest_int('F2', 8, 24)

        # Override the configuration with Optuna's suggestions
        cfg.optimizer.lr = lr
        cfg.model.net.F1 = F1
        cfg.model.net.F2 = F2

        # Instantiate the model
        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info("Starting hyperparameter optimization!")

        cv_score = []
        nums_folds = cfg.data.cross_validation["num_splits"]
        for k in range(nums_folds):
            print(f"Fold {k}...")

            # print memory usage
            print_memory_usage()
            # print cpu cores
            print_cpu_cores()
            # print gpu info
            print_gpu_memory()

            datamodule.update_kfold_index(k)

            trial_id = trial.number
            # here we train the model on given split...
            # inti trainer again
            log.info("Instantiating loggers...")
            OmegaConf.update(cfg.get("logger").mlflow, "run_name", f"T{trial_id}_CV_{k}_{run_name}", merge=True)
            logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

            log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
            trainer: Trainer = hydra.utils.instantiate(cfg.trainer,
                                                       callbacks=callbacks,
                                                       logger=logger,
                                                       num_sanity_val_steps=0)

            object_dict = {
                "cfg": cfg,
                "model": model,
                "datamodule": datamodule,
                "callbacks": callbacks,
                "logger": logger,
                "trainer": trainer,
            }

            if logger:
                log.info("Logging hyperparameters!")
                utils.log_hyperparameters(object_dict)

            trainer.fit(model,
                        datamodule=datamodule,
                        ckpt_path=cfg.get("ckpt_path"))

            train_metrics = trainer.callback_metrics

            cv_score.append(train_metrics[cfg.optimized_metric])

        # Return a value that you aim to optimize, e.g., validation loss
        return sum(cv_score) / nums_folds

    study.optimize(objective,
                   n_trials=optuna_hpo.n_trials,
                   timeout=optuna_hpo.timeout)

    print("Best hyperparameters:", study.best_params)

    return avg_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="dnn_optuna_hpo.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)

    print_gpu_info()

    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    return metric_dict[cfg.optimized_metric]


if __name__ == "__main__":
    main()
