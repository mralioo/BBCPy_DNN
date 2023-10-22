import os
from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from omegaconf import OmegaConf

from hydra.core.hydra_config import HydraConfig

import src.utils.srm_utils

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

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    avg_metrics = {}
    object_dict = {}

    run_name = cfg.get("logger").mlflow.run_name

    if cfg.get("tune"):
        log.info("Starting hyperparameter optimization!")

        trial_id = HydraConfig.get().job.id
        log.info(f"Hydra Job ID: {trial_id}")

        # load data
        datamodule.load_raw_data()

        cv_score = []
        nums_folds = src.utils.srm_utils.cross_validation["num_splits"]
        for k in range(nums_folds):

            log.info(f"Fold {k}...")

            # print memory usage
            print_memory_usage()
            # print cpu cores
            print_cpu_cores()
            # print gpu info
            print_gpu_memory()

            datamodule.update_kfold_index(k)
            # here we train the model on given split...
            # inti trainer again
            log.info("Instantiating loggers...")
            OmegaConf.update(cfg.get("logger").mlflow, "run_name", f"T{trial_id}_CV_{k}_{run_name}", merge=True)
            logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

            log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
            trainer: Trainer = hydra.utils.instantiate(cfg.trainer,
                                                       callbacks=callbacks,
                                                       logger=logger)

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

            cv_score.append(train_metrics)

        for key in cv_score[0].keys():
            avg_metrics[f"avg_{key}"] = sum([fold[key].item() for fold in cv_score]) / len(cv_score)

        for key, value in avg_metrics.items():
            trainer.logger.log_metrics({f"{key}": value})

    return avg_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="dnn_train_hpo.yaml")
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
