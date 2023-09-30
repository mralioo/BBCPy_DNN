from typing import Optional, Tuple, Any

import hydra
import numpy as np
import pyrootutils
from lightning import Callback
from omegaconf import DictConfig
import gc

import bbcpy

from src.data.smr_datamodule import SMR_Data
from trainer.baseline_trainer import SklearnTrainer

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

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> tuple[
    dict[Any, Any] | Any, dict[str, SMR_Data  | list[Callback] | DictConfig | SklearnTrainer | Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in numpy and python.random
    if cfg.get("seed"):
        np.random.seed(cfg.seed)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: SMR_Data = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    pipeline: bbcpy.pipeline.Pipeline = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger = hydra.utils.instantiate(cfg.get("logger"))

    log.info("Instantiating hyperparameter search...")
    hyperparameter_search = hydra.utils.instantiate(cfg.get("hparams_search"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: SklearnTrainer = hydra.utils.instantiate(cfg.trainer,
                                                      datamodule=datamodule,
                                                      logger=logger,
                                                      hyperparameter_search=hyperparameter_search)

    metric_dict = {}

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "pipeline": pipeline,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    hparams = utils.log_sklearn_hyperparameters(object_dict)

    if cfg.get("tune"):
        log.info("Starting hyperparameter search!")
        metric_dict = trainer.search_hyperparams(pipeline=pipeline,
                                                 hparams=hparams)
        log.info("Hyperparameter search finished!")

    if cfg.get("train"):
        log.info("Starting training!")
        metric_dict = trainer.train(pipeline=pipeline,
                                    hparams=hparams)
        log.info("Training finished!")

    if cfg.get("test"):
        log.info("Starting testing!")
        metric_dict = trainer.test(pipeline=pipeline,
                                   datamodule=datamodule)
        log.info("Training finished!")

    # clean up
    del datamodule, pipeline, callbacks, logger, trainer

    gc.collect()

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="baseline_train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entrypoint of the project."""

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # TODO: save metrics fro later
    # subject_name =list(cfg.data.subject_sessions_dict.keys())[0]
    #

    # # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
