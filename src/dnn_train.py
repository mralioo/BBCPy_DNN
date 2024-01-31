import os
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import lightning as L
import pandas as pd
import pyrootutils
import torch
from hydra.core.hydra_config import HydraConfig
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

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    # load data
    log.info("Loading data...")
    datamodule.load_raw_data()

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model,
                    datamodule=datamodule,
                    ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        log.info(f"Best ckpt path: {ckpt_path}")
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")

        trainer.test(model=model,
                     dataloaders=datamodule,
                     ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict

@utils.task_wrapper
def train_cross_validation(cfg: DictConfig) -> Tuple[dict, dict]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # load data
    log.info("Loading data...")
    datamodule.load_raw_data()

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    object_dict = {}

    run_name = cfg.get("logger").mlflow.run_name

    if cfg.get("train"):
        log.info("Starting training!, Cross Validation")

        # Initialize a dictionary to store the metrics
        cv_metric_dict = {"train": [], "val": [], "test": []}
        log.info("Cross validation strategy; k-fold runs 1,2,3,4,5 for train/val and run 6 for test")
        # FIXME : change the number of folds

        nums_folds = 5

        for k in range(nums_folds):

            log.info(f"Fold {k}...")

            # print memory usage
            print_memory_usage()
            # print cpu cores
            print_cpu_cores()
            # print gpu info
            print_gpu_memory()

            log.info("Instantiating loggers...")
            log.info(f"Kfold: {k}")

            OmegaConf.update(cfg.get("logger").mlflow, "run_name", f"{run_name}_fold_{k + 1}", merge=True)
            logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

            log.info("Instantiating callbacks...")
            callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

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

            # Update the indices for the dataloaders
            datamodule.update_kfold_index(k)

            # Train the model
            trainer.fit(model, datamodule)

            # sort the metrics train and val
            train_metrics = {}
            val_metrics = {}

            for key, value in trainer.callback_metrics.items():
                if key.startswith('train'):
                    train_metrics[key] = value
                elif key.startswith('val'):
                    val_metrics[key] = value

            cv_metric_dict["train"].append(train_metrics)
            cv_metric_dict["val"].append(val_metrics)

            if cfg.get("test"):
                log.info("Starting testing!")
                ckpt_path = trainer.checkpoint_callback.best_model_path
                log.info(f"Best ckpt path: {ckpt_path}")
                if ckpt_path == "":
                    log.warning("Best ckpt not found! Using current weights for testing...")

                trainer.test(model=model,
                             dataloaders=datamodule,
                             ckpt_path=ckpt_path)

                test_metrics = trainer.callback_metrics
                cv_metric_dict["test"].append(test_metrics)

        # Aggregate metrics from all folds
        final_metrics = {
            "mean_train": utils.utils.aggregate_metrics(cv_metric_dict["train"]),
            "mean_val": utils.utils.aggregate_metrics(cv_metric_dict["val"]),
            "mean_test": utils.utils.aggregate_metrics(cv_metric_dict["test"]),
        }

    # merge train and test metrics
    metric_dict = {**final_metrics, **cv_metric_dict}

    return metric_dict, object_dict

@utils.task_wrapper
def tune(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    trial_id = HydraConfig.get().job.id
    run_name = cfg.get("logger").mlflow.run_name
    log.info(f"Hydra Job ID: {trial_id}")
    OmegaConf.update(cfg.get("logger").mlflow, "run_name", f"T{trial_id}_{run_name}", merge=True)
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    # load data
    log.info("Loading data...")
    datamodule.load_raw_data()

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("tune"):
        log.info("Starting training!")
        trainer.fit(model,
                    datamodule=datamodule,
                    ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        log.info(f"Best ckpt path: {ckpt_path}")
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")

        trainer.test(model=model,
                     dataloaders=datamodule,
                     ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="dnn_train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)

    print_gpu_info()

    utils.extras(cfg)

    # save metrics to csv file for later analysis
    model_name = cfg.model.net._target_.split(".")[-1]
    subject_name = list(cfg.data.subject_sessions_dict.keys())[0]
    task_name = cfg.data.task_name
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    csv_file_path = Path(f"{cfg.paths.results_dir}/{task_name}_{model_name}_{subject_name}.csv")

    metric_dict = {}

    # train the model; there is 2 options : train/val split or cross_validation
    if cfg.data.train_val_split:
        if cfg.get("tune"):
            metric_dict, _ = tune(cfg)
        elif cfg.get("train"):
            metric_dict, _ = train(cfg)
        convert_tensor_to_float(metric_dict)

    if cfg.data.cross_validation:

        metric_dict, _ = tune(cfg)
        # Convert tensors to float values
        convert_tensor_to_float(metric_dict)

    # Open the CSV file in write mode
    flat_dict = {}
    for key, value in metric_dict.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat_key = f"{key}@{inner_key}"
                flat_dict[flat_key] = inner_value
        else:
            flat_dict[key] = value
    # Convert to pandas DataFrame
    df = pd.DataFrame([flat_dict])
    # Open the CSV file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        df.to_csv(file, index=False)

    log.info(f"Metrics saved to {csv_file_path}")

    log.info("Done!")

    return metric_dict[cfg.optimized_metric]


def convert_tensor_to_float(d):
    for key, value in d.items():
        if isinstance(value, dict):
            convert_tensor_to_float(value)
        elif isinstance(value, list):
            for item in value:
                convert_tensor_to_float(item)
        elif isinstance(value, torch.Tensor):
            d[key] = float(value)


if __name__ == "__main__":
    main()
