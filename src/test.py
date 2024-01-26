@utils.task_wrapper
def tune(cfg: DictConfig) -> Tuple[dict, dict]:
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
        pipeline: bbcpy.pipeline.Pipeline = hydra.utils.instantiate(cfg.model)

        log.info("Starting hyperparameter optimization!")

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
            "model": pipeline,
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


        # Return a value that you aim to optimize, e.g., validation loss
        return train_metrics[cfg.optimized_metric]


    log.info("Instantiating callbacks...")
    callbacks = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger = hydra.utils.instantiate(cfg.get("logger"))



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
        hpo_metric_dict = trainer.search_hyperparams(pipeline=pipeline,
                                                     hparams=hparams)
        metric_dict.update(hpo_metric_dict)
        log.info("Hyperparameter search finished!")

    if cfg.get("train"):
        log.info("Starting training!")
        train_metric_dict = trainer.train_baseline(pipeline=pipeline,
                                                   hparams=hparams)
        metric_dict.update(train_metric_dict)
        log.info("Training finished!")

    # clean up
    del datamodule, pipeline, callbacks, logger, trainer

    gc.collect()

    return metric_dict, object_dict