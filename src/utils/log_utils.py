"""Utilities for logging."""

from omegaconf import OmegaConf
from lightning.pytorch.utilities import rank_zero_only

@rank_zero_only
def log_config(config, logger):
    """Logs the config dict to the logger."""

    config = OmegaConf.to_container(config, resolve=True)

    hparams = {
        "data": config["data"],
        "model": config["model"],
        "trainer": config["trainer"],
    }

    logger._wandb_init.update({"config": hparams})