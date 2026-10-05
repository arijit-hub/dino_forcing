"""Facilitates the training of models."""

## Importing the fan favourite (not mine) ##
import hydra
from hydra.utils import instantiate
import torch
from src.utils.log_utils import log_config
import lightning.pytorch as L


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def train(cfg):
    """Main function whch orchestrates everything."""

    ## Instantiating the data object ##
    data = instantiate(cfg.data)

    ## Instantiating the model object ##
    model = instantiate(cfg.model)

    ## Instantiating the trainer object ##
    trainer = instantiate(cfg.trainer)

    ## Logging the config ##
    log_config(
        config=cfg,
        logger=trainer.logger,
    )

    ## Training the model ##
    trainer.fit(
        model,
        datamodule=data,
        ckpt_path=cfg.ckpt_path,
        weights_only=False,
    )


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    train()