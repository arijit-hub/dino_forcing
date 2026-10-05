"""Implements the image logging callback"""

import torch
from lightning.pytorch.callbacks import Callback
import os
from glob import glob
import torch.distributed
import wandb
import numpy as np
import textwrap
from einops import rearrange
from src.utils.imagenet_classes import imagenet_idx_to_text


class LogImageCallback(Callback):
    """Implements the image logging callback."""

    def __init__(
        self,
        num_resample=4,
        num_sampling_steps=250,
        log_after_n_steps=10000,
        num_total_images=80,
    ):
        """Constructor."""
        super().__init__()

        ## Saving attributes ##
        self.ready = True
        self.num_resample = num_resample
        self.num_sampling_steps = num_sampling_steps
        self.log_after_n_steps = log_after_n_steps
        self.num_total_images = num_total_images

    def on_sanity_check_start(self, trainer, pl_module):
        """Called when the sanity check starts."""

        ## Setting the ready flag to false ##
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Called when the sanity check ends."""

        ## Setting the ready flag to true ##
        self.ready = True

    def _get_conditions_for_model(self, pl_module):
        """Get condition and names."""

        class_conditions = np.linspace(
            0, 1000, self.num_total_images, endpoint=False, dtype=int
        ).tolist()
        class_names = [f"{imagenet_idx_to_text[i]}" for i in class_conditions]
        return class_conditions, class_names
    
    def _logging_images(self, trainer, pl_module):
        """Implements the logging to wandb."""

        ## Set the pl_module to evaluation mode ##
        pl_module.eval()

        ## Get conditions based on model type ##
        conditions, condition_names = self._get_conditions_for_model(pl_module)

        ## Calculating number of conditions per gpu ##
        ## We might miss the last few conditions if the number is not divisible by the number of GPUs. ##
        num_conditions_per_gpu = max(1, int(len(conditions) // trainer.world_size))

        start_idx = trainer.global_rank * num_conditions_per_gpu
        end_idx = (trainer.global_rank + 1) * num_conditions_per_gpu

        gpu_conditions = conditions[start_idx:end_idx]
        gpu_condition_names = condition_names[start_idx:end_idx]

        ## Looping through the conditions of the current GPU ##
        for idx, (name, condition) in enumerate(
            zip(gpu_condition_names, gpu_conditions)
        ):
            ## Making the name a list of strings for later all_gather operation ##
            names = [name]

            ## Generating the images ##
            # Prepare condition for generation based on model type
            ## Setting the generator ##
            generator = torch.Generator(device=pl_module.device).manual_seed(3407)
            gen_condition = (
                [condition] * self.num_resample if condition is not None else None
            )

            images = pl_module.reverse_process(
                conditions=gen_condition,
                num_sampling_steps=self.num_sampling_steps,
                use_ema=True,  # Use EMA model for sampling
                generator=generator,
                cfg=3.5,
            )  # we do sampling with the default 3.5 cfg for logging!

            ## Making a 2D grid of images ##
            images = rearrange(
                images,
                "(b1 b2) c h w -> c (b1 h) (b2 w)",
                b1=int(self.num_resample**0.5),
                b2=int(self.num_resample**0.5),
            )

            ## All gathering the images and the names for logging ##
            if trainer.world_size > 1:

                ## Gathering the images across all the gpus ##
                images = pl_module.all_gather(images)

                ## We do the same for names. However given its a list we need to set a placeholder ##
                all_names = [None for _ in range(trainer.world_size)]
                torch.distributed.all_gather_object(
                    all_names,  # placeholder for all names
                    names,
                )
                names = [
                    textwrap.shorten(name_str, width=50, placeholder=f"...")
                    for each_gpu_names in all_names
                    for name_str in each_gpu_names
                ]

            if trainer.world_size == 1:
                ## If we are on a single GPU, we can just use the names as is ##
                names = [
                    textwrap.shorten(name_str, width=50, placeholder=f"...")
                    for name_str in names
                ]
                images = [images]

            ## Logging images in case of global rank 0 ##
            if trainer.global_rank == 0:
                for img, name_str in zip(images, names):
                    img = img * 255.0
                    img = torch.clamp(img, 0, 255).to(torch.uint8)
                    trainer.logger.experiment.log(
                        {
                            f"{name_str}": [
                                wandb.Image(img, normalize=False, file_type="jpg")
                            ] # jpg saves space
                        },
                        step=trainer.global_step + 1,
                    )

        ## Setting the pl_module back to training mode ##
        pl_module.train()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Called before the train batch starts."""

        if self.ready and trainer.global_step == 0:
            ## Logging the images ##
            self._logging_images(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called just after the train batch ends."""

        ## Logging the images ##
        if self.ready and ((trainer.global_step + 1) % self.log_after_n_steps == 0):
            self._logging_images(trainer, pl_module)