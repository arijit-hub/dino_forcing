"""Implementing the lightning module for training the model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L


#############################################################
#                   LR scheduler for jit                #
#############################################################
def jit_lr_lambda(warmup_steps):
    """Builds the lr warmup used by JiT.

    Shamelessly adapted from:
        https://github.com/LTH14/JiT/blob/main/main_jit.py
    """
    warmup_steps = max(int(warmup_steps), 0)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        else:
            return 1.0

    return lr_lambda


#############################################################
#                       Loss Function                       #
#############################################################
class SiTLoss(nn.Module):
    """Implements the loss function for the flow matching model.
    Shamelessly adapted from:
        https://github.com/sihyun-yu/REPA/blob/main/loss.py
    """

    def __init__(
        self,
        self_cond_loss=False,
        feature_loss_weight=1.0,
        **kwargs,
    ):
        """Constructor."""
        super().__init__()
        self.self_cond_loss = self_cond_loss
        self.feature_loss_weight = feature_loss_weight

    def forward(self, batch):
        """Calculates the loss."""

        ## First calculate the v-loss and self_cond loss ##
        v_loss = (batch["out"] - batch["true_velocity"]) ** 2
        v_loss = v_loss.mean(dim=list(range(1, batch["out"].ndim)))
        v_loss = v_loss.mean()

        self_cond_loss = torch.tensor(0.0, device=v_loss.device)
        if self.self_cond_loss and "self_condition" in batch:
            alignment_features = batch.get("alignment_features", None)
            alignment_features = F.normalize(alignment_features, dim=-1)
            self_condition = F.normalize(batch["self_condition"], dim=-1)
            self_cond_loss = -(self_condition * alignment_features.detach()).sum(dim=-1)
            self_cond_loss = self_cond_loss.mean(
                dim=list(range(1, self_cond_loss.ndim))
            )
            self_cond_loss = self_cond_loss.mean()

        total_loss = v_loss + self.feature_loss_weight * self_cond_loss

        batch["loss"] = total_loss
        batch["v_loss"] = v_loss.detach()
        batch["self_cond_loss"] = self_cond_loss.detach()
        return batch


############################################################
#               JiT Loss Function                          #
############################################################
class JiTLoss(nn.Module):
    """Implements the loss function for the flow matching model."""

    def __init__(
        self,
        self_cond_loss=False,
        feature_loss_weight=1.0,
        t_eps: float = 1e-5,
    ):
        """Constructor."""
        super().__init__()
        self.self_cond_loss = self_cond_loss
        self.feature_loss_weight = feature_loss_weight
        self.t_eps = t_eps

    def forward(self, batch):
        """Calculates the loss."""
        ## First calculate the main loss ##
        timesteps = batch["timesteps"].view(-1, 1, 1, 1)
        v_pred = (batch["out"] - batch["noisy_imgs"]) / (1 - timesteps).clamp(
            min=self.t_eps
        )
        true_v = (batch["clean_imgs"] - batch["noisy_imgs"]) / (1 - timesteps).clamp(
            min=self.t_eps
        )
        v_loss = (v_pred - true_v) ** 2
        v_loss = v_loss.mean(dim=list(range(1, batch["out"].ndim)))
        v_loss = v_loss.mean()

        self_cond_loss = torch.tensor(0.0, device=v_loss.device)
        if self.self_cond_loss and "self_condition" in batch:
            alignment_features = batch.get("alignment_features", None)
            alignment_features = F.normalize(alignment_features, dim=-1)
            self_condition = F.normalize(batch["self_condition"], dim=-1)
            self_cond_loss = -(self_condition * alignment_features.detach()).sum(dim=-1)
            self_cond_loss = self_cond_loss.mean(
                dim=list(range(1, self_cond_loss.ndim))
            )
            self_cond_loss = self_cond_loss.mean()

        total_loss = v_loss + self.feature_loss_weight * self_cond_loss

        batch["loss"] = total_loss
        batch["v_loss"] = v_loss.detach()
        batch["self_cond_loss"] = self_cond_loss.detach()
        return batch


#############################################################
#               SiT Training module                         #
#############################################################
class SiTTrainingModule(L.LightningModule):
    def __init__(
        self,
        network,
        ema_network,
        loss,
        sampler,
        train_preprocessing,
        val_preprocessing,
        img_postprocessing,
        use_compile,
        # jit_optim <- whether to use jit optimizer
        # true_dino_features_prob <- probability of using true dino features for self conditioning
        **kwargs,
    ):
        super().__init__()

        ## Saving the hyperparameters for logging ##
        self.save_hyperparameters()
        ## Setting up the network and ema network ##
        if use_compile:
            self.network = torch.compile(network)
        else:
            self.network = network
        self.ema_network = ema_network(model=self.network)
        self.ema_network.requires_grad_(False)
        self.ema_network.eval()

        ## Setting up the loss ##
        self.loss = loss

        ## Setting up the sampler ##
        self.sampler = sampler

        ## Setting up the preprocessing and postprocessing ##
        self.train_preprocessing = train_preprocessing
        self.val_preprocessing = val_preprocessing
        self.img_postprocessing = img_postprocessing

        ## Setting strict loading to False ##
        self.strict_loading = False

    def state_dict(self):
        """Returns the state dictionary of the model. We remove the vae and dino encoder so that
        it doesnt gets saved in the checkpoint, saving space."""
        return {
            k: v
            for k, v in super().state_dict().items()
            if (("vae" not in k) and ("dino_encoder" not in k))
        }

    def forward(self, batch, use_ema=False, **kwargs):
        return (
            self.ema_network(batch, **kwargs)
            if use_ema
            else self.network(batch, **kwargs)
        )

    def forward_process(self, batch):
        """Forward ODE process."""

        ## Getting the noise and timesteps ##
        noise = torch.randn(
            *batch["clean_imgs"].shape,
            generator=batch.get("generator", None),
            device=batch["clean_imgs"].device,
            dtype=batch["clean_imgs"].dtype,
        )

        timesteps = batch["timesteps"].view(-1, 1, 1, 1)

        ## Adding noise to the clean images ##
        batch["noisy_imgs"] = timesteps * batch["clean_imgs"] + (1 - timesteps) * noise

        ## Getting the velocity ##
        batch["true_velocity"] = batch["clean_imgs"] - noise  # for jit this is not used

        return batch

    @torch.no_grad()
    def reverse_process(
        self,
        conditions=None,
        num_sampling_steps=50,
        use_ema=True,
        generator=None,
        cfg=7.5,
        return_self_condition=False,
        **kwargs,
    ):
        """Reverse ODE process."""
        assert conditions is not None, "Conditions must be provided for sampling."

        ## Convert conditions to tensor if they are not already ##
        if isinstance(conditions, list):
            conditions = torch.tensor(conditions, device=self.device, dtype=torch.int)

        null_conditions = torch.full_like(
            conditions, fill_value=self.network._get_null_condition()
        )
        condition_masks = None
        null_condition_masks = None

        samples, self_condition = self.sampler.sample(
            model=self.ema_network if use_ema else self.network,
            null_conditions=null_conditions,
            conditions=conditions,
            null_condition_masks=null_condition_masks,
            condition_masks=condition_masks,
            num_sampling_steps=num_sampling_steps,
            cfg=cfg,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
            return_self_condition=return_self_condition,
            **kwargs,
        )

        if return_self_condition:
            samples = [
                self.img_postprocessing(sample.to(self.device))
                .detach()
                .cpu()
                .squeeze(0)
                for sample in samples
            ]
            return samples, self_condition

        ## Postprocessing the images ##
        samples = self.img_postprocessing(samples)

        return samples

    def training_step(self, batch, batch_idx):
        """Implements the training step."""

        ## Doing the training preprocessing ##
        batch = self.train_preprocessing(
            batch, null_condition=self.network._get_null_condition()
        )

        ## Doing the forward process to get the noisy images and the velocity ##
        batch = self.forward_process(batch)

        ## We check if it is self_cond training or not ##
        if self.network.use_self_cond:
            self.eval()  # Set the model to eval mode for self-conditioning
            ## Get the self conditioning features ##
            with torch.no_grad():
                self_cond_batch = batch.copy()
                self_cond_batch = self(self_cond_batch)
                sc = self_cond_batch["self_condition"].detach()
                true_dino_features_prob = self.hparams.get(
                    "true_dino_features_prob", 0.0
                )
                if true_dino_features_prob > 0.0:
                    drop_ids = (
                        torch.rand(len(sc), device=sc.device) < true_dino_features_prob
                    )
                    batch["self_condition"] = torch.where(
                        drop_ids.view(-1, 1, 1),
                        batch["alignment_features"].detach(),
                        sc,
                    )
                else:
                    batch["self_condition"] = sc

                del self_cond_batch

            self.train()  # Set the model back to train mode

        ## Getting the model prediction ##
        batch = self(batch)

        batch["global_step"] = self.global_step

        ## Calculating the loss ##
        batch = self.loss(batch)

        ## Logging the losses ##
        self.log(
            "loss/train",
            batch["loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "loss/train_v_loss",
            batch["v_loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )

        if "self_cond_loss" in batch:
            self.log(
                "loss/train_self_cond_loss",
                batch["self_cond_loss"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )

        return batch["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Updating the ema network after every training batch."""
        self.ema_network.update()

    def validation_step(self, batch, batch_idx):
        """Implements the validation step."""

        ## Doing the validation preprocessing ##
        batch = self.val_preprocessing(
            batch, null_condition=self.network._get_null_condition()
        )

        ## Setting the generator ##
        batch["generator"] = torch.Generator(device=self.device).manual_seed(3407)

        ## Doing the forward process to get the noisy images and the velocity ##
        batch = self.forward_process(batch)

        ## We check if it is self_cond training or not ##
        if self.network.use_self_cond:
            ## Get the self conditioning features ##
            with torch.no_grad():
                self_cond_batch = batch.copy()
                self_cond_batch = self(self_cond_batch)
                batch["self_condition"] = self_cond_batch["self_condition"].detach()
                del self_cond_batch

        ## Getting the model prediction ##
        batch = self(batch)

        ## Calculating the loss ##
        batch = self.loss(batch)

        ## Logging the losses ##
        self.log(
            "loss/val",
            batch["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log(
            "loss/val_v_loss",
            batch["v_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        if "self_cond_loss" in batch:
            self.log(
                "loss/val_self_cond_loss",
                batch["self_cond_loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def _jit_steps_per_epoch(self):
        """Computes the number of optimizer steps per epoch.

        Needed to convert the epoch-based warmup/decay schedule used by
        https://github.com/bytetriper/RAE (warmup_epochs/decay_end_epoch) into
        steps, since this codebase trains against `max_steps` on a streaming
        WebDataset rather than a fixed number of epochs.
        """
        datamodule = self.trainer.datamodule
        global_batch_size = (
            datamodule.batch_size
            * self.trainer.world_size
            * self.trainer.accumulate_grad_batches
        )
        return max(datamodule.train_dataset_size // global_batch_size, 1)

    def configure_optimizers(self):
        """Configures the optimizers. We keep it fixed to compare properly.
        The optimizer itself is shamelessly copied from:
            https://github.com/CompVis/tread/blob/master/configs/config.yaml
            https://github.com/LTH14/JiT/blob/main/main_jit.py
        """

        jit_optim = self.hparams.get("jit_optim", False)
        if jit_optim:
            base_lr = 2e-4
            optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=base_lr,
                betas=(0.9, 0.95),
                weight_decay=0.0,
                eps=1e-08,
            )

            ## Building the warmup ##
            ## https://github.com/LTH14/JiT/blob/main/main_jit.py ##
            warmup_steps = self.hparams.get("jit_warmup_steps", None)
            if warmup_steps is None:
                warmup_epochs = self.hparams.get("jit_warmup_epochs", 5)
                warmup_steps = int(warmup_epochs * self._jit_steps_per_epoch())

            lr_lambda = jit_lr_lambda(warmup_steps=warmup_steps)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        else:
            optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=1e-4,
                betas=(0.9, 0.999),
                weight_decay=0.0,
                eps=1e-08,
            )

            return optimizer
