import torch
import torch.nn as nn
import numpy as np
from diffusers import AutoencoderKL
from torchvision.transforms import transforms
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

############################################################
#                       LATENT SPACE                       #
############################################################


############################################################
#                    SD1.5 Preprocessing                   #
############################################################
class SD1_5_LatentPreprocessing(nn.Module):
    """Implements the latent preprocessing for sd 1.5."""

    def __init__(self, channel_normalization: bool = False):
        """Constructor.

        :param channel_normalization: bool: Whether to normalize the channels or not.
        """
        super().__init__()

        if channel_normalization:
            scale = 0.5 / torch.tensor([4.8015, 5.3559, 3.9480, 3.9715]).unsqueeze(
                0
            ).unsqueeze(-1).unsqueeze(-1)
            bias = (
                -torch.tensor([0.9844, -0.4782, 0.3184, 0.5426])
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                * scale
            )
        else:
            scale = torch.tensor([0.18215])
            bias = torch.tensor([0.0])

        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, x):
        """Preprocesses the latents."""
        # with torch.no_grad():
        x = x * self.scale + self.bias
        return x


############################################################
#           Combined preprocess with timesteps             #
############################################################
class NormalPreprocessing(nn.Module):
    """Implements the normal training preprocessing of images and texts."""

    def __init__(
        self,
        image_preprocessing=SD1_5_LatentPreprocessing,
        train_timestep_sampling: str = "uniform",
        clip_max=1 - 1e-6,
        unconditional_cond_drop_prob: float = 0.1,
        null_condition=1000,
    ):
        """Constructor.

        :param image_preprocessing: Image preprocessing module.
        :param train_timestep_sampling: str: The type of sampling for timesteps during training.
            Options are 'uniform' or 'log_normal'.
        """
        super().__init__()

        self.image_preprocessing = (
            nn.Identity() if image_preprocessing == None else image_preprocessing
        )

        self.train_timestep_sampling = train_timestep_sampling
        self.clip_max = clip_max
        self.unconditional_cond_drop_prob = unconditional_cond_drop_prob
        assert (
            self.unconditional_cond_drop_prob > 0 and null_condition
        ), "If unconditional_cond_drop_prob is greater than 0, null_condition must be provided."
        self.null_condition = null_condition

    def forward(self, batch, *args, **kwargs):
        """Does the preprocessing of images and latents.

        :param batch: dict: The batch of data. It must consist of the following keys:
            - clean_imgs: The images or the latents.
        """

        with torch.no_grad():
            ## Setting the timesteps for training or validation ##
            if self.training:
                if self.train_timestep_sampling == "uniform":
                    batch["timesteps"] = torch.rand(
                        len(batch["clean_imgs"]),
                        device=batch["clean_imgs"].device,
                        dtype=batch["clean_imgs"].dtype,
                    )
                elif self.train_timestep_sampling == "log_normal":
                    batch["timesteps"] = torch.sigmoid(
                        torch.normal(
                            mean=0.0,
                            std=1.0,
                            size=(len(batch["clean_imgs"]),),
                            device=batch["clean_imgs"].device,
                            dtype=batch["clean_imgs"].dtype,
                        )
                    )

                else:
                    raise NotImplementedError(
                        f"Unknown timestep sampling method: {self.train_timestep_sampling}"
                    )

                conditions = batch["conditions"]
                drop_ids = (
                    torch.rand(
                        len(conditions),
                        device=conditions.device,
                    )
                    < self.unconditional_cond_drop_prob
                )

                batch["conditions"] = torch.where(
                    drop_ids, self.null_condition, conditions
                )

            else:
                batch["timesteps"] = torch.as_tensor(
                    np.linspace(
                        start=0.0,
                        stop=1.0,
                        num=len(batch["clean_imgs"]),
                        endpoint=False,
                    ),
                    dtype=batch["clean_imgs"].dtype,
                    device=batch["clean_imgs"].device,
                )

            ## Clamping the timesteps ##
            batch["timesteps"] = torch.clamp(batch["timesteps"], max=self.clip_max)
            ## Preprocessing the images ##
            batch["clean_imgs"] = self.image_preprocessing(batch["clean_imgs"])

        return batch


############################################################
#                    SD1.5 Postprocessing                   #
############################################################
## Implementing the latent postprocessing for sd 1.5##
class SD1_5_LatentPostprocessing(nn.Module):
    """Implements the latent postprocessing for sd 1.5."""

    def __init__(self, channel_normalization: bool = False):
        """Constructor.

        :param channel_normalization: bool: Whether to normalize the channels or not.
        """
        super().__init__()

        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-ema", use_safetensors=True
        )

        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        if channel_normalization:
            scale = 0.5 / torch.tensor([4.8015, 5.3559, 3.9480, 3.9715]).unsqueeze(
                0
            ).unsqueeze(-1).unsqueeze(-1)
            bias = (
                -torch.tensor([0.9844, -0.4782, 0.3184, 0.5426])
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                * scale
            )
        else:
            scale = torch.tensor([0.18215])
            bias = torch.tensor([0.0])

        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, x):
        """Postprocesses the latents."""
        # with torch.no_grad():
        x = (x - self.bias) / self.scale
        x = self.vae.decode(x).sample
        x = x.clamp(-1, 1)
        x = x * 0.5 + 0.5
        return x

############################################################
#                       PIXEL SPACE                        #
############################################################
class PixelPreprocessing(nn.Module):
    """Implements pixel space preprocessing for images."""

    def __init__(
        self,
        resolution=256,
        P_mean=0.0,
        P_std=1.0,
        unconditional_cond_drop_prob: float = 0.1,
        null_condition=1000,
        load_dino=False,
        dino_config="b",
    ):
        """Constructor.

        :param resolution: int: The resolution of the images.
        """
        super().__init__()

        self.resolution = resolution
        self.P_mean = P_mean
        self.P_std = P_std
        self.unconditional_cond_drop_prob = unconditional_cond_drop_prob
        assert (
            self.unconditional_cond_drop_prob > 0 and null_condition
        ), "If unconditional_cond_drop_prob is greater than 0, null_condition must be provided."
        self.null_condition = null_condition
        self.load_dino = load_dino
        self.dino_config = dino_config

        if self.load_dino:
            self._build_dino_encoder()

    def _build_dino_encoder(self):
        """Builds (or rebuilds) the frozen DINOv2 encoder."""
        self.dino_encoder = torch.hub.load(
            "facebookresearch/dinov2",
            f"dinov2_vit{self.dino_config}14",
        )
        del self.dino_encoder.head
        patch_resolution = 16 * (self.resolution // 256)
        self.dino_encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.dino_encoder.pos_embed.data,
            [patch_resolution, patch_resolution],
        )
        self.dino_encoder.head = torch.nn.Identity()
        self.dino_encoder.eval()
        for param in self.dino_encoder.parameters():
            param.requires_grad = False

    def __getstate__(self):
        """Excludes dino_encoder from pickling."""
        state = super().__getstate__()
        modules = state.get("_modules")
        if modules is not None and "dino_encoder" in modules:
            state = state.copy()
            modules = modules.copy()
            del modules["dino_encoder"]
            state["_modules"] = modules
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if (
            self.__dict__.get("load_dino", False)
            and "dino_encoder" not in self._modules
        ):
            self._build_dino_encoder()

    def forward(self, batch, *args, **kwargs):

        with torch.no_grad():
            if self.training:
                batch["timesteps"] = torch.sigmoid(
                    torch.randn(
                        len(batch["clean_imgs"]),
                        device=batch["clean_imgs"].device,
                        dtype=batch["clean_imgs"].dtype,
                    )
                    * self.P_std
                    + self.P_mean
                )
                conditions = batch["conditions"]
                drop_ids = (
                    torch.rand(
                        len(conditions),
                        device=conditions.device,
                    )
                    < self.unconditional_cond_drop_prob
                )

                batch["conditions"] = torch.where(
                    drop_ids, self.null_condition, conditions
                )

            else:
                batch["timesteps"] = torch.as_tensor(
                    np.linspace(
                        start=0.0,
                        stop=1.0,
                        num=len(batch["clean_imgs"]),
                        endpoint=False,
                    ),
                    dtype=batch["clean_imgs"].dtype,
                    device=batch["clean_imgs"].device,
                )

            ## Loading dino features if necessary ##
            if self.load_dino:
                features = self.dino_encoder.forward_features(
                    batch["alignment_features"]
                )
                batch["alignment_features"] = features["x_norm_patchtokens"]

            return batch


class PixelPostprocessing(nn.Module):
    """Implements pixel space postprocessing for images."""

    def __init__(self):
        """Constructor."""
        super().__init__()

    def forward(self, x):
        """Postprocesses the images."""
        with torch.no_grad():
            x = x.clamp(-1, 1)
            x = x * 0.5 + 0.5
        return x
