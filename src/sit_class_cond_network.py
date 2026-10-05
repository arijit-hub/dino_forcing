"""Implements from scratch the SiT model."""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from einops.layers.torch import Rearrange

#############################################################
#       Sine/Cosine Positional Embedding Functions          #
#############################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


###########################################################
#                   Modulation for ADALN                  #
###########################################################
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


############################################################
#                       Time Embedding                     #
#############################################################
class TimeEmbedding(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


############################################################
#                       Class Embedding                    #
#############################################################
class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.class_embedding = nn.Embedding(num_classes + 1, embedding_dim)
        self.num_classes = num_classes

    def forward(self, labels, force_drop_ids=None):
        embeddings = self.class_embedding(labels)
        return embeddings


############################################################
#                      Attention Block                     #
############################################################
class AttentionBlock(nn.Module):
    """Implements the attention block. It has an option to do qk normalization."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_dim: int = None,
        k_dim: int = None,
        attn_weights_dropout_prob: float = 0.0,
        projection_dropout_prob: float = 0.0,
        bias: bool = True,
        do_qk_norm: bool = False,
    ):
        """Constructor.

        :param embed_dim: int: Embedding dimension.
        :param num_heads: int: Number of heads.
        :param q_dim: int: Query dimension. If None, it is set to embed_dim.
        :param k_dim: int: Key dimension. If None, it is set to embed_dim.
        :param attn_weights_dropout_prob: float: Dropout prob for the attention weights.
        :param projection_dropout_prob: float: Dropout prob for the projection.
        :param bias: bool: Whether to use bias or not.
        :param do_qk_norm: bool: Whether to do qk normalization or not.
        """
        super().__init__()

        ## Setting the attributes ##
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.each_head_dim = embed_dim // num_heads
        self.attn_weights_dropout_prob = attn_weights_dropout_prob

        ## Defining the layers ##
        ## Projection layer for queries, keys, and values ##
        self.q_proj = nn.Linear(
            embed_dim if q_dim == None else q_dim, embed_dim, bias=bias
        )
        self.kv_proj = nn.Linear(
            embed_dim if k_dim == None else k_dim, 2 * embed_dim, bias=bias
        )

        ## Layernorm of queries and keys ##
        self.q_norm = nn.LayerNorm(self.each_head_dim) if do_qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.each_head_dim) if do_qk_norm else nn.Identity()

        ## Projection layer for the output of the attention ##
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        ## Dropout for the output of the attention ##
        self.projection_dropout = nn.Dropout(projection_dropout_prob)

    def forward(self, x, c=None, mask=None, return_attn_weights=False):
        """Forward pass.

        x: torch.tensor. The tensor from which the queries are calculated. Shape: (b, t, d).
        c: torch.tensor. The tensor from which the keys and values are calculated. Shape: (b, t', d).
        mask: torch.BoolTensor. The key mask for the attention. A true signifies take part in attention. Shape: (b, t').

        :return: torch.tensor. The output of the attention block. Shape: (b, t, d).
        """

        if c is None:
            c = x

        ## Doing the projections and splitting ##
        q = self.q_proj(x)
        k, v = self.kv_proj(c).chunk(2, dim=-1)

        ## Reshaping the queries, keys, and values in head dim ##
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        ## Normalizing the queries and keys ##
        q = self.q_norm(q)
        k = self.k_norm(k)

        ## Extending the key padding mask ##
        if mask is not None:
            if mask.dim() == 2:
                mask = rearrange(mask, "b t -> b 1 1 t")
            elif mask.dim() == 3:
                mask = rearrange(mask, "b tq tkv -> b 1 tq tkv")
            else:
                raise ValueError(
                    f"Mask must be either 2D or 3D. Got shape: {mask.shape}"
                )

        ## Scaled dot product attention ##
        if return_attn_weights:
            scale = self.each_head_dim**-0.5
            q = q * scale
            attn = q @ rearrange(k, "b h n d -> b h d n")
            if mask is not None:
                attn_bias = torch.zeros_like(attn)
                attn = attn + attn_bias.masked_fill(~mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            ## we are not dropping the attention weights ##
            out = attn @ v
            attn = attn.mean(dim=1)  # Averaging over the heads

        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attn_weights_dropout_prob if self.training else 0.0,
            )

        ## Final projection ##
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        ## Applying the dropout ##
        out = self.projection_dropout(out) if self.training else out

        if return_attn_weights:
            return out, attn
        return out


############################################################
#                      MLP Block                           #
############################################################
class FeedForwardBlock(nn.Module):
    """Implements the feedforward block of the transformer."""

    def __init__(
        self, embed_dim: int, expansion_factor: int = 4, dropout_prob: float = 0.0
    ):
        """Constructor.

        :param embed_dim: int: Embedding dimension.
        :param expansion_factor: int: Expansion factor for the feedforward block.
        :param dropout_p: float: Dropout probability.
        """

        super().__init__()

        self.pff = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim * expansion_factor),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features=embed_dim * expansion_factor, out_features=embed_dim),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        """Forward pass.

        :param x: torch.tensor: The input tensor. Shape: (b, t, d).
        :return: torch.tensor: The output tensor. Shape: (b, t, d).
        """

        return self.pff(x)


############################################################
#                      Repa proj mlp block                 #
############################################################
class RepaProjectionBlock(nn.Module):
    """Implements the projection block used for REPA.
    Shamelessly copied from:
        https://github.com/sihyun-yu/REPA/blob/main/models/sit.py
    """

    def __init__(self, hidden_size, projector_dim, z_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, z_dim),
        )

    def forward(self, x):
        return self.projector(x)


#############################################################
#      SiT Specific Blocks: (1) AdaLNZero SiT Block         #
#############################################################
class AdaLNZeroSiTBlock(nn.Module):
    """Implements the adaLN-Zero SiT block."""

    def __init__(
        self,
        embed_dim: int = 256,
        condition_dim: int = 256,
        # Self attention parameters
        sa_num_heads: int = 4,
        sa_weights_dropout_prob: float = 0.0,
        sa_projection_dropout_prob: float = 0.0,
        sa_block_bias: bool = True,
        sa_do_qk_norm: bool = False,
        # Feed forward parameters
        ffn_expansion_factor: int = 4,
        ffn_dropout_prob: float = 0.0,
    ):
        """Constructor.

        :param embed_dim: int: Model's internal dimension.
        :param condition_dim: int: Dimension of the conditional tensor (e.g., time + class embeddings).
        :param sa_num_heads: int: Number of attention heads for the self attention.
        :param sa_weights_dropout_prob: float: Dropout probability for the attention weights.
        :param sa_projection_dropout_prob: float: Dropout probability for the attention projection.
        :param sa_block_bias: bool: Whether to use bias in the attention block.
        :param sa_do_qk_norm: bool: Whether to do normalization on the query and key.
        :param ffn_expansion_factor: int: Expansion factor for the feed forward network.
        :param ffn_dropout_prob: float: Dropout probability for the feed forward network.
        """

        super().__init__()

        self.condition_proj = (
            nn.Linear(in_features=condition_dim, out_features=embed_dim)
            if (condition_dim != embed_dim)
            else nn.Identity()
        )

        ## MLP to generate adaptive layer norm parameters ##
        # This will output 6 sets of parameters:
        # - shift_sa, scale_sa, gate_sa (for self-attention)
        # - shift_ffn, scale_ffn, gate_ffn (for feedforward)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim, bias=True),
        )

        ## Layer norm for the self attention (without affine parameters) ##
        self.sa_ln = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

        ## Self attention block ##
        self.img_self_attn = AttentionBlock(
            embed_dim=embed_dim,
            num_heads=sa_num_heads,
            attn_weights_dropout_prob=sa_weights_dropout_prob,
            projection_dropout_prob=sa_projection_dropout_prob,
            bias=sa_block_bias,
            do_qk_norm=sa_do_qk_norm,
        )

        ## Layer norm for the feedforward (without affine parameters) ##
        self.ffn_ln = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

        ## Feed forward network ##
        self.ffn = FeedForwardBlock(
            embed_dim=embed_dim,
            expansion_factor=ffn_expansion_factor,
            dropout_prob=ffn_dropout_prob,
        )

    def forward(self, x, c, mask=None, return_attn_weights: bool = False, **kwargs):
        """Forward pass with adaptive layer normalization.

        :param x: torch.tensor: Input tensor (image tokens). Shape: (b, t, d)
        :param c: torch.tensor: Conditional tensor. Shape: (b, d_c).
        :return: torch.tensor: Output tensor. Shape: (b, t, d)
        """

        ## Generate adaptive parameters ##
        # Shape: (b, 6*d) -> 6 separate (b, d) tensors
        c = self.condition_proj(c)  # Shape: (b, d)
        modulation = self.adaLN_modulation(c)
        shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = modulation.chunk(
            6, dim=-1
        )

        ## Self-attention block with adaLN ##
        h = self.sa_ln(x)  # Shape: (b, t, d)
        h = modulate(h, shift_sa, scale_sa)  # Shape: (b, t, d)
        if return_attn_weights:
            h, attn_weights = self.img_self_attn(
                h, mask=mask, return_attn_weights=return_attn_weights
            )
        else:
            h = self.img_self_attn(h)  # Shape: (b, t, d)
        x = x + gate_sa.unsqueeze(1) * h  # Gated residual

        ## Feedforward block with adaLN ##
        # Apply layer norm, then scale and shift
        h = self.ffn_ln(x)  # Shape: (b, t, d)
        h = modulate(h, shift_ffn, scale_ffn)  # Shape: (b, t, d)
        h = self.ffn(h)  # Shape: (b, t, d)
        x = x + gate_ffn.unsqueeze(1) * h  # Gated residual

        if return_attn_weights:
            return x, attn_weights.detach()
        return x


#############################################################
#      SiT Specific Blocks: (2) Final SiT Block             #
#############################################################
class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x


#############################################################
#               SiT Class Conditional Model                 #
#############################################################
class SiT(nn.Module):
    def __init__(
        self,
        # base parameters
        num_blocks=12,
        embed_dim=768,
        num_heads=12,
        patch_size=2,
        in_channels=4,
        out_channels=None,
        resolution=32,
        condition_dim=768,
        timesteps_scaling_factor=10000,
        sit_sa_do_qk_norm=False,
        sit_ffn_expansion_factor=4,
        num_classes=1000,
        sa_weights_dropout_prob=0.0,
        sa_projection_dropout_prob=0.0,
        ffn_dropout_prob=0.0,
        # self-cond parameters
        use_self_cond=False,
        self_cond_in_dim=None,
        self_cond_out_dim=None,
        self_cond_drop_prob=0.0,
        self_cond_projector_dim=None,
        self_cond_layer_idx=None,
        use_layernorm_in_self_cond=False,
        **kwargs,
    ):
        super().__init__()

        ## Saving all the attributes ##
        ## 1. Base parameters ##
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.resolution = resolution
        self.condition_dim = condition_dim
        self.timesteps_scaling_factor = timesteps_scaling_factor
        self.sit_sa_do_qk_norm = sit_sa_do_qk_norm
        self.sit_ffn_expansion_factor = sit_ffn_expansion_factor
        self.num_classes = num_classes
        ## 2. self_cond parameters ##
        self.use_self_cond = use_self_cond
        assert not use_self_cond or (
            self_cond_in_dim is not None
        ), "self_cond_in_dim must be provided if use_self_cond is True."
        self.self_cond_in_dim = self_cond_in_dim
        self.self_cond_out_dim = (
            self_cond_out_dim
            if self_cond_out_dim is not None
            else self.self_cond_in_dim
        )
        self.self_cond_drop_prob = self_cond_drop_prob
        self.self_cond_projector_dim = (
            self_cond_projector_dim
            if self_cond_projector_dim is not None
            else int(self.embed_dim * 4)
        )
        self.self_cond_layer_idx = (
            self_cond_layer_idx
            if self_cond_layer_idx is not None
            else self.num_blocks - 1
        )

        ## Defining the layers ##
        ## 1. time embedding ##
        self.time_embedder = TimeEmbedding(embed_dim)
        ## 2. condition embedding ##
        self.class_embedder = ClassEmbedding(
            num_classes=self.num_classes,
            embedding_dim=self.embed_dim,
            # unconditional_drop_prob=unconditional_drop_prob,
        )

        ## 3. patch embedding ##
        in_dim = int(self.in_channels * (self.patch_size**2))
        self.project_to_embed = (
            nn.Linear(in_dim, embed_dim, bias=True)
            if not self.use_self_cond
            else nn.Linear(in_dim + self.self_cond_in_dim, embed_dim, bias=True)
        )

        ## 4. positional embedding ##
        self.num_tokens = int((resolution // patch_size) ** 2)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, embed_dim), requires_grad=False
        )

        ## 5. SiT blocks ##
        self.sit_blocks = nn.ModuleList(
            [
                AdaLNZeroSiTBlock(
                    embed_dim=self.embed_dim,
                    condition_dim=self.condition_dim,
                    sa_num_heads=self.num_heads,
                    sa_weights_dropout_prob=(
                        sa_weights_dropout_prob
                        if (self.num_blocks // 4 * 3 > i >= self.num_blocks // 4)
                        else 0.0
                    ),
                    sa_projection_dropout_prob=(
                        sa_projection_dropout_prob
                        if (self.num_blocks // 4 * 3 > i >= self.num_blocks // 4)
                        else 0.0
                    ),
                    sa_do_qk_norm=self.sit_sa_do_qk_norm,
                    ffn_expansion_factor=self.sit_ffn_expansion_factor,
                    ffn_dropout_prob=(
                        ffn_dropout_prob
                        if (self.num_blocks // 4 * 3 > i >= self.num_blocks // 4)
                        else 0.0
                    ),
                )
                for i in range(self.num_blocks)
            ]
        )

        ## 6. final layer ##
        self.final_block = FinalLayer(self.embed_dim)

        ## 7. output projection ##
        out_dim = int(self.out_channels * (self.patch_size**2))
        self.project_to_out = nn.Linear(embed_dim, out_dim, bias=True)

        ## 8. self_cond layers ##
        if self.use_self_cond:
            self.self_cond_layer_norm = (
                nn.LayerNorm(
                    self.self_cond_in_dim,
                    elementwise_affine=False,
                )
                if use_layernorm_in_self_cond
                else nn.Identity()
            )

            self.self_cond_block = RepaProjectionBlock(
                hidden_size=self.embed_dim,
                projector_dim=self.self_cond_projector_dim,
                z_dim=self.self_cond_out_dim,
            )

        else:
            self.self_cond_block = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        ## Initializing the class embedding weights ##
        nn.init.normal_(self.class_embedder.class_embedding.weight, std=0.02)

        ## Initializing the positional embedding weights ##
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embedding.shape[-1],
            int(self.num_tokens**0.5),
        )
        self.position_embedding.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        ## Initialize the timestep mlp ##
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        for i, block in enumerate(self.sit_blocks):
            nn.init.constant_(block.adaLN_modulation[1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[1].bias, 0)

        nn.init.constant_(self.final_block.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.final_block.adaLN_modulation[1].bias, 0)
        nn.init.constant_(self.project_to_out.weight, 0)
        nn.init.constant_(self.project_to_out.bias, 0)

    def _get_null_condition(self):
        return self.num_classes

    def _img2tokens(self, img):
        x = rearrange(
            img,
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return x

    def _tokens2img(self, tokens):
        x = rearrange(
            tokens,
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=self.resolution // self.patch_size,
            w=self.resolution // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return x

    def forward(self, batch):
        """Forward pass."""

        ## 1. Getting the time embeddings ##
        t = self.time_embedder(batch["timesteps"])

        ## 2. Getting the class embeddings ##
        c = self.class_embedder(batch["conditions"])

        ## 3. Making y which is the sum of t and c ##
        y = t + c

        b, _, h, w = batch["noisy_imgs"].shape

        ## 4. Getting the image tokens ##
        x = self._img2tokens(batch["noisy_imgs"])

        if self.use_self_cond:
            diff = batch.get(
                "self_condition",
                torch.zeros(
                    b,
                    self.num_tokens,
                    self.self_cond_in_dim,
                    device=batch["noisy_imgs"].device,
                    dtype=batch["noisy_imgs"].dtype,
                ),
            )
            diff = self.self_cond_layer_norm(diff)

            if self.training and self.self_cond_drop_prob > 0.0:
                drop_mask = torch.rand(b, device=diff.device) < self.self_cond_drop_prob
                diff = torch.where(
                    drop_mask[:, None, None],
                    torch.zeros_like(diff),
                    diff,
                )

            x = torch.cat(
                [x, diff], dim=-1
            )  # Concatenate the self-conditioning features

        x = self.project_to_embed(x)  # Shape: (b, num_tokens, d)
        x = x + self.position_embedding  # Adding positional embedding

        ## 5. Passing through the SiT blocks ##
        for idx, block in enumerate(self.sit_blocks):
            x = block(x, y)

            ## self_cond output if necessary ##
            if self.use_self_cond and idx == self.self_cond_layer_idx:
                batch["self_condition"] = self.self_cond_block(x)

        ## 6. Final block ##
        x = self.final_block(x, y)

        ## 7. Projecting back to the image space ##
        x = self.project_to_out(x)
        batch["out"] = self._tokens2img(x)
        return batch
