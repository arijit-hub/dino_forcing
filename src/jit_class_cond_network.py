"""Implements from scratch the JiT model.

Most of the code is shamelessly copy pasted from the JiT repo.
    - https://github.com/LTH14/JiT/blob/main/model_jit.py
"""

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
#                      RoPE                                #
############################################################
def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum("..., f -> ... f", t, freqs)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)

        freqs_w = torch.einsum("..., f -> ... f", t, freqs)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert (
            rot_dim <= t.shape[-1]
        ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        t_left, t, t_right = (
            t[..., :start_index],
            t[..., start_index:end_index],
            t[..., end_index:],
        )
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        num_cls_token=0,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])  # [N_img, D]
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()

            # prepend in-context cls token
            N_img, D = cos_img.shape
            cos_pad = torch.ones(
                num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device
            )
            sin_pad = torch.zeros(
                num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device
            )

            self.register_buffer(
                "freqs_cos", torch.cat([cos_pad, cos_img], dim=0)
            )  # [N_cls+N_img, D]
            self.register_buffer("freqs_sin", torch.cat([sin_pad, sin_img], dim=0))
        else:
            self.register_buffer("freqs_cos", freqs.cos().view(-1, freqs.shape[-1]))
            self.register_buffer("freqs_sin", freqs.sin().view(-1, freqs.shape[-1]))

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


############################################################
#                      RMSNorm                             #
############################################################
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


############################################################
#                      Attention Block with RoPE           #
############################################################
class AttentionBlockwithRope(nn.Module):
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
        do_qk_norm: bool = True,
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

        ## QK Normalization layer ##
        self.q_norm = RMSNorm(self.each_head_dim) if do_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.each_head_dim) if do_qk_norm else nn.Identity()

        ## Projection layer for the output of the attention ##
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        ## Dropout for the output of the attention ##
        self.projection_dropout = nn.Dropout(projection_dropout_prob)

    def forward(self, x, rope, c=None, mask=None, return_attn_weights=False):
        """Forward pass.

        x: torch.tensor. The tensor from which the queries are calculated. Shape: (b, t, d).
        rope: rope object.
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

        ## Applying RoPE ##
        q, k = rope(q), rope(k)

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
#                      SwigluFFN Block                     #
############################################################
class SwiGLUFFN(nn.Module):
    """Implements the SwiGLU feedforward block."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout_prob: float = 0.0):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()

    def forward(self, x):
        x = torch.nn.functional.silu(self.w1(x)) * self.w3(x)
        x = self.w2(self.dropout(x))
        return x


#############################################################
#      JiT Specific Blocks: (1) AdaLNZero JiT Block         #
#############################################################
class AdaLNZeroJiTBlock(nn.Module):
    """Implements the adaLN-Zero JiT block."""

    def __init__(
        self,
        embed_dim: int = 256,
        condition_dim: int = 256,
        # Self attention parameters
        sa_num_heads: int = 4,
        sa_weights_dropout_prob: float = 0.0,
        sa_projection_dropout_prob: float = 0.0,
        sa_block_bias: bool = True,
        sa_do_qk_norm: bool = True,
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
        # self.sa_ln = nn.RMSNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.sa_ln = RMSNorm(embed_dim)

        ## Self attention block ##
        self.img_self_attn = AttentionBlockwithRope(
            embed_dim=embed_dim,
            num_heads=sa_num_heads,
            attn_weights_dropout_prob=sa_weights_dropout_prob,
            projection_dropout_prob=sa_projection_dropout_prob,
            bias=sa_block_bias,
            do_qk_norm=sa_do_qk_norm,
        )

        ## Layer norm for the feedforward (without affine parameters) ##
        # self.ffn_ln = nn.RMSNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.ffn_ln = RMSNorm(embed_dim)

        ## Feed forward network ##
        self.ffn = SwiGLUFFN(
            embed_dim=embed_dim,
            hidden_dim=int(ffn_expansion_factor * embed_dim),
            dropout_prob=ffn_dropout_prob,
        )

    def forward(
        self, x, c, feat_rope, mask=None, return_attn_weights: bool = False, **kwargs
    ):
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
                h, rope=feat_rope, mask=mask, return_attn_weights=return_attn_weights
            )
        else:
            h = self.img_self_attn(h, rope=feat_rope)  # Shape: (b, t, d)
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
#      JiT Specific Blocks: (2) Final JiT Block             #
#############################################################
class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """

    def __init__(self, embed_dim):
        super().__init__()

        self.norm_final = RMSNorm(embed_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x


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
#               JiT Class Conditional Model                 #
#############################################################
class JiT(nn.Module):
    """Implements the JiT model."""

    def __init__(
        self,
        # base parameters
        num_blocks: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        patch_size: int = 16,
        in_channels: int = 3,
        out_channels: int = 3,
        resolution: int = 256,
        bottleneck_dim: int = 128,
        jit_sa_weights_dropout_prob: float = 0.0,
        jit_sa_projection_dropout_prob: float = 0.0,
        jit_sa_do_qk_norm: bool = True,
        jit_ffn_expansion_factor: int = 4,
        jit_ffn_dropout_prob: float = 0.0,
        num_classes: int = 1000,
        in_context_cls_length: int = 16,
        in_context_start_idx: int = 0,
        # self-cond parameters
        use_self_cond: bool = False,
        self_cond_in_dim: int = None,
        self_cond_out_dim: int = None,
        self_cond_drop_prob=0.0,
        self_cond_projector_dim=None,
        self_cond_layer_idx=None,
        use_layernorm_in_self_cond=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        ## 1. Base parameters ##
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.bottleneck_dim = bottleneck_dim
        self.jit_sa_weights_dropout_prob = jit_sa_weights_dropout_prob
        self.jit_sa_projection_dropout_prob = jit_sa_projection_dropout_prob
        self.jit_sa_do_qk_norm = jit_sa_do_qk_norm
        self.jit_ffn_expansion_factor = jit_ffn_expansion_factor
        self.jit_ffn_dropout_prob = jit_ffn_dropout_prob
        self.num_classes = num_classes
        self.in_context_cls_length = in_context_cls_length
        self.in_context_start_idx = in_context_start_idx

        ## 2. Self-conditioning parameters ##
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
        self.use_layernorm_in_self_cond = use_layernorm_in_self_cond

        ## Defining the layers ##
        ## 1. Time embedding layer ##
        self.time_embedder = TimeEmbedding(self.embed_dim)

        ## 2. Class embedding layer ##
        self.class_embedder = ClassEmbedding(
            num_classes=self.num_classes,
            embedding_dim=self.embed_dim,
        )

        ## 3. Patch embedding layer with bottleneck (including self-cond) ##
        in_dim = int(self.in_channels * self.patch_size**2)
        self.bottleneck_layer = (
            nn.Linear(in_dim, self.bottleneck_dim)
            if self.bottleneck_dim != in_dim
            else nn.Identity()
        )
        self.project_to_embed = nn.Linear(self.bottleneck_dim, self.embed_dim)

        ## 4. Positional embedding layer ##
        self.num_tokens = int((resolution // patch_size) ** 2)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, self.embed_dim), requires_grad=False
        )

        ## 5. Incontext cls token ##
        if self.in_context_cls_length > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_cls_length, self.embed_dim),
                requires_grad=True,
            )
            torch.nn.init.normal_(self.in_context_posemb, std=0.02)

        ## 6. RoPE ##
        half_head_dim = self.embed_dim // self.num_heads // 2
        hw_seq_len = self.resolution // self.patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_cls_length,
        )

        ## 7. JiT blocks ##
        self.jit_blocks = nn.ModuleList(
            [
                AdaLNZeroJiTBlock(
                    embed_dim=self.embed_dim,
                    condition_dim=self.embed_dim,
                    sa_num_heads=self.num_heads,
                    sa_weights_dropout_prob=(
                        self.jit_sa_weights_dropout_prob
                        if (self.num_blocks // 4 * 3 > i >= self.num_blocks // 4)
                        else 0.0
                    ),
                    sa_projection_dropout_prob=(
                        self.jit_sa_projection_dropout_prob
                        if (self.num_blocks // 4 * 3 > i >= self.num_blocks // 4)
                        else 0.0
                    ),
                    sa_do_qk_norm=self.jit_sa_do_qk_norm,
                    ffn_expansion_factor=self.jit_ffn_expansion_factor,
                    ffn_dropout_prob=(
                        self.jit_ffn_dropout_prob
                        if (self.num_blocks // 4 * 3 > i >= self.num_blocks // 4)
                        else 0.0
                    ),
                )
                for i in range(self.num_blocks)
            ]
        )

        ## 8. Final layer ##
        self.final_block = FinalLayer(self.embed_dim)

        ## 9. Output projection layer ##
        out_dim = int(self.out_channels * self.patch_size**2)
        self.project_to_out = nn.Linear(self.embed_dim, out_dim)

        ## 10. Self-conditioning layers ##
        if self.use_self_cond:
            self.self_cond_layer_norm = (
                nn.LayerNorm(
                    self.self_cond_in_dim,
                    elementwise_affine=False,
                )
                if use_layernorm_in_self_cond
                else nn.Identity()
            )

            self.fusion_projector = nn.Linear(
                self.embed_dim + self_cond_in_dim, self.embed_dim
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

        for i, block in enumerate(self.jit_blocks):
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
        """Simple forward JiT pass"""

        ## Getting the time embedding ##
        t = self.time_embedder(batch["timesteps"])

        ## Getting the class embedding ##
        c = self.class_embedder(batch["conditions"])

        ## Combining the two for main adaln conditioning ##
        y = t + c

        b, _, h, w = batch["noisy_imgs"].shape

        ## Getting the image tokens ##
        x = self._img2tokens(batch["noisy_imgs"])

        ## Doing the bottleneck projection ##
        x = self.bottleneck_layer(x)

        ## Project to embed ##
        x = self.project_to_embed(x)

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

            x = torch.cat([x, diff], dim=-1)
            x = self.fusion_projector(x)

        ## Add pos embedding ##
        x = x + self.position_embedding

        ## Passing through the JiT blocks ##
        for idx, block in enumerate(self.jit_blocks):
            ## in-context stuff ##
            if self.in_context_cls_length > 0 and idx == self.in_context_start_idx:
                in_context_tokens = c.unsqueeze(1).expand(
                    -1, self.in_context_cls_length, -1
                )
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)

            x = block(
                x,
                y,
                feat_rope=(
                    self.feat_rope
                    if idx < self.in_context_start_idx
                    else self.feat_rope_incontext
                ),
            )

            if self.use_self_cond and idx == self.self_cond_layer_idx:
                if self.in_context_cls_length > 0 and idx >= self.in_context_start_idx:
                    batch["self_condition"] = self.self_cond_block(
                        x[:, self.in_context_cls_length :]
                    )
                else:
                    batch["self_condition"] = self.self_cond_block(x)

        ## Final layer ##
        x = x[:, self.in_context_cls_length :]
        x = self.final_block(x, y)

        ## Project to output ##
        x = self.project_to_out(x)
        batch["out"] = self._tokens2img(x)
        return batch