# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Common layers for defining score networks.
"""
import functools
from functools import partial
import math
import string
from typing import Any, Sequence, Optional

# import flax.linen as nn
import flax.nnx as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp

from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

from functools import partial

# Pytorch-like initialization
torch_linear = partial(nn.Dense, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)

def safe_split(rng):
    if rng is None:
        return None, None
    else:
        return jax.random.split(rng)

class PermutationFlip:
    """Permutation and flip for images."""

    def __init__(self, flip: bool = True):
        self.flip = flip

    def __call__(self, x: jnp.ndarray, inverse: bool = False):
        # inverse: legacy, not used.
        if self.flip:
            x = jnp.flip(x, axis=1)
        return x

def sqa_attention(q, k, v, mask, train, dropout_module=None, dropout_rng=None, dtype=jnp.float32):
    """
    Compute the attention weights and output.
    sanity check with nn.dot_product_attention
    qkv shape: [B, T, num_head, head_dim]
    """
    if train: assert dropout_rng is not None

    head_dim = q.shape[-1]
    q = q / jnp.sqrt(head_dim).astype(dtype)
    dot_product = jnp.einsum('...qhd, ...khd->...hqk', q, k)

    # deal with mask
    big_neg = jnp.finfo(dtype).min # to avoid -inf
    dot_product = jnp.where(mask, dot_product, big_neg)
    
    attn_weights = nn.softmax(dot_product, axis=-1).astype(dtype)
    
    # Apply dropout on attn_weights
    if train:
        attn_weights = dropout_module(attn_weights, rng=dropout_rng, deterministic=False)
    
    # Compute the output
    output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, v)
    
    return output

class Attention(nn.Module):
    """Attention mechanism."""

    in_channels: int
    num_heads: int
    dropout: float = 0.0
    dtype: Any = jnp.float32
    
    def setup(self):
        self.head_dim = self.in_channels // self.num_heads
        assert self.head_dim * self.num_heads == self.in_channels, "in_channels must be divisible by num_heads"

        self.qkv = torch_linear(features=3 * self.in_channels, dtype=self.dtype)
        self.proj = torch_linear(features=self.in_channels, dtype=self.dtype)
        self.norm = nn.LayerNorm(dtype=self.dtype)
        self.attndrop = nn.Dropout(rate=self.dropout)
        self.projdrop = nn.Dropout(rate=self.dropout)
        
    def __call__(self, 
                 x: jnp.ndarray, 
                 mask, 
                 k_cache: dict | None = None, 
                 v_cache: dict | None = None, 
                 temp: float = 1.0, 
                 which_cache: str = 'cond', 
                 train: bool = True,
                 rng = None):
        
        B, T, C = x.shape
        x = self.norm(x.astype(self.dtype))
        qkv = self.qkv(x) # [B, T, 3 * C]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.num_heads, self.head_dim) / temp
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)
        # [B, T, H, D]
        
        if not train and k_cache is not None and v_cache is not None:
            k_idx = k_cache["idx"]
            v_idx = v_cache["idx"]
            k_cache[which_cache] = jax.lax.dynamic_update_slice(k_cache[which_cache], k, (0, k_idx, 0, 0))
            v_cache[which_cache] = jax.lax.dynamic_update_slice(v_cache[which_cache], v, (0, v_idx, 0, 0))
            k = k_cache[which_cache]
            v = v_cache[which_cache]
            assert mask is not None, 'this KV cache impl. is different and require attn mask'
        else:
            assert k_cache is None and v_cache is None, "k_cache and v_cache must be None during training"
  
        rng, rng_used = safe_split(rng)
        
        # attn = nn.dot_product_attention(query=q, key=k, value=v, mask=mask, dropout_rng=rng_used, deterministic=not train, precision=None, dropout_rate=self.dropout, broadcast_dropout=False, dtype=self.dtype)
        attn = sqa_attention(q, k, v, mask=mask, train=train, dropout_module=self.attndrop, dropout_rng=rng_used, dtype=self.dtype)
        del rng_used
        attn = attn.reshape(B, T, C)
        attn = self.proj(attn) # [B, T, C]

        rng, rng_used = safe_split(rng)
        attn = self.projdrop(attn, rng=rng_used, deterministic=not train)
        return attn, k_cache, v_cache

class MLP(nn.Module):
    """MLP."""

    # in_channels: int
    out_channels: int
    dropout: float = 0.0
    dtype: Any = jnp.float32

    def setup(self):
        self.norm = nn.LayerNorm(dtype=self.dtype)
        self.net1 = torch_linear(features=4*self.out_channels, dtype=self.dtype)
        self.net2 = torch_linear(features=self.out_channels, dtype=self.dtype)
        self.drop = nn.Dropout(rate=self.dropout)

    def __call__(self, x: jnp.ndarray, train: bool = True, rng = None):
        # print("mlp:", x.shape, self.out_channels)
        x = self.norm(x.astype(self.dtype))
        x = self.net1(x)
        rng, rng_used = safe_split(rng)
        x = self.drop(x, deterministic=not train, rng=rng_used)
        del rng_used
        x = jax.nn.gelu(x, approximate=False)
        x = self.net2(x)
        rng, rng_used = safe_split(rng)
        x = self.drop(x, deterministic=not train, rng=rng_used)
        del rng_used
        return x
    
class AttentionBlock(nn.Module):
    """Attention block."""

    num_heads: int
    channels: int
    dropout: float = 0.0
    dtype: Any = jnp.float32

    def setup(self):
        self.attn = Attention(in_channels=self.channels, num_heads=self.num_heads, dtype=self.dtype, dropout=self.dropout)
        self.mlp = MLP(out_channels=self.channels, dtype=self.dtype, dropout=self.dropout)

    def __call__(self, 
                 x: jnp.ndarray, 
                 k_cache: dict | None = None,
                 v_cache: dict | None = None,
                 mask: jnp.ndarray | None = None,
                 temp: float = 1.0, 
                 which_cache: str = 'cond', 
                 train: bool = True,
                 rng = None):
        rng, rng_used = safe_split(rng)
        attn, k_cache, v_cache = self.attn(x, mask, k_cache=k_cache, v_cache=v_cache, temp=temp, which_cache=which_cache, train=train, rng=rng_used)
        del rng_used
        x = x + attn
        rng, rng_used = safe_split(rng)
        x = x + self.mlp(x, train=train, rng=rng_used)
        return x, k_cache, v_cache


# next cell prediction is not supported for now. Please refer to lyy's github repo.

class MetaBlock(nn.Module):
    """Meta block."""

    in_channels: int
    channels: int
    num_patches: int
    num_layers: int
    num_heads: int
    num_classes: int
    permutation: PermutationFlip
    dropout: float = 0.0
    debug: bool = False
    mode: str = "same" # options: same, reverse

    dtype: Any = jnp.float32
    
    def setup(self):
        self.proj_in = torch_linear(features=self.channels, dtype=self.dtype)
        self.pos_emebdding = self.param('pos_embedding', 
                                        nn.initializers.normal(stddev=0.02), 
                                        (1, self.num_patches, self.channels))
        
        self.class_embedding = None
        if self.num_classes > 0:
            self.class_embedding = self.param('class_embedding',
                                    nn.initializers.normal(stddev=0.02),
                                    (self.num_classes, 1, self.channels))
        
        self.blocks = [AttentionBlock(num_heads=self.num_heads, channels=self.channels, dtype=self.dtype, dropout=self.dropout) for _ in range(self.num_layers)]
        
        self.proj_out = nn.Dense(features=2*self.in_channels, dtype=self.dtype, 
            kernel_init=nn.initializers.zeros if not self.debug else nn.initializers.normal(stddev=0.01),
        bias_init=nn.initializers.zeros) 
        self.attn_mask = lambda: jnp.tril(jnp.ones((self.num_patches, self.num_patches), dtype=jnp.bool))

    def forward_flatten(self,
                        x: jnp.ndarray,
                        y: jnp.ndarray | None = None,
                        temp: float = 1.0,
                        which_cache: str = 'cond',
                        train: bool = True,
                        rng = None):
        raise DeprecationWarning
        # for student use. image only forward. The input has been permuted.
        x_proj = self.proj_in(x)
        pos_embed = self.permutation(self.pos_emebdding)
        x_proj = x_proj + pos_embed
        if self.class_embedding is not None:
            if y is not None:
                mask = (y < 0).astype(jnp.float32).reshape(-1, 1, 1)
                class_embed = (1 - mask) * self.class_embedding[y] + mask * self.class_embedding.mean(axis=0)
                x_proj = x_proj + class_embed
            else:
                x_proj = x_proj + self.class_embedding.mean(axis=0)
                
        for block in self.blocks:
            rng, rng_used = safe_split(rng)
            x_proj, _, _ = block(x_proj, mask=self.attn_mask(), temp=temp, which_cache=which_cache, train=train, rng=rng_used)
            del rng_used
            
        x_proj = self.proj_out(x_proj) # [B, T, 2*C]
        x_proj = jnp.concatenate([jnp.zeros_like(x_proj[:, :1]), x_proj[:, :-1]], axis=1)
        alpha, mu = jnp.split(x_proj, 2, axis=-1) 
        assert self.mode == REV_ORDER_L2_EACH_BLOCK
        x_new = x * jnp.exp(alpha) + mu
        return x_new
        
    def forward(self, 
                x: jnp.ndarray, 
                y: jnp.ndarray | None = None,
                temp: float = 1.0, 
                which_cache: str = 'cond', 
                train: bool = True,
                rng = None):
        """
        Args:
            x: [B, T, C]
            y: [B]
            temp: scalar
            which_cache: str
            train: bool
        forward with mu, alpha, jacobian. The input will be permuted in this function.
        foward: *exp(alpha) + mu
        """
        x_in = self.permutation(x)
        x = self.proj_in(x_in)
        x = x + self.permutation(self.pos_emebdding) # [B, T, C]
        
        if self.class_embedding is not None:
            if y is not None:
                mask = (y < 0).astype(jnp.float32).reshape(-1, 1, 1)
                class_embed = (1 - mask) * self.class_embedding[y] + mask * self.class_embedding.mean(axis=0)
                x = x + class_embed
            else:
                x = x + self.class_embedding.mean(axis=0)
        
        for block in self.blocks:
            rng, rng_used = safe_split(rng)
            x, _, _ = block(x, mask=self.attn_mask(), temp=temp, which_cache=which_cache, train=train, rng=rng_used)
            del rng_used
        
        x = self.proj_out(x) # [B, T, 2*C]
        x = jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        alpha, mu = jnp.split(x, 2, axis=-1)
        if self.mode == "same":
            x_new = (x_in - mu) * jnp.exp(-alpha) # [B, T, C_in]
            log_jacob = - alpha.mean(axis=(1, 2))
        elif self.mode == "reverse":
            x_new = x_in * jnp.exp(alpha) + mu
            log_jacob = alpha.mean(axis=(1, 2))
        else: raise NotImplementedError(f"Unknown mode: {self.mode}")
        x_new = self.permutation(x_new, inverse=True)
        return x_new, log_jacob, alpha, mu

    def reverse_step(
        self, 
        x: jnp.ndarray, 
        i: int,
        y: jnp.ndarray | None = None, 
        k_cache: dict | None = None,
        v_cache: dict | None = None,
        temp: float = 1.0, 
        which_cache: str = 'cond', 
        train: bool = False,
    ):
        # NOTE: train=True is not supported
        """
        Args:
            x: [B, T, C]
            y: [B]
            temp: scalar
            which_cache: str
            train: bool
        """
        B, T, C = x.shape
        x_in = jax.lax.dynamic_slice(x, (0, i, 0), (x.shape[0], 1, x.shape[2]))
        assert x_in.shape == (B, 1, C)
        x = self.proj_in(x_in)
        pos_embed = self.permutation(self.pos_emebdding)
        assert pos_embed.shape == (1, T, x.shape[-1]), f'{pos_embed.shape}, {(1, T, x.shape[-1])}'
        x = x + jax.lax.dynamic_slice(pos_embed, (0, i, 0), (pos_embed.shape[0], 1, pos_embed.shape[2]))
        
        if self.num_classes > 0:
            if y is not None:
                class_embed = self.class_embedding[y]
            else:
                class_embed = self.class_embedding.mean(axis=0)
            x = x + class_embed
        
        for bi, block in enumerate(self.blocks):
            mask = self.attn_mask()
            mask = jax.lax.dynamic_slice(mask, (i, 0), (1, mask.shape[1]))
            x, k_cache[bi], v_cache[bi] = block(x, k_cache=k_cache[bi], v_cache=v_cache[bi], temp=temp, which_cache=which_cache, train=train, mask=mask)
        
        x = self.proj_out(x) # [B, 1, 2*C]
        mu, alpha = jnp.split(x, 2, axis=-1)
        return mu, alpha, k_cache, v_cache # [B, C_in]

    def reverse(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray | None = None,
        temp: float = 1.0,
        which_cache: str = 'cond',
        train: bool = False,
    ):
        """
        reverse: (x-mu) * exp(-alpha)
        """
        raise LookupError("没用到？")
        x = self.permutation(x)
        B, T, C = x.shape
        num_heads = self.num_heads
        head_dim = self.channels // num_heads
        k_cache, v_cache = [{'cond': jnp.full((B, T, num_heads, head_dim), fill_value=jnp.nan), 'uncond': jnp.full((B, T, num_heads, head_dim), fill_value=jnp.nan), "idx": 0} for _ in range(self.num_layers)], [{'cond': jnp.full((B, T, num_heads, head_dim), fill_value=1e8), 'uncond': jnp.full((B, T, num_heads, head_dim), fill_value=1e8), "idx": 0} for _ in range(self.num_layers)]
        
        def step_fn(i, vals):
            x_step, log_jacob, k_cache_step, v_cache_step = vals
            for obj in k_cache_step + v_cache_step:
                obj["idx"] = i
                
            alpha_i, mu_i, k_cache_step, v_cache_step = self.reverse_step(x_step, i, y, k_cache_step, v_cache_step, temp, which_cache, train)
            
            x_sliced = jax.lax.dynamic_slice(x_step, (0, (i+1), 0), (x_step.shape[0], 1, x_step.shape[2]))
            assert x_sliced.shape == (B, 1, C)
            assert self.mode == REV_ORDER_L2_EACH_BLOCK
            val = (x_sliced - mu_i) * jnp.exp(-alpha_i.astype(jnp.float32))
            log_jacob -= alpha_i.mean(axis=(1, 2))
            x_step = jax.lax.dynamic_update_slice(x_step, val, (0, (i+1), 0))
            return x_step, log_jacob, k_cache_step, v_cache_step
        
        tot_log_jacob = jnp.zeros((B,), dtype=jnp.float32)
        x, tot_log_jacob, _, _ = jax.lax.fori_loop(0, T - 1, step_fn, (x, tot_log_jacob, k_cache, v_cache))
        
        x = self.permutation(x, inverse=True)
        
        return x, tot_log_jacob
    
    def __call__(self, 
            x: jnp.ndarray, 
            y: jnp.ndarray | None = None,
            temp: float = 1.0, 
            which_cache: str = 'cond', 
            train: bool = True,
            rng = None):
        return self.forward(x, y, temp, which_cache, train, rng)
    
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Common layers for defining score networks.
"""
import functools
import math
import string
from typing import Any, Sequence, Optional

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp


def get_act(config):
    """Get activation functions from the config file."""

    if config.model.nonlinearity.lower() == "elu":
        return nn.elu
    elif config.model.nonlinearity.lower() == "relu":
        return nn.relu
    elif config.model.nonlinearity.lower() == "lrelu":
        return functools.partial(nn.leaky_relu, negative_slope=0.2)
    elif config.model.nonlinearity.lower() == "swish":
        return nn.swish
    else:
        raise NotImplementedError("activation function does not exist!")


def ncsn_conv1x1(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0):
    """1x1 convolution with PyTorch initialization. Same as NCSNv1/v2."""
    init_scale = 1e-10 if init_scale == 0 else init_scale
    kernel_init = jnn.initializers.variance_scaling(
        1 / 3 * init_scale, "fan_in", "uniform"
    )
    kernel_shape = (1, 1) + (x.shape[-1], out_planes)
    bias_init = lambda key, shape: kernel_init(key, kernel_shape)[0, 0, 0, :]
    output = nn.Conv(
        out_planes,
        kernel_size=(1, 1),
        strides=(stride, stride),
        padding="SAME",
        use_bias=bias,
        kernel_dilation=(dilation, dilation),
        kernel_init=kernel_init,
        bias_init=bias_init,
    )(x)
    return output


def default_init(scale=1.0):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return jnn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def ddpm_conv1x1(
    x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, name=None
):
    """1x1 convolution with DDPM initialization."""
    bias_init = jnn.initializers.zeros
    output = nn.Conv(
        out_planes,
        kernel_size=(1, 1),
        strides=(stride, stride),
        padding="SAME",
        use_bias=bias,
        kernel_dilation=(dilation, dilation),
        kernel_init=default_init(init_scale),
        bias_init=bias_init,
        name=name,
    )(x)
    return output


def ncsn_conv3x3(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0):
    """3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2."""
    init_scale = 1e-10 if init_scale == 0 else init_scale
    kernel_init = jnn.initializers.variance_scaling(
        1 / 3 * init_scale, "fan_in", "uniform"
    )
    kernel_shape = (3, 3) + (x.shape[-1], out_planes)
    bias_init = lambda key, shape: kernel_init(key, kernel_shape)[0, 0, 0, :]
    output = nn.Conv(
        out_planes,
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding="SAME",
        use_bias=bias,
        kernel_dilation=(dilation, dilation),
        kernel_init=kernel_init,
        bias_init=bias_init,
    )(x)
    return output


def ddpm_conv3x3(
    x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, name=None
):
    """3x3 convolution with DDPM initialization."""
    bias_init = jnn.initializers.zeros
    output = nn.Conv(
        out_planes,
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding="SAME",
        use_bias=bias,
        kernel_dilation=(dilation, dilation),
        kernel_init=default_init(init_scale),
        bias_init=bias_init,
        name=name,
    )(x)
    return output


###########################################################################
# Functions below are ported over from the NCSNv1/NCSNv2 codebase:
# https://github.com/ermongroup/ncsn
# https://github.com/ermongroup/ncsnv2
###########################################################################


class CRPBlock(nn.Module):
    """CRPBlock for RefineNet. Used in NCSNv2."""

    features: int
    n_stages: int
    act: Any = nn.relu

    @nn.compact
    def __call__(self, x):
        x = self.act(x)
        path = x
        for _ in range(self.n_stages):
            path = nn.max_pool(
                path, window_shape=(5, 5), strides=(1, 1), padding="SAME"
            )
            path = ncsn_conv3x3(path, self.features, stride=1, bias=False)
            x = path + x
        return x


class CondCRPBlock(nn.Module):
    """Noise-conditional CRPBlock for RefineNet. Used in NCSNv1."""

    features: int
    n_stages: int
    normalizer: Any
    act: Any = nn.relu

    @nn.compact
    def __call__(self, x, y):
        x = self.act(x)
        path = x
        for _ in range(self.n_stages):
            path = self.normalizer()(path, y)
            path = nn.avg_pool(
                path, window_shape=(5, 5), strides=(1, 1), padding="SAME"
            )
            path = ncsn_conv3x3(path, self.features, stride=1, bias=False)
            x = path + x
        return x


class RCUBlock(nn.Module):
    """RCUBlock for RefineNet. Used in NCSNv2."""

    features: int
    n_blocks: int
    n_stages: int
    act: Any = nn.relu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_blocks):
            residual = x
            for _ in range(self.n_stages):
                x = self.act(x)
                x = ncsn_conv3x3(x, self.features, stride=1, bias=False)
            x = x + residual

        return x


class CondRCUBlock(nn.Module):
    """Noise-conditional RCUBlock for RefineNet. Used in NCSNv1."""

    features: int
    n_blocks: int
    n_stages: int
    normalizer: Any
    act: Any = nn.relu

    @nn.compact
    def __call__(self, x, y):
        for _ in range(self.n_blocks):
            residual = x
            for _ in range(self.n_stages):
                x = self.normalizer()(x, y)
                x = self.act(x)
                x = ncsn_conv3x3(x, self.features, stride=1, bias=False)
            x += residual
        return x


class MSFBlock(nn.Module):
    """MSFBlock for RefineNet. Used in NCSNv2."""

    shape: Sequence[int]
    features: int
    interpolation: str = "bilinear"

    @nn.compact
    def __call__(self, xs):
        sums = jnp.zeros((xs[0].shape[0], *self.shape, self.features))
        for i in range(len(xs)):
            h = ncsn_conv3x3(xs[i], self.features, stride=1, bias=True)
            if self.interpolation == "bilinear":
                h = jax.image.resize(
                    h, (h.shape[0], *self.shape, h.shape[-1]), "bilinear"
                )
            elif self.interpolation == "nearest_neighbor":
                h = jax.image.resize(
                    h, (h.shape[0], *self.shape, h.shape[-1]), "nearest"
                )
            else:
                raise ValueError(f"Interpolation {self.interpolation} does not exist!")
            sums = sums + h
        return sums


class CondMSFBlock(nn.Module):
    """Noise-conditional MSFBlock for RefineNet. Used in NCSNv1."""

    shape: Sequence[int]
    features: int
    normalizer: Any
    interpolation: str = "bilinear"

    @nn.compact
    def __call__(self, xs, y):
        sums = jnp.zeros((xs[0].shape[0], *self.shape, self.features))
        for i in range(len(xs)):
            h = self.normalizer()(xs[i], y)
            h = ncsn_conv3x3(h, self.features, stride=1, bias=True)
            if self.interpolation == "bilinear":
                h = jax.image.resize(
                    h, (h.shape[0], *self.shape, h.shape[-1]), "bilinear"
                )
            elif self.interpolation == "nearest_neighbor":
                h = jax.image.resize(
                    h, (h.shape[0], *self.shape, h.shape[-1]), "nearest"
                )
            else:
                raise ValueError(f"Interpolation {self.interpolation} does not exist")
            sums = sums + h
        return sums


class RefineBlock(nn.Module):
    """RefineBlock for building NCSNv2 RefineNet."""

    output_shape: Sequence[int]
    features: int
    act: Any = nn.relu
    interpolation: str = "bilinear"
    start: bool = False
    end: bool = False

    @nn.compact
    def __call__(self, xs):
        rcu_block = functools.partial(RCUBlock, n_blocks=2, n_stages=2, act=self.act)
        rcu_block_output = functools.partial(
            RCUBlock,
            features=self.features,
            n_blocks=3 if self.end else 1,
            n_stages=2,
            act=self.act,
        )
        hs = []
        for i in range(len(xs)):
            h = rcu_block(features=xs[i].shape[-1])(xs[i])
            hs.append(h)

        if not self.start:
            msf = functools.partial(
                MSFBlock, features=self.features, interpolation=self.interpolation
            )
            h = msf(shape=self.output_shape)(hs)
        else:
            h = hs[0]

        crp = functools.partial(
            CRPBlock, features=self.features, n_stages=2, act=self.act
        )
        h = crp()(h)
        h = rcu_block_output()(h)
        return h


class CondRefineBlock(nn.Module):
    """Noise-conditional RefineBlock for building NCSNv1 RefineNet."""

    output_shape: Sequence[int]
    features: int
    normalizer: Any
    act: Any = nn.relu
    interpolation: str = "bilinear"
    start: bool = False
    end: bool = False

    @nn.compact
    def __call__(self, xs, y):
        rcu_block = functools.partial(
            CondRCUBlock,
            n_blocks=2,
            n_stages=2,
            act=self.act,
            normalizer=self.normalizer,
        )
        rcu_block_output = functools.partial(
            CondRCUBlock,
            features=self.features,
            n_blocks=3 if self.end else 1,
            n_stages=2,
            act=self.act,
            normalizer=self.normalizer,
        )
        hs = []
        for i in range(len(xs)):
            h = rcu_block(features=xs[i].shape[-1])(xs[i], y)
            hs.append(h)

        if not self.start:
            msf = functools.partial(
                CondMSFBlock,
                features=self.features,
                interpolation=self.interpolation,
                normalizer=self.normalizer,
            )
            h = msf(shape=self.output_shape)(hs, y)
        else:
            h = hs[0]

        crp = functools.partial(
            CondCRPBlock,
            features=self.features,
            n_stages=2,
            act=self.act,
            normalizer=self.normalizer,
        )
        h = crp()(h, y)
        h = rcu_block_output()(h, y)
        return h


class ConvMeanPool(nn.Module):
    """ConvMeanPool for building the ResNet backbone."""

    output_dim: int
    kernel_size: int = 3
    biases: bool = True

    @nn.compact
    def __call__(self, inputs):
        output = nn.Conv(
            features=self.output_dim,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(1, 1),
            padding="SAME",
            use_bias=self.biases,
        )(inputs)
        output = (
            sum(
                [
                    output[:, ::2, ::2, :],
                    output[:, 1::2, ::2, :],
                    output[:, ::2, 1::2, :],
                    output[:, 1::2, 1::2, :],
                ]
            )
            / 4.0
        )
        return output


class MeanPoolConv(nn.Module):
    """MeanPoolConv for building the ResNet backbone."""

    output_dim: int
    kernel_size: int = 3
    biases: bool = True

    @nn.compact
    def __call__(self, inputs):
        output = inputs
        output = (
            sum(
                [
                    output[:, ::2, ::2, :],
                    output[:, 1::2, ::2, :],
                    output[:, ::2, 1::2, :],
                    output[:, 1::2, 1::2, :],
                ]
            )
            / 4.0
        )
        output = nn.Conv(
            features=self.output_dim,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(1, 1),
            padding="SAME",
            use_bias=self.biases,
        )(output)
        return output


class ResidualBlock(nn.Module):
    """The residual block for defining the ResNet backbone. Used in NCSNv2."""

    output_dim: int
    normalization: Any
    resample: Optional[str] = None
    act: Any = nn.elu
    dilation: int = 1

    @nn.compact
    def __call__(self, x):
        h = self.normalization()(x)
        h = self.act(h)
        if self.resample == "down":
            h = ncsn_conv3x3(h, h.shape[-1], dilation=self.dilation)
            h = self.normalization()(h)
            h = self.act(h)
            if self.dilation > 1:
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
                shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
            else:
                h = ConvMeanPool(output_dim=self.output_dim)(h)
                shortcut = ConvMeanPool(output_dim=self.output_dim, kernel_size=1)(x)
        elif self.resample is None:
            if self.dilation > 1:
                if self.output_dim == x.shape[-1]:
                    shortcut = x
                else:
                    shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
                h = self.normalization()(h)
                h = self.act(h)
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
            else:
                if self.output_dim == x.shape[-1]:
                    shortcut = x
                else:
                    shortcut = ncsn_conv1x1(x, self.output_dim)
                h = ncsn_conv3x3(h, self.output_dim)
                h = self.normalization()(h)
                h = self.act(h)
                h = ncsn_conv3x3(h, self.output_dim)

        return h + shortcut


class ConditionalResidualBlock(nn.Module):
    """The noise-conditional residual block for building NCSNv1."""

    output_dim: int
    normalization: Any
    resample: Optional[str] = None
    act: Any = nn.elu
    dilation: int = 1

    @nn.compact
    def __call__(self, x, y):
        h = self.normalization()(x, y)
        h = self.act(h)
        if self.resample == "down":
            h = ncsn_conv3x3(h, h.shape[-1], dilation=self.dilation)
            h = self.normalization(h, y)
            h = self.act(h)
            if self.dilation > 1:
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
                shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
            else:
                h = ConvMeanPool(output_dim=self.output_dim)(h)
                shortcut = ConvMeanPool(output_dim=self.output_dim, kernel_size=1)(x)
        elif self.resample is None:
            if self.dilation > 1:
                if self.output_dim == x.shape[-1]:
                    shortcut = x
                else:
                    shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
                h = self.normalization()(h, y)
                h = self.act(h)
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
            else:
                if self.output_dim == x.shape[-1]:
                    shortcut = x
                else:
                    shortcut = ncsn_conv1x1(x, self.output_dim)
                h = ncsn_conv3x3(h, self.output_dim)
                h = self.normalization()(h, y)
                h = self.act(h)
                h = ncsn_conv3x3(h, self.output_dim)

        return h + shortcut


###########################################################################
# Functions below are ported over from the DDPM codebase:
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
###########################################################################


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class NIN(nn.Module):
    num_units: int
    init_scale: float = 0.1

    @nn.compact
    def __call__(self, x):
        in_dim = int(x.shape[-1])
        W = self.param(
            "W", default_init(scale=self.init_scale), (in_dim, self.num_units)
        )
        b = self.param("b", jnn.initializers.zeros, (self.num_units,))
        y = contract_inner(x, W) + b
        assert y.shape == x.shape[:-1] + (self.num_units,)
        return y


def _einsum(a, b, c, x, y):
    einsum_str = "{},{}->{}".format("".join(a), "".join(b), "".join(c))
    return jnp.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[: len(x.shape)])
    y_chars = list(string.ascii_uppercase[: len(y.shape)])
    assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class AttnBlock(nn.Module):
    """Channel-wise self-attention block."""

    normalize: Any

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        h = self.normalize()(x)
        q = NIN(C)(h)
        k = NIN(C)(h)
        v = NIN(C)(h)

        w = jnp.einsum("bhwc,bHWc->bhwHW", q, k) * (int(C) ** (-0.5))
        w = jnp.reshape(w, (B, H, W, H * W))
        w = jax.nn.softmax(w, axis=-1)
        w = jnp.reshape(w, (B, H, W, H, W))
        h = jnp.einsum("bhwHW,bHWc->bhwc", w, v)
        h = NIN(C, init_scale=0.0)(h)
        return x + h


class Upsample(nn.Module):
    with_conv: bool = False

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        h = jax.image.resize(x, (x.shape[0], H * 2, W * 2, C), "nearest")
        if self.with_conv:
            h = ddpm_conv3x3(h, C)
        return h


class Downsample(nn.Module):
    with_conv: bool = False

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        if self.with_conv:
            x = ddpm_conv3x3(x, C, stride=2)
        else:
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        assert x.shape == (B, H // 2, W // 2, C)
        return x


class ResnetBlockDDPM(nn.Module):
    """The ResNet Blocks used in DDPM."""

    act: Any
    normalize: Any
    out_ch: Optional[int] = None
    conv_shortcut: bool = False
    dropout: float = 0.5

    @nn.compact
    def __call__(self, x, temb=None, train=True):
        B, H, W, C = x.shape
        out_ch = self.out_ch if self.out_ch else C
        h = self.act(self.normalize()(x))
        h = ddpm_conv3x3(h, out_ch)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += nn.Dense(out_ch, kernel_init=default_init())(self.act(temb))[
                :, None, None, :
            ]
        h = self.act(self.normalize()(h))
        h = nn.Dropout(self.dropout)(h, deterministic=not train)
        h = ddpm_conv3x3(h, out_ch, init_scale=0.0)
        if C != out_ch:
            if self.conv_shortcut:
                x = ddpm_conv3x3(x, out_ch)
            else:
                x = NIN(out_ch)(x)
        return x + h

