from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

from functools import partial

from utils.logging_utils import log_for_0
from models.jcm.layers import safe_split, MetaBlock, PermutationFlip
from models.models_ncsnpp import NCSNpp

ModuleDef = Any
# print = lambda *args, **kwargs : None

############################

def get_map_fn(map_method, teacher_nblocks, self_nblocks):
    map_fn = None
    log_for_0(f"Use map method: {map_method}, teacher is {teacher_nblocks} blocks, student is {self_nblocks} blocks")
    if map_method == "skip":
        # method: skip
        assert teacher_nblocks % self_nblocks == 0, f"{teacher_nblocks} % {self_nblocks} != 0"
        ratio = teacher_nblocks // self_nblocks
        map_fn = lambda idx: idx * ratio
    elif map_method == "extreme":
        assert self_nblocks % 2 == 0, f"self_nblocks must be even, but got {self_nblocks}"
        half_layers = self_nblocks // 2
        map_fn = lambda idx: (
            idx if idx < half_layers else teacher_nblocks - self_nblocks + idx
        )
    elif map_method == "extreme2":
        assert self_nblocks % 2 == 0, f"self_nblocks must be even, but got {self_nblocks}"
        assert teacher_nblocks >= 2 * self_nblocks, f"teacher_nblocks must be larger than 2 * self_nblocks, but got {teacher_nblocks} and {self_nblocks}"
        half_layers = self_nblocks // 2
        map_fn = lambda idx: (
            idx * 2 if idx < half_layers else teacher_nblocks + 1 - self_nblocks * 2 + idx * 2
        )
    elif map_method == "last":
        map_fn = lambda idx: teacher_nblocks - self_nblocks + idx
    else:
        raise ValueError(f"Unknown load_pretrain_method: {map_method}")
    return map_fn

def edm_ema_scales_schedules(step, config, steps_per_epoch):
    # ema_halflife_kimg = 500  # from edm
    # ema_halflife_kimg = 50000  # from flow
    ema_halflife_kimg = config.training.get("ema_halflife_kimg", 50000)
    ema_halflife_nimg = ema_halflife_kimg * 1000

    ema_rampup_ratio = 0.05
    ema_halflife_nimg = jnp.minimum(
        ema_halflife_nimg, step * config.training.batch_size * ema_rampup_ratio
    )

    ema_beta = 0.5 ** (config.training.batch_size / jnp.maximum(ema_halflife_nimg, 1e-8))
    scales = jnp.ones((1,), dtype=jnp.int32)
    return ema_beta, scales


def reverse_block(
    params,
    block: MetaBlock, 
    x: jnp.ndarray, 
    y: jnp.ndarray | None = None,
    temp: float = 1.0, 
    which_cache: str = 'cond', 
    guidance: float = 0,
    train: bool = False):
    # NOTE: train=True is not supported
    """
    Args:
        x: [B, T, C]
        y: [B]
        temp: scalar
        which_cache: str
        train: bool
    reverse: (x-mu) * exp(-alpha)
    """
    x = block.permutation(x)
    B, T, C = x.shape
    num_heads = block.num_heads
    head_dim = block.channels // num_heads
    k_cache, v_cache = [{'cond': jnp.full((B, T, num_heads, head_dim), fill_value=jnp.nan), 'uncond': jnp.full((B, T, num_heads, head_dim), fill_value=jnp.nan), "idx": 0} for _ in range(block.num_layers)], [{'cond': jnp.full((B, T, num_heads, head_dim), fill_value=1e8), 'uncond': jnp.full((B, T, num_heads, head_dim), fill_value=1e8), "idx": 0} for _ in range(block.num_layers)]
    
    assert T == block.num_patches
    
    # loop over num_tokens.
    def step_fn(i, vals):
        x_step, k_cache_step, v_cache_step = vals
        for obj in k_cache_step + v_cache_step:
            obj["idx"] = i
        alpha_i, mu_i, k_cache_step, v_cache_step = block.apply(params, x_step, i, y, k_cache_step, v_cache_step, temp, "cond", train, method=block.reverse_step)
        
        if guidance > 0:
            alpha_i_uncond, mu_i_uncond, k_cache_step, v_cache_step = block.apply(params, x_step, i, None, k_cache_step, v_cache_step, temp, "uncond", train, method=block.reverse_step)
            
            alpha_i = (1 + guidance * i / (T - 1)) * alpha_i - guidance * i / (T - 1) * alpha_i_uncond
            mu_i = (1 + guidance * i / (T - 1)) * mu_i - guidance * i / (T - 1) * mu_i_uncond
        
        x_sliced = jax.lax.dynamic_slice(x_step, (0, i+1, 0), (x_step.shape[0], 1, x_step.shape[2]))
        assert x_sliced.shape == (B, 1, C)
        if block.mode == "same":
            val = x_sliced * jnp.exp(alpha_i.astype(jnp.float32)) + mu_i
        elif block.mode == "reverse":
            val = (x_sliced - mu_i) * jnp.exp(-alpha_i.astype(jnp.float32))
        else:
            raise ValueError(f"Unknown mode: {block.mode}")
        x_step = jax.lax.dynamic_update_slice(x_step, val, (0, i+1, 0))
        return x_step, k_cache_step, v_cache_step
    
    x, _, _ = jax.lax.fori_loop(0, T-1, step_fn, (x, k_cache, v_cache))
    return block.permutation(x, inverse=True)

def reverse_block_student(
    params,
    block: MetaBlock, 
    x: jnp.ndarray, 
    y: jnp.ndarray | None = None,
    temp: float = 1.0, 
    which_cache: str = 'cond', 
    guidance: float = 0,
    guidance_method: str = "x", # options: ['x', 'ma']
    train: bool = False):
    
    x_out, _, alpha, mu = block.apply(params, x, y, temp, which_cache, train, method=block.forward)
    if guidance > 0:
        if guidance_method == "x":
            x_uncond, _, _, _ = block.apply(params, x, None, temp, which_cache, train, method=block.forward)
            x_out = (1 + guidance) * x_out - guidance * x_uncond # simple guidance.
        elif guidance_method == "ma":
            B, T, C = x.shape
            # guidance = guidance * jnp.arange(0, T, dtype=jnp.float32) / (T - 1)
            guidance = guidance * jnp.ones((T,), dtype=jnp.float32)
            guidance = guidance.reshape(1, T, 1)
            x_uncond, _, alpha_uncond, mu_uncond = block.apply(params, x, None, temp, which_cache, train, method=block.forward)
            alpha = (1 + guidance) * alpha - guidance * alpha_uncond
            mu = (1 + guidance) * mu - guidance * mu_uncond
            assert block.mode == "reverse"
            x_in = block.permutation(x)
            x_out = x_in * jnp.exp(alpha) + mu
            x_out = block.permutation(x_out, inverse=True)
    
    return x_out
class UNetStudent(nn.Module):
    """
    UNet student model.
    """
    img_size: int = 32
    patch_size: int = 4
    out_channels: int = 4
    base_width: int = 32
    num_classes: int = 1000
    num_unets: int = 1
    time_cond_method: str = "base" # options: base, cond, wot
    dtype: Any = jnp.float32
    dropout: float = 0.0

    def patchify(self, x):
        B, H, W, C = x.shape
        assert C == self.out_channels, "input must be RGB image"
        x = x.reshape(B, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, self.num_patches, self.patch_size * self.patch_size * C)
        return x
    
    def unpatchify(self, x):
        B, T, C = x.shape
        # assert self.patch_size * self.patch_num == self.img_size, "img_size must be divisible by num_patches"
        x = x.reshape(B, self.img_size // self.patch_size, self.img_size // self.patch_size, self.patch_size, self.patch_size, self.out_channels)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, self.img_size, self.img_size, self.out_channels)
        return x
    
    def setup(self):
        self.patch_num = patch_num = self.img_size // self.patch_size
        self.num_patches = patch_num ** 2
        if self.time_cond_method == 'base':
            assert False
            self.unets = [
                    NCSNpp(
                        base_width=self.base_width,
                        img_size=self.img_size,
                        dropout=self.dropout,
                        name=f"unet_{i}",
                    ) for i in range(self.num_unets)
                ]
        else:
            self.unets = [
                    NCSNpp(
                        base_width=self.base_width,
                        img_size=self.img_size,
                        dropout=self.dropout,
                        name=f"unet_0",
                    )
                ]

    def forward_with_sg(self,
                x: jnp.ndarray,
                y: jnp.ndarray | None = None,
                train: bool = True,
                rng = None,
        ):
            """
            used for student training.
            input: x, the output of teacher (latent)
            Return: a sequence of output of student, from latent to image
            """
            B, T, C0 = x.shape
            x = self.unpatchify(x) 

            B, H, W, C = x.shape
            assert C == self.out_channels, "input must be RGB image"
            assert H == W and H == self.img_size, "input must be square image"

            zs = []
            current_x = x
            labels = jnp.eye(self.num_classes)[y] if y is not None else None
            for i in range(self.num_unets):
                rng, rng_used = safe_split(rng)
                time_cond = i * jnp.ones((B,), dtype=jnp.float32)
                if self.time_cond_method == "wot" or self.time_cond_method == "base":
                    time_cond = 0 * time_cond
                if self.time_cond_method == "cond" or self.time_cond_method == "wot":
                    next_x = self.unets[0](current_x, time_cond=time_cond, labels=labels, train=train)
                else:
                    next_x = self.unets[i](current_x, time_cond=time_cond, labels=labels, train=train)
                zs.append(self.patchify(next_x))
                current_x = jax.lax.stop_gradient(next_x)
                del rng_used
            zs = jnp.stack(zs, axis=0)
            assert zs.shape == (self.num_unets, B, T, C0), f"zs shape: {zs.shape}"
            return zs
    
    # @nn.compact
    def __call__(self, x, y, *, guidance: float, train: bool, rng):
        """
        Args
        ----
        x        : (B, T, C0)  — patch tokens
        y        : (B,)        — class id 或 None
        guidance : float       — classifier‑free guidance scale
        train    : bool        — True=dropout/drop‑path on
        rng      : jax.random.PRNGKey
        """

        x = self.unpatchify(x)
        B, _, _, C = x.shape
        assert C == self.out_channels, "input must be RGB image"

        for i in range(self.num_unets):
            unet = self.unets[i % self.num_unets]
            rng, subkey = jax.random.split(rng)
            time_cond = jnp.full((B,), i, dtype=jnp.float32)

            if self.time_cond_method in {"wot", "base"}:
                time_cond = jnp.zeros_like(time_cond)

            x_cond = unet(
                x,
                time_cond = time_cond,
                labels    = y,
                train     = train,
                rng       = subkey
            )

            if guidance > 0:
                x_uncond = unet(
                    x,
                    time_cond = time_cond,
                    labels    = None,
                    train     = train,
                    rng       = subkey
                )
                x = (1 + guidance) * x_cond - guidance * x_uncond
            else:
                x = x_cond

        return self.patchify(x)


class UNetTeacherStudent(nn.Module):
    """
    UNet teacher-student model.
    """
    img_size: int
    out_channels: int
    channels: int
    patch_size: int
    time_cond_method: str = "base" # options: base, condt, wot
    num_layers: int = 8
    num_heads: int = 6
    num_blocks: int = 8
    num_classes: int = 1000
    base_width: int = 32
    dtype: Any = jnp.float32
    teacher_dropout: float = 0.0
    student_dropout: float = 0.0
    mode: str = "same" # options: same, reverse
    debug: bool = False
    prior_norm: float = 1.0 # not supported for now

    def setup(self):
        self.teacher = NormalizingFlow(
            img_size=self.img_size,
            out_channels=self.out_channels,
            channels=self.channels,
            patch_size=self.patch_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            # perms=self.perms,
            num_classes=self.num_classes,
            dtype=self.dtype,
            dropout=self.teacher_dropout,
            debug=self.debug,
            prior_norm=self.prior_norm,
        )
        assert self.num_blocks % 2 == 0, f"num_blocks must be even, but got {self.num_blocks}"
        num_unets = self.num_blocks // 2
        self.student = UNetStudent(
            patch_size=self.patch_size,
            img_size=self.img_size,
            out_channels=self.out_channels,
            base_width=self.base_width,
            num_classes=self.num_classes,
            num_unets=num_unets,
            time_cond_method=self.time_cond_method,
            dropout=self.student_dropout,
        )

    def run_student(self, z, y, *, guidance, train, rng):
        assert z.ndim == 3, f"input must be 3D tensor, but got {z.ndim}"
        return self.student(
            x=z, y=y, guidance=guidance, train=train, rng=rng    # 调子模块
        )
                    

    def patchify(self, x):
        B, H, W, C = x.shape
        assert C == self.out_channels, "input must be RGB image"
        x = x.reshape(B, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        patch_num = self.img_size // self.patch_size
        num_patches = patch_num ** 2
        x = x.reshape(B, num_patches, self.patch_size * self.patch_size * C)
        return x
        
    def unpatchify(self, x):
        B, T, C = x.shape
        x = x.reshape(B, self.img_size // self.patch_size, self.img_size // self.patch_size, self.patch_size, self.patch_size, self.out_channels)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, self.img_size, self.img_size, self.out_channels)
        return x
    
    def calc_teacher_forward(self, x, y, temp: float = 1.0, which_cache: str = 'cond', train: bool = True):
        # used for denoise. The loss here is logp.
        loss, _, _, _, _ = self.teacher(x, y, temp=temp, which_cache=which_cache, train=train)
        return loss

    def __call__(self, 
                x: jnp.ndarray, 
                y: jnp.ndarray | None = None, 
                # zs: jnp.ndarray | None = None,
                temp: float = 1.0, 
                which_cache: str = 'cond', 
                train: bool = True,
        ):
        """
        Student Train step.
        """
        
        rng = self.make_rng('dropout')
        rng, rng_used = safe_split(rng)
        rng, rng_used_2 = safe_split(rng)
        
        # generate zs by teacher, from dataset images
        loss_1, loss_dict_1, zs, _, _ = self.teacher(x, y, temp=temp, which_cache=which_cache, train=False, rng=rng_used)
        del rng_used
        loss_dict = {}

        # only implement REVERSE L2
        # here, the zs is from image to latent, num_blocks+1 z in total.
        zs = jax.lax.stop_gradient(zs)
        zs = jnp.flip(zs, axis=0) # flip to latent to image
        
        # xs, alphas, mus = self.student.forward_on_each_block(zs[:-1], y, temp=temp, which_cache=which_cache, train=train, rng=rng_used_2) # old loss
        xs = self.student.forward_with_sg(zs[0], y, train=train, rng=rng_used_2) # lyy's smart loss
        # xs: from latent (not contained) to image
        zs = zs[2::2] # two blocks per supervision
        
        losses = jnp.mean((xs - zs) ** 2, axis=(1, 2, 3))
        norm_x = jnp.mean(xs ** 2, axis=(1, 2, 3))
        norm_z = jnp.mean(zs ** 2, axis=(1, 2, 3))
        norm_x = jax.lax.stop_gradient(norm_x)
        norm_z = jax.lax.stop_gradient(norm_z)
        
        # this order is latent to image
        for i in range(len(losses)):
            loss_dict[f"block_{i}"] = losses[i]
        for i in range(len(norm_x)): # student
            loss_dict[f"norm_x_{i}"] = norm_x[i]
        for i in range(len(norm_z)): # teacher
            loss_dict[f"norm_z_{i}"] = norm_z[i]
            
        # losses /= (norm_x + norm_z)
        # losses *= jnp.mean(norm_x + norm_z)
        loss = jnp.sum(losses)
            
        loss_dict['loss'] = loss
        
        return loss, loss_dict, zs


class NormalizingFlow(nn.Module):
    """Normalizing flow."""
    
    img_size: int
    out_channels: int
    channels: int
    patch_size: int
    num_layers: int
    num_heads: int
    num_blocks: int
    reverse_perm: int = 0 # whether we should reverse the order of perms
    num_classes: int = 0
    # teacher_nblocks: int = None
    load_pretrain_method: str = "skip"
    dtype: Any = jnp.float32
    dropout: float = 0.0
    mode: str = "same" # options: same, reverse
    debug: bool = False
    prior_norm: float = 1.0 # not supported for now
    
    def setup(self):
        assert self.prior_norm == 1.0, f"prior_norm is not supported for now, but got {self.prior_norm}"
        self.patch_num = patch_num = self.img_size // self.patch_size
        self.num_patches = patch_num ** 2
        self.in_channels = self.out_channels * self.patch_size * self.patch_size
        assert self.patch_size * patch_num == self.img_size, "img_size must be divisible by num_patches"
        
        judge = lambda idx: (idx + self.reverse_perm) % 2 == 1
        
        log_for_0(f"judge result: {[judge(i) for i in range(self.num_blocks)]}")
        self.blocks = [
            MetaBlock(
                in_channels=self.in_channels, 
                channels=self.channels, 
                num_patches=self.num_patches, 
                num_layers=self.num_layers, 
                num_heads=self.num_heads, 
                num_classes=self.num_classes, 
                # permutation=self.permutation
                permutation=PermutationFlip(judge(i)),
                mode=self.mode,
                dropout=self.dropout,
                debug=self.debug,
            ) for i in range(self.num_blocks)
        ]
        
        self.prior_shape = (self.num_patches, self.in_channels)
        
    def patchify(self, x):
        B, H, W, C = x.shape
        assert C == self.out_channels, "input must be RGB image"
        x = x.reshape(B, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, self.num_patches, self.patch_size * self.patch_size * C)
        return x
    
    def unpatchify(self, x):
        B, T, C = x.shape
        # assert self.patch_size * self.patch_num == self.img_size, "img_size must be divisible by num_patches"
        x = x.reshape(B, self.img_size // self.patch_size, self.img_size // self.patch_size, self.patch_size, self.patch_size, self.out_channels)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, self.img_size, self.img_size, self.out_channels)
        return x
    
    def forward_with_sg(self,
                x: jnp.ndarray,
                y: jnp.ndarray | None = None,
                temp: float = 1.0,
                which_cache: str = 'cond',
                train: bool = True,
                rng = None,
        ):
            """
            used for student training.
            input: x, the output of teacher (latent)
            Return: a sequence of output of student, from latent to image
            """
            B, T, C = x.shape
            
            zs = []
            current_x = x
            
            for i in range(self.num_blocks):
                rng, rng_used = safe_split(rng)
                next_x, _, _, _ = self.blocks[i].forward(current_x, y, temp=temp, which_cache=which_cache, train=train, rng=rng_used)
                zs.append(next_x)
                current_x = jax.lax.stop_gradient(next_x) # stop grad before feeding into next block. TODO: try to remove this.
                del rng_used

            zs = jnp.stack(zs, axis=0)
            assert zs.shape == (self.num_blocks, B, T, C), f"zs shape: {zs.shape}, {B}, {T}, {C}"
            return zs
        
    def __call__(self, 
                x: jnp.ndarray, 
                y: jnp.ndarray | None = None, 
                temp: float = 1.0, 
                which_cache: str = 'cond', 
                train: bool = True,
                rng = jax.random.PRNGKey(0),
        ):
        """
        Args:
            x: [B, H, W, C]
            y: [B]
            temp: scalar
            which_cache: str
            train: bool
        """
        x = self.patchify(x)  # [B, T, C]
        B, T, C = x.shape
        xs = [x]
        alphas = []
        mus = []

        tot_logdet = 0.0
        for block in self.blocks:
            rng, rng_used = safe_split(rng)
            x, logdet, alpha, mu = block.forward(x, y, temp=temp, which_cache=which_cache, train=train, rng=rng_used)
            alphas.append(alpha)
            mus.append(mu)
            xs.append(x)
            tot_logdet = tot_logdet + logdet
            del rng_used

        xs = jnp.stack(xs, axis=0)
        alphas = jnp.stack(alphas, axis=0)
        mus = jnp.stack(mus, axis=0)

        assert xs.shape == (self.num_blocks+1, B, T, C)
        assert alphas.shape == (self.num_blocks, B, T, C)
        assert mus.shape == (self.num_blocks, B, T, C)
        
        log_prior = 0.5 * (x ** 2).mean() * (self.prior_norm ** -2.0)
        
        # TODO: impl. update prior
        
        loss = - tot_logdet.mean() + log_prior
        
        loss_dict = {'loss': loss, 'log_det': tot_logdet.mean(), 'log_prior': log_prior}
        
        return loss, loss_dict, xs, alphas, mus


# move this out from model for JAX compilation
def generate(params, model: UNetTeacherStudent, rng, n_sample, noise_level, guidance, guidance_method, temperature=1.0, label_cond=True, denoise=True, use_student=False):
    """
    Generate samples from the model.
    used for teacher generation.
    """
    patch_num = model.img_size // model.patch_size
    num_patches = patch_num ** 2
    in_channels = model.out_channels * model.patch_size * model.patch_size
    x_shape = (n_sample, num_patches, in_channels)
    rng_used, rng = jax.random.split(rng, 2)
    rng_used_2, rng = jax.random.split(rng, 2)
    rng_used_3, rng = jax.random.split(rng, 2)
    z = jax.random.normal(rng_used, x_shape, dtype=model.dtype) * model.prior_norm

    if label_cond:
        y = jax.random.randint(rng_used_2, (n_sample,), 0, model.num_classes)
    else:
        y = None

    x = model.apply({'params':params['params']}, z=z, y=y, guidance=guidance, train=False, rng=rng_used_3, method=model.run_student)
    
    # rev_fn = reverse_student if use_student else reverse
    # if label_cond:
    #     x = rev_fn(params, model, z, y, guidance=guidance, guidance_method=guidance_method, temp=1.0)
    # else:
    #     x = rev_fn(params, model, z, y, guidance=guidance, guidance_method=guidance_method, temp=temperature)
    
    if noise_level == 0 or not denoise: return x

    def nabla_log_prob(x):
        loss = model.apply(params, x, y, method=model.calc_teacher_forward)
        return loss
    
    nabla_log_prob = jax.grad(nabla_log_prob)
    grad_dir = nabla_log_prob(x) * jnp.prod(jnp.array(x.shape)) # because the loss is mean by teacher forward.
    
    x = x - (noise_level ** 2) * grad_dir # use score-based model to denoise
    
    return x
    
# class TeacherStudent(nn.Module):
#     """Normalizing flow, teacher-student model."""
    
#     img_size: int
#     out_channels: int
#     channels: int
#     patch_size: int
#     num_layers: int
#     num_heads: int
#     num_blocks: int
#     # perms: list[PermutationConfig]
#     num_classes: int = 0
#     dtype: Any = jnp.float32
#     teacher_dropout: float = 0.0
#     student_dropout: float = 0.0
#     mode: str = "same" # options: same, reverse
#     debug: bool = False
#     prior_norm: float = 1.0 # not supported for now
    
#     def setup(self):
#         assert self.prior_norm == 1.0, f"prior_norm is not supported for now, but got {self.prior_norm}"
#         assert self.mode == "reverse"
#         self.teacher = NormalizingFlow(
#             img_size=self.img_size,
#             out_channels=self.out_channels,
#             channels=self.channels,
#             patch_size=self.patch_size,
#             num_layers=self.num_layers,
#             num_heads=self.num_heads,
#             num_blocks=self.num_blocks,
#             # perms=self.perms,
#             num_classes=self.num_classes,
#             dtype=self.dtype,
#             dropout=self.teacher_dropout,
#             debug=self.debug,
#             prior_norm=self.prior_norm,
#         )
#         # the order of student is: first block corresponds to noise end. It is reverse of teacher.
#         self.student = NormalizingFlow(
#             img_size=self.img_size,
#             out_channels=self.out_channels,
#             channels=self.channels,
#             patch_size=self.patch_size,
#             num_layers=int(self.num_layers),
#             num_heads=self.num_heads,
#             num_blocks=self.num_blocks,
#             reverse_perm=self.num_blocks-1,
#             num_classes=self.num_classes,
#             dtype=self.dtype,
#             dropout=self.student_dropout,
#             mode=self.mode,
#             debug=self.debug,
#             prior_norm=self.prior_norm,
#         )
        
#     def patchify(self, x):
#         B, H, W, C = x.shape
#         assert C == self.out_channels, "input must be RGB image"
#         x = x.reshape(B, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size, C)
#         x = x.transpose(0, 1, 3, 2, 4, 5)
#         patch_num = self.img_size // self.patch_size
#         num_patches = patch_num ** 2
#         x = x.reshape(B, num_patches, self.patch_size * self.patch_size * C)
#         return x
        
#     def unpatchify(self, x):
#         B, T, C = x.shape
#         x = x.reshape(B, self.img_size // self.patch_size, self.img_size // self.patch_size, self.patch_size, self.patch_size, self.out_channels)
#         x = x.transpose(0, 1, 3, 2, 4, 5)
#         x = x.reshape(B, self.img_size, self.img_size, self.out_channels)
#         return x

#     def calc_teacher_forward(self, x, y, temp: float = 1.0, which_cache: str = 'cond', train: bool = True):
#         # used for denoise. The loss here is logp.
#         loss, _, _, _, _ = self.teacher(x, y, temp=temp, which_cache=which_cache, train=train)
#         return loss
    
#     def __call__(self, 
#                 x: jnp.ndarray, 
#                 y: jnp.ndarray | None = None, 
#                 # zs: jnp.ndarray | None = None,
#                 temp: float = 1.0, 
#                 which_cache: str = 'cond', 
#                 train: bool = True,
#         ):
#         """
#         Student Train step.
#         """
        
#         rng = self.make_rng('dropout')
#         rng, rng_used = safe_split(rng)
#         rng, rng_used_2 = safe_split(rng)
        
#         # generate zs by teacher, from dataset images
#         loss_1, loss_dict_1, zs, _, _ = self.teacher(x, y, temp=temp, which_cache=which_cache, train=False, rng=rng_used)
#         del rng_used
#         loss_dict = {}

#         # only implement REVERSE L2
#         # here, the zs is from image to latent, num_blocks+1 z in total.
#         zs = jax.lax.stop_gradient(zs)
#         zs = jnp.flip(zs, axis=0) # flip to latent to image
        
#         # xs, alphas, mus = self.student.forward_on_each_block(zs[:-1], y, temp=temp, which_cache=which_cache, train=train, rng=rng_used_2) # old loss
#         xs = self.student.forward_with_sg(zs[0], y, temp=temp, which_cache=which_cache, train=train, rng=rng_used_2) # lyy's smart loss
#         # xs: from latent (not contained) to image
        
#         losses = jnp.mean((xs - zs[1:]) ** 2, axis=(1, 2, 3))
#         norm_x = jnp.mean(xs ** 2, axis=(1, 2, 3))
#         norm_z = jnp.mean(zs[1:] ** 2, axis=(1, 2, 3))
#         norm_x = jax.lax.stop_gradient(norm_x)
#         norm_z = jax.lax.stop_gradient(norm_z)
        
#         # this order is latent to image
#         for i in range(len(losses)):
#             loss_dict[f"block_{i}"] = losses[i]
#         for i in range(len(norm_x)): # student
#             loss_dict[f"norm_x_{i}"] = norm_x[i]
#         for i in range(len(norm_z)): # teacher
#             loss_dict[f"norm_z_{i}"] = norm_z[i]
            
#         # losses /= (norm_x + norm_z)
#         # losses *= jnp.mean(norm_x + norm_z)
#         loss = jnp.sum(losses)
            
#         loss_dict['loss'] = loss
        
#         return loss, loss_dict, zs

    
# def reverse(params,
#             nf: TeacherStudent,
#             x: jnp.ndarray,
#             y: jnp.ndarray | None = None,
#             temp: float = 1.0,
#             guidance: float = 0,
#             guidance_method: str = "x", # options: ['x', 'ma']
#             which_cache: str = 'cond',
#             train: bool = False):
#     """
#     used for teacher generation.
#     """
#     assert guidance_method == "ma", f"only support x guidance, but got {guidance_method}"
#     # print('param keys:', params['params'].keys())
#     patch_num = nf.img_size // nf.patch_size
#     num_patches = patch_num ** 2
#     in_channels = nf.out_channels * nf.patch_size * nf.patch_size

#     for i in range(nf.num_blocks-1,-1,-1):
#         block_param = params['params']['teacher'][f'blocks_{i}']
#         # x = block.reverse(x, y, temp=temp, which_cache=which_cache, train=train)
#         x = reverse_block({"params": block_param}, MetaBlock(
#                 in_channels=in_channels, 
#                 channels=nf.channels, 
#                 num_patches=num_patches, 
#                 num_layers=nf.num_layers, 
#                 num_heads=nf.num_heads, 
#                 num_classes=nf.num_classes, 
#                 permutation=PermutationFlip(i % 2 == 1),
#                 debug=nf.debug,
#             ), x, y, temp=temp, which_cache=which_cache, train=train, guidance=guidance)
#         # print("mean during layer:", x.mean())
#     x = nf.unpatchify(x)
#     return x

# def reverse_student(params,
#             nf: TeacherStudent,
#             x: jnp.ndarray,
#             y: jnp.ndarray | None = None,
#             temp: float = 1.0,
#             guidance: float = 0,
#             guidance_method: str = "x", # options: ['x', 'ma']
#             which_cache: str = 'cond',
#             train: bool = False):
#     print('param keys:', params['params'].keys())
#     patch_num = nf.img_size // nf.patch_size
#     num_patches = patch_num ** 2
#     in_channels = nf.out_channels * nf.patch_size * nf.patch_size
#     assert nf.mode == "reverse", f"only support reverse, but got {nf.mode}"
#     # start loop. The order of student is the first block corresponds to noise end. It is reverse of teacher.
#     for i in range(nf.num_blocks):
#         block_param = params['params']['student'][f'blocks_{i}']
#         x = reverse_block_student({"params": block_param}, MetaBlock(
#                 in_channels=in_channels, 
#                 channels=nf.channels, 
#                 num_patches=num_patches, 
#                 num_layers=int(nf.num_layers), 
#                 num_heads=nf.num_heads, 
#                 num_classes=nf.num_classes, 
#                 permutation=PermutationFlip((nf.num_blocks-1-i)%2==1),
#                 mode=nf.mode,
#                 debug=nf.debug,
#             ), x, y, temp=temp, which_cache=which_cache, train=train, guidance=guidance, guidance_method=guidance_method)
#         # print("mean during layer:", x.mean())
#     x = nf.unpatchify(x)
#     return x

# def generate_prior(params, model: TeacherStudent, rng, n_sample, noise_level, guidance, temperature=1.0, label_cond=True):
#     # first teacher generate image, then flow using student to get latent. just for debug.
#     patch_num = model.img_size // model.patch_size
#     num_patches = patch_num ** 2
#     in_channels = model.out_channels * model.patch_size * model.patch_size
#     x_shape = (n_sample, num_patches, in_channels)
#     rng_used, rng = jax.random.split(rng, 2)
#     rng_used_2, rng = jax.random.split(rng, 2)
#     z = jax.random.normal(rng_used, x_shape, dtype=model.dtype)
#     if label_cond:
#         y = jax.random.randint(rng_used_2, (n_sample,), 0, model.num_classes)
#     else:
#         y = None
    
#     if label_cond:
#         x = reverse(params, model, z, y, guidance=guidance, temp=1.0)
#     else:
#         x = reverse(params, model, z, y, guidance=guidance, temp=temperature)
    
#     z = model.apply(params, x, method=model.patchify)
    
#     # student. from block n-1 to 0.
#     for i in range(model.num_blocks-1,-1,-1):
#         block_param = params['params']['student'][f'blocks_{i}']
#         z = reverse_block({"params": block_param}, MetaBlock(
#                 in_channels=in_channels, 
#                 channels=model.channels, 
#                 num_patches=num_patches, 
#                 num_layers=int(model.num_layers), 
#                 num_heads=model.num_heads, 
#                 num_classes=model.num_classes, 
#                 permutation=PermutationFlip((model.num_blocks-1-i)%2==1),
#                 debug=model.debug,
#                 mode=model.mode,
#         ), z, y)
        
#     z = model.unpatchify(z)
#     # x_recover = model.apply(params, z, y, method=model.calc_student_forward)
#     # x_recover = model.unpatchify(x_recover)
#     return x, z
    

NF_Debug = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=4, patch_size=2, num_layers=1, num_heads=1, num_blocks=1, debug=True,
)

NF_Base = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=768, patch_size=4, num_layers=4, num_heads=12, num_blocks=12,
)

NF_Small = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=4, num_layers=4, num_heads=6, num_blocks=12,
)

NF_Small_p2 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=2, num_layers=4, num_heads=6, num_blocks=12,
) # 91M

NF_Small_p2_b6 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=2, num_layers=4, num_heads=6, num_blocks=6,
)

NF_Small_p2_b2 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=2, num_layers=4, num_heads=6, num_blocks=2,
)

# sqa add
NF_Small_p2_b8_l8 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=2, num_layers=8, num_heads=6, num_blocks=8,
) # 118M

# for b4l8, 59M. DiT-B is 130M.

NF_Small_p2_b16_l4 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=2, num_layers=4, num_heads=6, num_blocks=16,
) # 121M

NF_Small_p2_b4_l16 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=2, num_layers=16, num_heads=6, num_blocks=4,
) # 115M

NF_Small_p4_b8_l8 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=4, num_layers=8, num_heads=6, num_blocks=8,
) # 117M

NF_2x_p2_b4_l4 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=768, patch_size=2, num_layers=4, num_heads=12, num_blocks=4,
) # 117M

NF_2x_p2_b8_l4 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=768, patch_size=2, num_layers=4, num_heads=12, num_blocks=8,
) # 234M

NF_Default = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=4, num_layers=8, num_heads=6, num_blocks=8,
)

# TSNF_Small_p2_b8_l8 = partial(
#     TeacherStudent, img_size=32, out_channels=4, channels=384, patch_size=2, num_layers=8, num_heads=6, num_blocks=8, mode="reverse",
# )

UNET = partial(
    UNetTeacherStudent, img_size=32, out_channels=4, channels=384, patch_size=2, num_layers=8, num_heads=6, num_blocks=8,
)

if __name__== "__main__":
    print = __builtins__.print
    jax.config.update("jax_default_matmul_precision", "float32")
    # jax.config.update("jax_disable_jit", True)
    model = NormalizingFlow(
        img_size=6,
        out_channels=3,
        channels=4,
        patch_size=2,
        num_layers=4,
        num_heads=2,
        num_blocks=4,
        num_classes=10,
        debug=True,
    )
    # model = NormalizingFlow(
    #     img_size=6,
    #     out_channels=3,
    #     channels=4,
    #     patch_size=2,
    #     num_layers=4,
    #     num_heads=2,
    #     num_blocks=4,
    #     num_classes=10,
    #     debug=True,
    # )
    # model = NormalizingFlow(
    #     img_size=6,
    #     out_channels=3,
    #     channels=4,
    #     patch_size=2,
    #     num_layers=1,
    #     num_heads=2,
    #     num_blocks=1,
    #     num_classes=10,
    #     debug=True,
    # )
    # variables = model.init(jax.random.PRNGKey(0), jnp.ones((2, 6, 6, 3),dtype=jnp.float32), jnp.ones((2,),dtype=jnp.int32))
    # x = jax.random.normal(jax.random.PRNGKey(0), (7, 6, 6, 3))
    # print(x.mean(axis=(1, 2, 3)))
    # # y = np.random.randint(0, 10, (7,))
    # y = jax.random.randint(jax.random.PRNGKey(0), (7,), 0, 10)
    # loss, _, z = model.apply(variables, x, y, rngs={"dropout": jax.random.PRNGKey(0)})
    # print('latent shape:', z.shape) # should be: (7, 9, 12)
    # print('is z nan?', jnp.isnan(z).any())
    # # back = reverse(variables, model, z, y)
    # loss, _, backagain = model.apply(variables, back, y, rngs={"dropout": jax.random.PRNGKey(0)})
    # # print(back.mean(axis=(1, 2, 3)))
    # print('back.shape', back.shape) # should be: (7, 6, 6, 3)
    # print('check x==back', ((x-back) ** 2).mean(axis=(1, 2, 3)))
    # print('check z==backagain', ((z-backagain) ** 2).mean(axis=(1, 2)))
    
    # print(jnp.allclose(x, back, atol=1e-5))

    # # test generate
    # x_gen = generate(variables, model, jax.random.PRNGKey(0), 7)
    # print(x_gen.shape) # should be: (7, 6, 6, 3)