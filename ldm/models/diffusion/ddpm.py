"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import numpy as np
from functools import partial

from ldm.util import default, count_params, instantiate_from_config

from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule

# xxxx1111
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from models.local_adapter import LocalControlUNetModel

import pdb


class DDPM(nn.Module):
    '''uni_v15.yaml

    unet_config:
      target: models.local_adapter.LocalControlUNetModel
      params:
        image_size: 32
        in_channels: 4
        model_channels: 320
        out_channels: 4
        num_res_blocks: 2
        attention_resolutions: [4, 2, 1]
        channel_mult: [1, 2, 4, 4]
        use_checkpoint: True
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        legacy: False
    '''

    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 version="v1.5",
                 # unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 ignore_keys=[],
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 ):
        super().__init__()
        # unet_config
        # unet_config = {'target': 'models.local_adapter.LocalControlUNetModel', 
        #     'params': {
        #         'image_size': 32, 
        #         'in_channels': 4, 
        #         'model_channels': 320, 
        #         'out_channels': 4, 
        #         'num_res_blocks': 2, 
        #         'attention_resolutions': [4, 2, 1], 
        #         'channel_mult': [1, 2, 4, 4], 
        #         'use_checkpoint': True, 
        #         'num_heads': 8, 
        #         'use_spatial_transformer': True, 
        #         'transformer_depth': 1, 
        #         'context_dim': 768, 
        #         'legacy': False}
        # }


        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.model = DiffusionWrapper(version=version) # unet_config, conditioning_key)
        count_params(self.model, verbose=True)

        self.v_posterior = v_posterior

        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=1e-4, linear_end=2e-2)

        logvar = torch.full(fill_value=0.0, size=(self.num_timesteps,))
        self.register_buffer('logvar', logvar)


    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=1e-4, linear_end=2e-2)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = 1e-4
        self.linear_end = 2e-2
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # def forward(self, x, *args, **kwargs):
    #     t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
    #     return self.p_losses(x, t, *args, **kwargs)


class LatentDiffusion(DDPM):
    """main class"""

    def __init__(self,
                version="v1.5",
                # first_stage_config,
                # cond_stage_config,
                num_timesteps_cond=1,
                cond_stage_key="image",
                cond_stage_trainable=False,
                # concat_mode=True,
                conditioning_key=None,
                scale_factor=0.18215,
                # scale_by_std=False,
                # *args, **kwargs
                ):
        # first_stage_config
        # {'target': 'ldm.models.autoencoder.AutoencoderKL', 'params': {'embed_dim': 4, 'monitor': 'val/rec_loss', 'ddconfig': {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}, 'lossconfig': {'target': 'torch.nn.Identity'}}}
        # (Pdb) cond_stage_config
        # {'target': 'ldm.modules.encoders.modules.FrozenCLIPEmbedder'}
        # conditioning_key -- 'crossattn'

        # ignore_keys = kwargs.pop("ignore_keys", [])

        super().__init__(version=version)
        # self.concat_mode = concat_mode
        # self.cond_stage_trainable = cond_stage_trainable
        # self.cond_stage_key = cond_stage_key

        # if not scale_by_std:
        #     self.scale_factor = scale_factor
        # else:
        #     self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.version = version
        self.scale_factor = scale_factor
        self.instantiate_first_stage()
        self.instantiate_cond_stage()


    def register_schedule(self,
                          beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2):
        super().register_schedule(beta_schedule, timesteps, linear_start, linear_end)

    def instantiate_first_stage(self):
        model = AutoencoderKL(version=self.version) #instantiate_from_config(config)
        self.first_stage_model = model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self):
        model = FrozenCLIPEmbedder() # instantiate_from_config(config)
        self.cond_stage_model = model.eval()
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

    def get_learned_conditioning(self, c):
        # pdb.set_trace()
        c = self.cond_stage_model.encode(c)
        return c

    @torch.no_grad()
    def decode_first_stage(self, z):
        # ==> come here
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    # @torch.no_grad()
    # def encode_first_stage(self, x):
    #     return self.first_stage_model.encode(x)


class DiffusionWrapper(nn.Module):
    '''uni_v15.yaml

    unet_config:
      target: models.local_adapter.LocalControlUNetModel
      params:
        image_size: 32
        in_channels: 4
        model_channels: 320
        out_channels: 4
        num_res_blocks: 2
        attention_resolutions: [4, 2, 1]
        channel_mult: [1, 2, 4, 4]
        use_checkpoint: True
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        legacy: False
    '''

    def __init__(self, version="1.5"): # unet_config, conditioning_key='crossattn'):
        super().__init__()
        self.diffusion_model = LocalControlUNetModel(version=version) #instantiate_from_config(unet_config) # models.local_adapter.LocalControlUNetModel
        # self.conditioning_key = conditioning_key # 'crossattn'
        # assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm', 'hybrid-adm', 'crossattn-adm']
