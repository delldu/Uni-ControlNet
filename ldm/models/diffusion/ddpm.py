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

from ldm.util import count_params
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from ldm.modules.diffusionmodules.openaimodel import LocalControlUNetModel

import pdb

class DDPM(nn.Module):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self, version="v1.5", timesteps=1000):
        super().__init__()
        self.diffusion_model = LocalControlUNetModel(version=version)
        count_params(self.diffusion_model, verbose=True)
        self.register_schedule(timesteps=timesteps)

    def register_schedule(self, timesteps=1000):
        betas = make_beta_schedule(timesteps, linear_start=1e-4, linear_end=2e-2)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))


class LatentDiffusion(DDPM):
    def __init__(self, version="v1.5", scale_factor=0.18215):
        super().__init__(version=version)
        self.version = version
        self.scale_factor = scale_factor
        self.instantiate_first_stage()
        self.instantiate_cond_stage()

    def instantiate_first_stage(self):
        model = AutoencoderKL(version=self.version)
        self.first_stage_model = model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self):
        model = FrozenCLIPEmbedder()
        self.cond_stage_model = model.eval()
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

    # xxxx8888
    def get_learned_conditioning(self, c):
        c = self.cond_stage_model.encode(c)
        return c

    def decode_first_stage(self, z):
        # ==> come here
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    # def encode_first_stage(self, x):
    #     return self.first_stage_model.encode(x)
