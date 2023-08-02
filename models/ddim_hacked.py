"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
import pdb

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps


class DDIMSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps

    def make_schedule(self, ddim_num_steps, ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        # ddim sampling parameters
        self.ddim_sigmas, self.ddim_alphas, self.ddim_alphas_prev = \
            make_ddim_sampling_parameters(alphacums=self.model.alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta, verbose=verbose)
        self.ddim_sqrt_one_minus_alphas = np.sqrt(1. - self.ddim_alphas)


    @torch.no_grad()
    def sample(self,
               S, # 20
               batch_size, # 1
               shape, # (4, 80, 64)
               condition=None,
               eta=0.,
               log_every_t=100,
               uc_guide_scale=7.5,
               uc_condition=None, # this has to come in the same format as the condition, # e.g. as encoded tokens, ...
               global_strength=1.0,
               ):

        # condition.keys() -- ['local_control', 'c_crossattn', 'global_control']
        # condition['local_control'][0].size() -- [1, 21, 640, 512]
        # condition['c_crossattn'][0].size() -- [1, 77, 768]
        # condition['global_control'][0].size() -- [1, 768]
        # -----------------------------------------------------------------------------------------
        # uc_condition.keys() -- ['local_control', 'c_crossattn', 'global_control']
        # uc_condition['local_control'][0].size() -- [1, 21, 640, 512]
        # uc_condition['c_crossattn'][0].size() -- [1, 77, 768]
        # uc_condition['global_control'][0].size() -- [1, 768]

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=True)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(condition, size,
                                                    # x0=x0,
                                                    log_every_t=log_every_t,
                                                    uc_guide_scale=uc_guide_scale,
                                                    uc_condition=uc_condition,
                                                    global_strength=global_strength
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, condition, shape, log_every_t=100, uc_guide_scale=7.5, uc_condition=None, global_strength=1.0):

        device = self.model.betas.device
        b = shape[0]
        noise = torch.randn(shape, device=device)

        intermediates = {'x_inter': [noise], 'pred_x0': [noise]}
        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim(noise, condition, ts, index=index, 
                                      uc_guide_scale=uc_guide_scale,
                                      uc_condition=uc_condition,
                                      global_strength=1.0)
            noise, pred_x0 = outs

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(noise)
                intermediates['pred_x0'].append(pred_x0)

        return noise, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, condition, t, index, uc_guide_scale=7.5, uc_condition=None, global_strength=1.0):
        # here x is noise
        b, *_, device = *x.shape, x.device
        model_t = self.model.apply_model(x, t, condition, global_strength) # xxxx1111
        model_uncond = self.model.apply_model(x, t, uc_condition, global_strength)
        e_t = model_uncond + uc_guide_scale * (model_t - model_uncond)

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn(x.shape, device=device) # * temperature for temperature == 1.0
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

