import einops
import torch

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config

# xxxx1111
from models.local_adapter import LocalAdapter
from models.global_adapter import GlobalAdapter

import pdb

class UniControlNet(LatentDiffusion):
    '''uni_v15.yaml

      target: models.uni_controlnet.UniControlNet
      params:
        linear_start: 0.00085
        linear_end: 0.0120
        num_timesteps_cond: 1
        log_every_t: 200
        timesteps: 1000
        first_stage_key: "jpg"
        cond_stage_key: "txt"
        image_size: 64
        channels: 4
        cond_stage_trainable: false
        conditioning_key: crossattn
        monitor: val/loss_simple_ema
        scale_factor: 0.18215
        use_ema: False
        mode: uni
    '''
    def __init__(self, mode, local_control_config=None, global_control_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # pp mode -- 'uni'
        # local_control_config -- {'target': 'models.local_adapter.LocalAdapter' ... }
        # global_control_config -- {'target': 'models.global_adapter.GlobalAdapter' ... }

        assert mode in ['local', 'global', 'uni']
        self.mode = mode
        if self.mode in ['local', 'uni']:
            self.local_adapter = instantiate_from_config(local_control_config) # models.local_adapter.LocalAdapter
            self.local_control_scales = [1.0] * 13
        if self.mode in ['global', 'uni']:
            self.global_adapter = instantiate_from_config(global_control_config) # models.global_adapter.GlobalAdapter


    def apply_model(self, x_noisy, t, cond, global_strength=1, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if self.mode in ['global', 'uni']:
            assert cond['global_control'][0] != None
            global_control = self.global_adapter(cond['global_control'][0])
            cond_txt = torch.cat([cond_txt, global_strength*global_control], dim=1)
        if self.mode in ['local', 'uni']:
            assert cond['local_control'][0] != None
            local_control = torch.cat(cond['local_control'], 1)
            local_control = self.local_adapter(x=x_noisy, timesteps=t, context=cond_txt, local_conditions=local_control)
            local_control = [c * scale for c, scale in zip(local_control, self.local_control_scales)]
        
        if self.mode == 'global':
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt)
        else:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, local_control=local_control)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):

        return self.get_learned_conditioning([""] * N)


    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            if self.mode in ['local', 'uni']:
                self.local_adapter = self.local_adapter.cuda()
            if self.mode in ['global', 'uni']:
                self.global_adapter = self.global_adapter.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            if self.mode in ['local', 'uni']:
                self.local_adapter = self.local_adapter.cpu()
            if self.mode in ['global', 'uni']:
                self.global_adapter = self.global_adapter.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
