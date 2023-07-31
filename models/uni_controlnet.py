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
    def __init__(self, version="v1.5"):
        super().__init__(version=version)
        # pp mode -- 'uni'
        # local_control_config -- {'target': 'models.local_adapter.LocalAdapter' ... }
        # global_control_config -- {'target': 'models.global_adapter.GlobalAdapter' ... }

        self.local_adapter = LocalAdapter(version=version) # instantiate_from_config(local_control_config) # models.local_adapter.LocalAdapter
        self.local_control_scales = [1.0] * 13
        self.global_adapter = GlobalAdapter(version=version) # instantiate_from_config(global_control_config) # models.global_adapter.GlobalAdapter


    def apply_model(self, x_noisy, t, cond, global_strength=1.0):
        assert isinstance(cond, dict)
        # x_noisy.size() -- [1, 4, 80, 64]
        # t -- tensor([801], device='cuda:0')
        # cond.keys() -- ['local_control', 'c_crossattn', 'global_control']
        # (Pdb) pp args -- ()
        # (Pdb) pp kwargs -- {}

        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1) # size() -- [1, 77, 768]

        # cond['local_control'][0].size() -- [1, 21, 640, 512]
        # cond['global_control'][0].size() -- [1, 768]

        assert cond['global_control'][0] != None
        global_control = self.global_adapter(cond['global_control'][0]) # global_control.size() -- [1, 4, 768]
        cond_txt = torch.cat([cond_txt, global_strength*global_control], dim=1)

        assert cond['local_control'][0] != None
        local_control = torch.cat(cond['local_control'], 1)
        # torch.cat(cond['local_control'], 1).size() -- [1, 21, 640, 512]

        local_control = self.local_adapter(x=x_noisy, timesteps=t, context=cond_txt, local_conditions=local_control)
        local_control = [c * scale for c, scale in zip(local_control, self.local_control_scales)]
        
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, local_control=local_control)
        # eps.size() -- [1, 4, 80, 64]
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):

        return self.get_learned_conditioning([""] * N)


    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.local_adapter = self.local_adapter.cuda()
            self.global_adapter = self.global_adapter.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.local_adapter = self.local_adapter.cpu()
            self.global_adapter = self.global_adapter.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
