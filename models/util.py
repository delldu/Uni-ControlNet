import os
import torch
from  models.uni_controlnet import UniControlNet
import pdb

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)

    remove_keys = [
        "alphas_cumprod_prev",
        "sqrt_alphas_cumprod",
        "sqrt_one_minus_alphas_cumprod",
        "sqrt_recip_alphas_cumprod",
        "sqrt_recipm1_alphas_cumprod",
        "log_one_minus_alphas_cumprod",
        "posterior_variance",
        "posterior_log_variance_clipped",
        "posterior_mean_coef1",
        "posterior_mean_coef2",
        "logvar"
    ]
    for k in state_dict.keys():
        if k.startswith("first_stage_model.encoder."):
            remove_keys.append(k)

    for key in remove_keys:
        del state_dict[key]

    # model.diffusion_model. --> diffusion_model.
    new_state_dict = {}
    for n, p in state_dict.items():
        if n.startswith("model.diffusion_model."):
            n = n.replace("model.diffusion_model.", "diffusion_model.")
        new_state_dict[n] = p


    print(f'Loaded state_dict from [{ckpt_path}]')
    return new_state_dict


def create_model(version):
    print(f'Create model {version} ...')
    model = UniControlNet(version=version).cpu()
    return model