import torch.nn as nn
# from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.diffusionmodules.model import Decoder
import pdb

class AutoencoderKL(nn.Module):
    # """uni_v15.yaml

    #   target: ldm.models.autoencoder.AutoencoderKL
    #   params:
    #     embed_dim: 4
    #     monitor: val/rec_loss
    #     ddconfig:
    #       double_z: true
    #       z_channels: 4
    #       resolution: 256
    #       in_channels: 3
    #       out_ch: 3
    #       ch: 128
    #       ch_mult:
    #       - 1
    #       - 2
    #       - 4
    #       - 4
    #       num_res_blocks: 2
    #       attn_resolutions: []
    #       dropout: 0.0
    #     lossconfig:
    #       target: torch.nn.Identity
    # """
    def __init__(self,
                 version="v1.5",
                 embed_dim=4,
                 ):
        super().__init__()
        ddconfig = {
            'z_channels': 4, 
            'resolution': 256, 
            'in_channels': 3, 
            'out_ch': 3, 
            'ch': 128, 
            'ch_mult': [1, 2, 4, 4], 
            'num_res_blocks': 2, 
            'dropout': 0.0
        }

        # self.encoder = Encoder(**ddconfig) # model size 130 M
        self.decoder = Decoder(**ddconfig) # model size 190 M

        self.quant_conv = nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        # self.quant_conv -- Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # self.post_quant_conv -- Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1))


    def forward(self, x):
        return self.decode(x)

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

