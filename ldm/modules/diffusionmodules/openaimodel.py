from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
import pdb

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, emb=None, context=None, local_features=None):
        # for layer in self:
        #     if isinstance(layer, TimestepBlock): # xxxx8888
        #         x = layer(x, emb, context)
        #     elif isinstance(layer, SpatialTransformer):
        #         x = layer(x, context, context)
        #     else:
        #         x = layer(x)
        for layer in self:
            x = layer(x, emb, context, local_features)
        return x

class NormalEmbededSequential(nn.Sequential):
    def forward(self, x, emb=None, context=None, local_features=None):
        for layer in self:
            x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x, emb=None, context=None, local_features=None): # xxxx8888
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        stride = 2 if dims != 3 else (1, 2, 2)
        self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)

    def forward(self, x):
        return self.op(x)
    

class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)


    def forward(self, x, emb, context=None, local_features=None): # xxxx8888
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class LocalControlUNetModel(nn.Module):
    '''uni_v15.yaml

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
    def __init__(
        self,
        version="v1.5",
        in_channels=4,
        model_channels=320,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=[4, 2, 1],
        dropout=0,
        channel_mult=(1, 2, 4, 4),
        dims=2,
        num_heads=8,
        transformer_depth=1,              # custom transformer support
        context_dim=768,                 # custom transformer support
    ):
        super().__init__()
        self.model_channels = model_channels
        self.num_res_blocks = len(channel_mult) * [num_res_blocks] # [2, 2, 2, 2]
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                NormalEmbededSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(NormalEmbededSequential(Downsample(ch, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
            # ==> layers -- [ResBlock, SpatialTransformer, ...]
        dim_head = ch // num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims),
            SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim),
            ResBlock(ch, time_embed_dim, dropout, dims=dims),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]: # channel_mult -- (1, 2, 4, 4)
            for i in range(self.num_res_blocks[level] + 1): # self.num_res_blocks -- [2, 2, 2, 2] ==> i in range(0, 3)
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult, dims=dims)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions: # [4, 2, 1]
                    dim_head = ch // num_heads
                    layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim))
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(Upsample(ch, dims=dims, out_channels=out_ch))
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, model_channels, out_channels, 3, padding=1),
        )


    def forward(self, x, timesteps=None, context=None, local_control=None):
        # x.size() -- [1, 4, 80, 64], sample noise ?
        # context.size() -- [1, 81, 768], global_control, comes from CLIP("ViT-L-14")
        # len(local_control) -- 13
        # local_control[0].size() -- [1, 320, 80, 64], local_control[12].size() -- [1, 1280, 10, 8]

        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels) # self.model_channels -- 320
            emb = self.time_embed(t_emb)
            h = x
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        h += local_control.pop()

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop() + local_control.pop()], dim=1)
            h = module(h, emb, context)

        return self.out(h)