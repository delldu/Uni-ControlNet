import torch
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import ResBlock, Downsample
from typing import Optional

import pdb


class LocalTimestepEmbedSequential(nn.Sequential):
    '''
    1) ResBlock
    2) LocalResBlock
    3) LocalResBlock, SpatialTransformer
    4) ResBlock, SpatialTransformer
    5) ResBlock, SpatialTransformer, ResBlock
    '''

    def forward(self, x, emb, context, local_features):
        # for layer in self:
        #     if isinstance(layer, ResBlock):
        #         x = layer(x, emb, context)
        #     elif isinstance(layer, LocalResBlock):
        #         x = layer(x, emb, local_features)
        #     elif isinstance(layer, SpatialTransformer):
        #         x = layer(x, context, context)
        #     else:
        #         x = layer(x)
        for layer in self:
            x = layer(x, emb, context, local_features)
        return x

class LocalNormalEmbededSequential(nn.Sequential):
    def forward(self, x, emb: Optional[torch.Tensor]=None, context: Optional[torch.Tensor]=None, local_features: Optional[torch.Tensor]=None):
        for layer in self:
            x = layer(x)
        return x


class FDN(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        ks = 3
        pw = ks // 2
        self.param_free_norm = nn.GroupNorm(32, norm_nc, affine=False)
        self.conv_gamma = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, local_features):
        normalized = self.param_free_norm(x)
        assert local_features.size()[2:] == x.size()[2:]
        gamma = self.conv_gamma(local_features)
        beta = self.conv_beta(local_features)
        out = normalized * (1 + gamma) + beta
        return out


class LocalResBlock(nn.Module):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, dims=2, inject_channels=None):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.norm_in = FDN(channels, inject_channels)
        self.norm_out = FDN(self.out_channels, inject_channels)

        self.in_layers = nn.Sequential(
            nn.Identity(),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, self.out_channels, ),
        )
        self.out_layers = nn.Sequential(
            nn.Identity(),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, context=None, local_conditions=None): # for LocalTimestepEmbedSequential
        h = self.norm_in(x, local_conditions)
        h = self.in_layers(h)
        
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        h = h + emb_out
        h = self.norm_out(h, local_conditions)
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h


class FeatureExtractor(nn.Module):
    def __init__(self, local_channels, inject_channels, dims=2):
        super().__init__()
        self.pre_extractor = LocalNormalEmbededSequential(
            conv_nd(dims, local_channels, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 64, 64, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 64, 128, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 128, 128, 3, padding=1),
            nn.SiLU(),
        )
        self.extractors = nn.ModuleList([
            LocalNormalEmbededSequential(
                conv_nd(dims, 128, inject_channels[0], 3, padding=1, stride=2),
                nn.SiLU()
            ),
            LocalNormalEmbededSequential(
                conv_nd(dims, inject_channels[0], inject_channels[1], 3, padding=1, stride=2),
                nn.SiLU()
            ),
            LocalNormalEmbededSequential(
                conv_nd(dims, inject_channels[1], inject_channels[2], 3, padding=1, stride=2),
                nn.SiLU()
            ),
            LocalNormalEmbededSequential(
                conv_nd(dims, inject_channels[2], inject_channels[3], 3, padding=1, stride=2),
                nn.SiLU()
            )
        ])
        self.zero_convs = nn.ModuleList([
            conv_nd(dims, inject_channels[0], inject_channels[0], 3, padding=1),
            conv_nd(dims, inject_channels[1], inject_channels[1], 3, padding=1),
            conv_nd(dims, inject_channels[2], inject_channels[2], 3, padding=1),
            conv_nd(dims, inject_channels[3], inject_channels[3], 3, padding=1)
        ])
        # torch.jit.script(self.pre_extractor) ==> OK
        # torch.jit.script(self.extractors) ==> OK
        # torch.jit.script(self.zero_convs) ==> OK
        # torch.jit.script(self) ==> OK

    def forward(self, local_conditions):
        # local_conditions.size() -- [1, 21, 640, 512]

        local_features = self.pre_extractor(local_conditions)
        assert len(self.extractors) == len(self.zero_convs)
        
        output_features = []
        # for idx in range(len(self.extractors)):
        #     local_features = self.extractors[idx](local_features)
        #     output_features.append(self.zero_convs[idx](local_features))
        for idx, (e, z) in enumerate(zip(self.extractors, self.zero_convs)):
            local_features = e(local_features)
            output_features.append(z(local_features))

        return output_features


class LocalAdapter(nn.Module):
    '''uni_v15.yaml
    
      target: models.local_adapter.LocalAdapter
      params:
        in_channels: 4
        model_channels: 320
        local_channels: 21
        inject_channels: [192, 256, 384, 512]
        inject_layers: [1, 4, 7, 10]
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
    def __init__(self,
            version="v1.5",
            in_channels=4,
            model_channels=320,
            local_channels=21,
            inject_channels=[192, 256, 384, 512],
            inject_layers=[1, 4, 7, 10],
            num_res_blocks=2,
            attention_resolutions=[4, 2, 1],
            dropout=0.0,
            channel_mult=(1, 2, 4, 4),
            dims=2,
            num_heads=8,
            transformer_depth=1,  # custom transformer support
            context_dim=768,  # custom transformer support
    ):
        super().__init__()
        self.model_channels = model_channels
        self.inject_layers = inject_layers

        num_res_blocks = len(channel_mult) * [num_res_blocks]
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.feature_extractor = FeatureExtractor(local_channels, inject_channels)
        self.input_blocks = nn.ModuleList(
            [LocalNormalEmbededSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self.zero_convs = nn.ModuleList(
            [LocalNormalEmbededSequential(conv_nd(dims, model_channels, model_channels, 1, padding=0))]
        )

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(num_res_blocks[level]):
                if (1 + 3*level + nr) in self.inject_layers:
                    layers = [
                        LocalResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims,
                            inject_channels=inject_channels[level]
                        )
                    ]
                else:
                    layers = [
                        ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims)
                    ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                    )
                self.input_blocks.append(LocalTimestepEmbedSequential(*layers))

                self.zero_convs.append(LocalNormalEmbededSequential(conv_nd(dims, ch, ch, 1, padding=0)))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    LocalNormalEmbededSequential(Downsample(ch, dims=dims, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(LocalNormalEmbededSequential(conv_nd(dims, ch, ch, 1, padding=0)))
                ds *= 2

        dim_head = ch // num_heads
        self.middle_block = LocalTimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, ),
            SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, ),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, ),
        )
        self.middle_block_out = LocalNormalEmbededSequential(conv_nd(dims, ch, ch, 1, padding=0))
        # torch.jit.script(self) ==> OK
        # torch.jit.script(self.feature_extractor) ==> OK
        # torch.jit.script(self.input_blocks) ==> OK
        # torch.jit.script(self.zero_convs) ==> OK
        # torch.jit.script(self.middle_block_out) ==> OK

        # torch.jit.script(self.middle_block) ==> OK

    def forward(self, x, timesteps, context, local_conditions):
        # x.size() -- [1, 4, 80, 64
        # timesteps -- tensor([801], device='cuda:0')
        # context.size() -- [1, 81, 768]
        # local_conditions.size() -- [1, 21, 640, 512] ??? 21

        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)
        local_features = self.feature_extractor(local_conditions)

        outs = []
        h = x

        for layer_idx, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            if layer_idx in self.inject_layers:
                h = module(h, emb, context, local_features[self.inject_layers.index(layer_idx)])
            else:
                h = module(h, emb, context, context) # !!!useless!!!, just for torch.jit.script
            outs.append(zero_conv(h, emb, context, context)) # same as !!!useless!!!

        h = self.middle_block(h, emb, context, context) # same as !!!useless!!!
        outs.append(self.middle_block_out(h, emb, context, context)) # same as !!!useless!!!

        return outs


