from torch import nn
from ldm.modules.attention import FeedForward
import pdb

class GlobalAdapter(nn.Module):
    '''uni_v15.yaml
      target: models.global_adapter.GlobalAdapter
      params:
        in_dim: 768
        channel_mult: [2, 4]
    '''

    def __init__(self, version="v1.5", in_dim=768, channel_mult=[2, 4]):
        super().__init__()
        dim_out1, mult1 = in_dim*channel_mult[0], channel_mult[0]*2
        dim_out2, mult2 = in_dim*channel_mult[1], channel_mult[1]*2//channel_mult[0]
        self.in_dim = in_dim
        self.channel_mult = channel_mult
        self.last_channel_dim = channel_mult[-1] # support torch.jit.script
        
        self.ff1 = FeedForward(in_dim, dim_out=dim_out1, mult=mult1, glu=True, dropout=0.1)
        self.ff2 = FeedForward(dim_out1, dim_out=dim_out2, mult=mult2, glu=True, dropout=0.3)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(dim_out1)
        # torch.jit.script(self) ==> OK

    def forward(self, x):
        # x.size() -- [1, 768]
        # x.min(), x.max() -- 0., 0

        x = self.ff1(self.norm1(x))
        x = self.ff2(self.norm2(x))
        # x = rearrange(x, 'b (n d) -> b n d', n=self.channel_mult[-1], d=self.in_dim).contiguous()
        x = x.reshape(x.shape[0], self.last_channel_dim, self.in_dim).contiguous()
        
        return x