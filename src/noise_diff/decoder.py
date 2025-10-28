import math
import torch
import torch.nn as nn
from torch.nn.modules import Linear, Sequential

# This model is a simpler version of the model proposed in tqch/ddpm-torch/blob/master/ddpm_torch/toy/toy_model.py

DEFAULT_DTYPE = torch.float32

def get_timestep_embedding(timesteps, embed_dim: int, dtype: torch.dtype = DEFAULT_DTYPE):
    """
    Adapted from fairseq/fairseq/modules/sinusoidal_positional_embedding.py
    The implementation is slightly different from the decription in Section 3.5 of [1]
    [1] Vaswani, Ashish, et al. "Attention is all you need."
     Advances in neural information processing systems 30 (2017).
    """
    half_dim = embed_dim // 2
    embed = math.log(10000) / (half_dim - 1)
    embed = torch.exp(-torch.arange(half_dim, dtype=dtype, device=timesteps.device) * embed)
    embed = torch.outer(timesteps.ravel().to(dtype), embed)
    embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=1)
    return embed

DEFAULT_NONLINEARITY = nn.ReLU(inplace=True)

class TemporalLayer(nn.Module):
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(self, in_features, out_features, temporal_features):
        super(TemporalLayer, self).__init__()
        self.fc1 = Linear(in_features, out_features, bias=False)
        self.enc = Linear(temporal_features, out_features, bias=False)

    def forward(self, x, t_emb):
        out = self.nonlinearity(self.fc1(x) + self.enc(t_emb))
        return x + out # residual network

class Sequential(nn.Sequential):
    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input

class Decoder(nn.Module):
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(self, in_features, mid_features, num_temporal_layers):
        super(Decoder, self).__init__()

        self.in_fc = Linear(in_features, mid_features, bias=False)
        self.temp_fc = Sequential(*(
            [TemporalLayer(mid_features, mid_features, mid_features) 
            for _ in range(num_temporal_layers) ]))
        self.out_fc = Linear(mid_features, in_features)
        self.t_proj = nn.Sequential(
            Linear(mid_features, mid_features, bias=False),
            self.nonlinearity) 
        self.mid_features = mid_features

    def forward(self, x, t):
        t_emb = get_timestep_embedding(t, self.mid_features)
        t_emb = self.t_proj(t_emb)
        out = self.in_fc(x)
        out = self.temp_fc(out, t_emb=t_emb)
        out = self.out_fc(out)
        return out
