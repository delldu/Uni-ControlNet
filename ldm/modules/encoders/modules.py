# coding=utf-8
import torch
from torch import nn
import pdb


class FrozenCLIPEmbedder(nn.Module):
    '''uni_v15.yaml

      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder
    '''
    def __init__(self):
        super().__init__()
        self.transformer = CLIPTextModel()
        self.freeze()
        # torch.jit.script(self) ==> OK

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, tokens):
        # tensor([[49406,   320, 22697,   267,   949,  3027,   267,  6519, 12609, 49407,
        #          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #          49407, 49407, 49407, 49407, 49407, 49407, 49407]])

        z = self.transformer(input_tokens=tokens)
        return z # z.size() -- [1, 77, 768]

    def encode(self, tokens):
        return self(tokens)


class DictToClass(object):
    def __init__(self, _obj):
        if _obj:
            self.__dict__.update(_obj)


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    def forward(self, input):
        return input * torch.sigmoid(1.702 * input)


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        embed_dim = config.hidden_size # 768
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim) # (49408, 768)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim) # (77, 768)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # (self.position_ids -- size(): [1, 77]
        # tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        #          18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        #          36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        #          54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        #          72, 73, 74, 75, 76]], device='cuda:0')

    def forward(self, input_tokens):
        seq_length = input_tokens.shape[-1]

        position_ids = self.position_ids[:, :seq_length]
        # (Pdb) position_ids
        # tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        #          18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        #          36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        #          54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        #          72, 73, 74, 75, 76]], device='cuda:0')

        inputs_embeds = self.token_embedding(input_tokens)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings # size() -- [1, 77, 768]

class CLIPAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor, seq_len: int, B: int):
        return tensor.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, causal_attention_mask):
        """Input shape: Batch x Time x Channel"""
        B, L, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, B)
        value_states = self._shape(self.v_proj(hidden_states), -1, B)

        proj_shape = (B * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, L, B).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # apply the causal_attention_mask first
        attn_weights = attn_weights.view(B, self.num_heads, L, src_len) + causal_attention_mask
        attn_weights = attn_weights.view(B * self.num_heads, L, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(B, self.num_heads, L, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, L, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = QuickGELUActivation()

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return hidden_states

class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states, causal_attention_mask):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            causal_attention_mask=causal_attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds, causal_attention_mask):
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            layer_output = encoder_layer(hidden_states, causal_attention_mask)
            hidden_states = layer_output

        return hidden_states # size() -- [1, 77, 768], last_hidden_state


def make_causal_mask(x):
    """
    Make causal mask used for bi-directional self-attention.
    """
    B, L = x.size() # torch.Size([1, 77])
    mask = torch.full((L, L), -1000000000.0, device=x.device)
    mask_cond = torch.arange(mask.size(-1), device=x.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(x.dtype) # mask.size() -- [77, 77]

    return mask[None, None, :, :].expand(B, 1, L, L) # size() -- [1, 1, 77, 77]

class CLIPTextTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_tokens):
        input_shape = input_tokens.size() # [1, 77]
        input_tokens = input_tokens.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_tokens)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = make_causal_mask(input_tokens)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states, causal_attention_mask=causal_attention_mask)
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        return last_hidden_state # size() -- [1, 77, 768]

class CLIPTextModel(nn.Module): 
    def __init__(self):
        super().__init__()
        config = DictToClass({
          "_name_or_path": "openai/clip-vit-large-patch14",
          "transformers_version": "4.30.2",
          "attention_dropout": 0.0,
          "dropout": 0.0,
          "hidden_size": 768,
          "intermediate_size": 3072,
          "layer_norm_eps": 1e-05,
          "max_position_embeddings": 77,
          "num_attention_heads": 12,
          "num_hidden_layers": 12,
          "vocab_size": 49408
        })
        self.text_model = CLIPTextTransformer(config)

    def forward(self, input_tokens):
        z = self.text_model(input_tokens=input_tokens)
        return z # size() -- ([1, 77]
