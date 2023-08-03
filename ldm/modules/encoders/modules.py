import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
# transformers version: 4.30.2
import pdb

class FrozenCLIPEmbedder(nn.Module):
    '''uni_v15.yaml

      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder
    '''

    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77, freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        # self.tokenizer --
        # CLIPTokenizer(name_or_path='openai/clip-vit-large-patch14', 
        #     vocab_size=49408, 
        #     model_max_length=77, 
        #     is_fast=False, 
        #     padding_side='right', 
        #     truncation_side='right', 
        #     special_tokens={'bos_token': AddedToken("<|startoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        #         'eos_token': AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        #         'unk_token': AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
        #         'pad_token': '<|endoftext|>'},
        #     clean_up_tokenization_spaces=True)

        self.transformer = CLIPTextModel.from_pretrained(version)
        # (Pdb) self.transformer --
        # CLIPTextModel(
        #   (text_model): CLIPTextTransformer(
        #     (embeddings): CLIPTextEmbeddings(
        #       (token_embedding): Embedding(49408, 768)
        #       (position_embedding): Embedding(77, 768)
        #     )
        #     (encoder): CLIPEncoder(
        #       (layers): ModuleList(
        #         (0-11): 12 x CLIPEncoderLayer(
        #           (self_attn): CLIPAttention(
        #             (k_proj): Linear(in_features=768, out_features=768, bias=True)
        #             (v_proj): Linear(in_features=768, out_features=768, bias=True)
        #             (q_proj): Linear(in_features=768, out_features=768, bias=True)
        #             (out_proj): Linear(in_features=768, out_features=768, bias=True)
        #           )
        #           (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        #           (mlp): CLIPMLP(
        #             (activation_fn): QuickGELUActivation()
        #             (fc1): Linear(in_features=768, out_features=3072, bias=True)
        #             (fc2): Linear(in_features=3072, out_features=768, bias=True)
        #           )
        #           (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        #         )
        #       )
        #     )
        #     (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        #   ) # text_model
        # )

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer

        # torch.jit.script(self) ==> Errors, comes from modeling_clip.py, xxxx8888
        # torch.jit.script(self.tokenizer) ==> Errors
        # torch.jit.script(self.transformer) ==> Errors


    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        # ['a diagram, best quality, extremely detailed']
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)

        # (Pdb) batch_encoding["input_ids"].size() -- [1, 77]
        # (Pdb) batch_encoding["input_ids"]
        # tensor([[49406,   320, 22697,   267,   949,  3027,   267,  6519, 12609, 49407,
        #          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        #          49407, 49407, 49407, 49407, 49407, 49407, 49407]])

        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        z = outputs.last_hidden_state
        return z # z.size() -- [1, 77, 768]

    def encode(self, text):
        return self(text)

