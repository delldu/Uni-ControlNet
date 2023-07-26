import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class FrozenCLIPEmbedder(nn.Module):
    '''uni_v15.yaml

      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder
    '''

    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx


    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

