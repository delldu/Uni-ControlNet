import cv2
from PIL import Image

import torch
from transformers import AutoProcessor, CLIPModel

from annotator.util import annotator_ckpts_path
import pdb

class ContentDetector:
    def __init__(self):

        model_name = "openai/clip-vit-large-patch14"
        # annotator_ckpts_path -- 'annotator/ckpts'

        self.model = CLIPModel.from_pretrained(model_name, cache_dir=annotator_ckpts_path).cuda().eval()
        # self.model
        # CLIPModel(
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
        #   )
        #   (vision_model): CLIPVisionTransformer(
        #     (embeddings): CLIPVisionEmbeddings(
        #       (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
        #       (position_embedding): Embedding(257, 1024)
        #     )
        #     (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        #     (encoder): CLIPEncoder(
        #       (layers): ModuleList(
        #         (0-23): 24 x CLIPEncoderLayer(
        #           (self_attn): CLIPAttention(
        #             (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
        #             (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
        #             (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
        #             (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
        #           )
        #           (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        #           (mlp): CLIPMLP(
        #             (activation_fn): QuickGELUActivation()
        #             (fc1): Linear(in_features=1024, out_features=4096, bias=True)
        #             (fc2): Linear(in_features=4096, out_features=1024, bias=True)
        #           )
        #           (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        #         )
        #       )
        #     )
        #     (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        #   )
        #   (visual_projection): Linear(in_features=1024, out_features=768, bias=False)
        #   (text_projection): Linear(in_features=768, out_features=768, bias=False)
        # )

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=annotator_ckpts_path)
        # (Pdb) self.processor
        # CLIPProcessor:
        # - image_processor: CLIPImageProcessor {
        #   "crop_size": {
        #     "height": 224,
        #     "width": 224
        #   },
        #   "do_center_crop": true,
        #   "do_convert_rgb": true,
        #   "do_normalize": true,
        #   "do_rescale": true,
        #   "do_resize": true,
        #   "feature_extractor_type": "CLIPFeatureExtractor",
        #   "image_mean": [
        #     0.48145466,
        #     0.4578275,
        #     0.40821073
        #   ],
        #   "image_processor_type": "CLIPImageProcessor",
        #   "image_std": [
        #     0.26862954,
        #     0.26130258,
        #     0.27577711
        #   ],
        #   "resample": 3,
        #   "rescale_factor": 0.00392156862745098,
        #   "size": {
        #     "shortest_edge": 224
        #   }
        # }

        # - tokenizer: CLIPTokenizerFast(name_or_path='openai/clip-vit-large-patch14', vocab_size=49408, model_max_length=77, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': AddedToken("<|startoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'eos_token': AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'unk_token': AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'pad_token': '<|endoftext|>'}, clean_up_tokenization_spaces=True)


    def __call__(self, img):
        assert img.ndim == 3
        with torch.no_grad():
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=img, return_tensors="pt").to('cuda')
            # inputs.keys() -- dict_keys(['pixel_values'])
            # inputs['pixel_values'].size() -- [1, 3, 224, 224], (min, max) -- (-1.7339, 2.0606)
            image_features = self.model.get_image_features(**inputs) # image_features[0].size() -- [768]

            image_feature = image_features[0].detach().cpu().numpy()

        # img.getpixel((0,0)) -- (255, 255, 255)
        # inputs['pixel_values'].size(), inputs['pixel_values'].min(), inputs['pixel_values'].max()
        # ([1, 3, 224, 224], -1.7923, 2.1459, device='cuda:0')
        # image_feature.shape -- (768,) min(), max() -- (-7.8111997, 8.373651)

        return image_feature
