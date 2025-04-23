# --------------------------------------------------------
# Copyright (2025) Bytedance Ltd. and/or its affiliates 
# Licensed under the Apache License, Version 2.0 (the "License")
# SAIL Project
# Written by Weixian Lei & Jiacong Wang
# --------------------------------------------------------

import io
import copy
import math
import base64
import torch
from math import ceil
import torchvision.transforms as transforms
from PIL import Image
from einops import rearrange

from modeling_sail import get_transformer_and_tokenizer
from transformers import DynamicCache, GenerationConfig


NON_VISION_TOKEN_ID = -1


def load_image_to_base64(image_path: str) -> str:
    # convert image to jpeg, then to data:image/jpeg;base64,
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"


def load_base64_to_PILImage(base64_string: str) -> Image:
    # convert data:image/jpeg;base64, to jpeg
    base64_string = base64_string.split(",")[1]
    decoded_string = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(decoded_string)).convert('RGB')


def get_resize_output_image_size(
    image_size,
    max_token_num=4096,
    patch_size=32
) -> tuple:
    l1, l2 = image_size # 540, 32
    short, long = (l2, l1) if l2 <= l1 else (l1, l2)

    requested_new_long = ceil(long / patch_size) * patch_size
    new_long, new_short = requested_new_long, int(requested_new_long * short / long)
    # Find the nearest multiple of 64 for new_short
    new_short = ceil(new_short / patch_size) * patch_size
    token_num = new_long * new_short / (patch_size*patch_size)
    if token_num > max_token_num:
        scale_factor =  math.sqrt(token_num / max_token_num)
        new_long = int(new_long / scale_factor / patch_size) * patch_size
        new_short = int(new_short / scale_factor/ patch_size) * patch_size

    return (new_long, new_short) if l2 <= l1 else (new_short, new_long)


def preprocess_image(
    image_tensor: torch.Tensor,
    patch_size: int = 32
) -> torch.Tensor:
    patches = image_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous() # -> (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
    return patches


def get_transform(height, width):
    preprocess_transform = transforms.Compose([
            transforms.Resize((height, width)),             
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return preprocess_transform


def convert_image_base64_to_patches(base64_image: str, patch_size: int, fix_res_size: int = None) -> torch.Tensor:
    img_pil = load_base64_to_PILImage(base64_image)
    # resize the image to the nearest multiple of patch_size
    width, height = img_pil.size
    new_width, new_height = get_resize_output_image_size((width, height), patch_size=patch_size)
    img_tensor = get_transform(new_height, new_width)(img_pil) # 3,height, width 
    img_patches = preprocess_image(img_tensor, patch_size=patch_size) # seq_length, 64*64*3
    return img_patches


def prepare_image_textual_seq_norowsep(h, w, tokenizer, add_cls=True):
    seq = ""
    tok_len = 0
    
    seq += tokenizer.vis_beg_tok
    tok_len += 1
    
    seq += tokenizer.vis_patch_tok * (w * h)
    tok_len += (w * h)
    
    seq += tokenizer.vis_end_tok
    tok_len += 1
    
    if add_cls:
        seq += tokenizer.vis_cls_tok
        tok_len += 1
    
    return seq, tok_len


def create_single_prefix_mask(prefix_len, max_len):
    attn_mask = torch.zeros(max_len, max_len)
    attn_mask[:prefix_len, :prefix_len] = 1
    causal_mask = torch.tril(torch.ones(max_len, max_len))
    attn_mask = attn_mask.bool() | causal_mask.bool()
    return attn_mask


def generate_mm_pos_ids_singleit(input_ids, vpatch_id, h, w):
    input_ids_pt = torch.Tensor(input_ids).int()
    vpatch_pos = torch.argwhere(input_ids_pt == vpatch_id)
    vpatch_start_pos = vpatch_pos[0].item()
    nt = len(input_ids) - (h*w) + 1
 
    t_indices = torch.arange(1)
    h_indices = torch.arange(h)
    w_indices = torch.arange(w)
    v_pos_id = torch.stack(torch.meshgrid(t_indices, h_indices, w_indices, indexing='ij'), dim=0)
    v_pos_id = rearrange(v_pos_id, "d t h w -> (t h w) d")
    v_pos_id += vpatch_start_pos
    position_id = torch.cat(
        [
            torch.arange(vpatch_start_pos).unsqueeze(-1).repeat(1,3),
            v_pos_id,
            torch.arange(nt-vpatch_start_pos-1).unsqueeze(-1).repeat(1,3) + v_pos_id.max() + 1,
        ],
        dim=0
    )
    assert len(input_ids) == position_id.size(0)
    position_id = rearrange(position_id, "slen d -> d slen").long()
    
    return position_id

PATH_TO_MODEL = "path to model"
PATH_TO_TOKENIZER = "path to tokenizer"
IMAGE_PATH = "path to image"
PROMPT = "content of prompt"

if __name__ == "__main__":
    model, tokenizer = get_transformer_and_tokenizer(
        PATH_TO_MODEL,
        PATH_TO_TOKENIZER
    )
    print("== attention impl: {}".format(model.config._attn_implementation))
    model = model.cuda()

    image_processor = lambda x: convert_image_base64_to_patches(load_image_to_base64(x), model.config.vision_patch_size, fix_res_size=None)
    prompt_inp = tokenizer.bos_token + '[INST] {} [/INST]'.format(PROMPT)
    image_path = IMAGE_PATH   

    image_patches = image_processor(image_path)
    nh, nw = image_patches.shape[:2]
    image_tokens, image_tokens_len = prepare_image_textual_seq_norowsep(nh, nw, tokenizer, add_cls=False)
    
    input_tokens = image_tokens + prompt_inp
    input_ids = tokenizer(input_tokens, add_special_tokens=False, return_tensors="pt").input_ids
    vision_patch_indices = torch.full_like(input_ids, fill_value=NON_VISION_TOKEN_ID)
    vision_patches = image_patches.view(nh*nw, -1)
    assert (input_ids == tokenizer.vis_patch_tok_id).sum() == vision_patches.size(0)
    assert (input_ids >= tokenizer.vis_beg_tok_id).sum() == image_tokens_len

    vision_patch_indices[input_ids==tokenizer.vis_patch_tok_id] = torch.arange(vision_patches.size(0))
    attention_mask = create_single_prefix_mask(image_tokens_len, input_ids.size(-1)).unsqueeze(0).unsqueeze(0)
    position_ids = generate_mm_pos_ids_singleit(input_ids.squeeze(0).numpy().tolist(), tokenizer.vis_patch_tok_id, nh, nw).unsqueeze(1)
    
    input_ids = input_ids.long().cuda()
    vision_patch_indices = vision_patch_indices.long().cuda()
    vision_patches = vision_patches.to(torch.bfloat16).cuda()
    position_ids = position_ids.long().cuda()
    attention_mask = attention_mask.cuda()

    padding_attention_mask = torch.ones_like(input_ids).cuda()

    inputs = dict(
        input_ids = input_ids,
        position_ids = position_ids,
        attention_mask = padding_attention_mask,
        vision_patches = vision_patches,
        vision_patch_indices = vision_patch_indices,
        use_cache=True
    )

    cached_inputs = dict(
        input_ids = input_ids[:, :image_tokens_len],
        position_ids = position_ids[:, :, :image_tokens_len],
        attention_mask = attention_mask[:,:, :image_tokens_len, :image_tokens_len],
        vision_patches = vision_patches,
        vision_patch_indices = vision_patch_indices[:, :image_tokens_len],
        use_cache=True
    )

    prefix_cache = DynamicCache()
    with torch.no_grad():
        prefix_cache = model.forward(**cached_inputs, past_key_values=prefix_cache).past_key_values

    past_key_values = copy.deepcopy(prefix_cache)
    generate_config = GenerationConfig(
        max_new_tokens=1024,
        return_dict_in_generate=True,
        output_attentions=False
    )
    generated = model.generate(
        **inputs,
        past_key_values=past_key_values,
        generation_config=generate_config
    )
    generated_ids = generated['sequences'][:, input_ids.size(1):]
    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"\nModel Response: ===\n{response}\n===")
