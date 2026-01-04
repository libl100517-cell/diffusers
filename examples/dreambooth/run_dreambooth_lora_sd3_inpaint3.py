#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusion3InpaintPipeline


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Run SD3 inpaint LoRA trained with train_dreambooth_lora_sd3_inpaint3.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--lora_weights", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--mask", type=str, required=True, help="Path to the inpaint mask (white=replace).")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def apply_zero_mask(image, mask):
    image_array = np.array(image.convert("RGB")).astype(np.float32)
    mask_array = np.array(mask.convert("L")) > 0
    if mask_array.any():
        image_array[mask_array] = 0.0
    return Image.fromarray(image_array.astype(np.uint8))


def main(args):
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    pipe = StableDiffusion3InpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=torch_dtype
    )
    pipe.load_lora_weights(args.lora_weights)
    pipe.to(device)

    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    masked_image = apply_zero_mask(image, mask)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=masked_image,
        mask_image=mask,
        height=args.height,
        width=args.width,
        strength=args.strength,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    output_path = Path(args.output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    result.images[0].save(output_path)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
