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
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers import SD3ControlNetModel, StableDiffusion3ControlNetPipeline
from diffusers.utils import load_image


def generate_low_frequency_noise(height, width):
    downsample = 8
    noise_height = max(1, height // downsample)
    noise_width = max(1, width // downsample)
    noise = np.random.rand(noise_height, noise_width, 3).astype(np.float32)
    noise_image = Image.fromarray((noise * 255).astype(np.uint8))
    noise_image = noise_image.resize((width, height), resample=Image.BILINEAR)
    return np.array(noise_image).astype(np.float32)


def apply_masked_noise(image, mask, mask_threshold):
    if mask.mode != "L":
        mask = mask.convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, resample=Image.NEAREST)

    image_array = np.array(image.convert("RGB")).astype(np.float32)
    mask_array = np.array(mask) > mask_threshold

    noise_array = generate_low_frequency_noise(*image_array.shape[:2])
    if mask_array.any():
        background_pixels = image_array[~mask_array]
        if background_pixels.size == 0:
            background_mean = image_array.mean(axis=(0, 1))
        else:
            background_mean = background_pixels.reshape(-1, 3).mean(axis=0)

        noise_pixels = noise_array[mask_array]
        if noise_pixels.size == 0:
            noise_mean = noise_array.reshape(-1, 3).mean(axis=0)
        else:
            noise_mean = noise_pixels.reshape(-1, 3).mean(axis=0)

        noise_array = noise_array - noise_mean + background_mean
        noise_array = np.clip(noise_array, 0, 255)
        image_array[mask_array] = noise_array[mask_array]

    return Image.fromarray(image_array.astype(np.uint8))


def build_inpainting_conditioning(input_image, mask_image, mask_threshold):
    masked_image = apply_masked_noise(input_image, mask_image, mask_threshold)
    mask = mask_image.convert("L").resize(input_image.size, resample=Image.NEAREST)
    conditioning = np.concatenate(
        [np.array(masked_image), np.array(mask)[:, :, None]], axis=2
    )
    return Image.fromarray(conditioning, mode="RGBA")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SD3 ControlNet inference on a trained model.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path or Hub ID for the base SD3 model.",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        required=True,
        help="Path to the trained ControlNet checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="crack",
        help="Prompt to condition the generation.",
    )
    parser.add_argument(
        "--conditioning_image",
        type=str,
        default=None,
        help="Path to a conditioning image (mask or ControlNet input).",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Path to the non-crack input image (required with --use_inpainting_conditioning).",
    )
    parser.add_argument(
        "--mask_image",
        type=str,
        default=None,
        help="Path to the mask image (required with --use_inpainting_conditioning).",
    )
    parser.add_argument(
        "--use_inpainting_conditioning",
        action="store_true",
        help="Build a 4-channel conditioning image from input + mask (matches training).",
    )
    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=0,
        help="Threshold (0-255) for converting masks to binary when applying masked noise.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.png",
        help="Where to save the generated image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generation. Use -1 for random.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.use_inpainting_conditioning:
        if args.input_image is None or args.mask_image is None:
            raise ValueError("--input_image and --mask_image are required with --use_inpainting_conditioning.")
    elif args.conditioning_image is None:
        raise ValueError("--conditioning_image is required unless --use_inpainting_conditioning is set.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=dtype)
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.base_model_path,
        controlnet=controlnet,
        torch_dtype=dtype,
    )
    pipe.to(device)

    if args.use_inpainting_conditioning:
        input_image = load_image(args.input_image)
        mask_image = load_image(args.mask_image)
        control_image = build_inpainting_conditioning(input_image, mask_image, args.mask_threshold)
    else:
        control_image = load_image(args.conditioning_image)

    if args.seed == -1:
        generator = None
    else:
        generator = torch.manual_seed(args.seed)

    image = pipe(
        args.prompt,
        control_image=control_image,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()
