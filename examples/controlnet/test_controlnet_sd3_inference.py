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

import torch

from diffusers import SD3ControlNetModel, StableDiffusion3ControlNetInpaintingPipeline
from diffusers.utils import load_image


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
        "--input_image",
        type=str,
        required=True,
        help="Path to the non-crack input image.",
    )
    parser.add_argument(
        "--mask_image",
        type=str,
        required=True,
        help="Path to the mask image.",
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=dtype)
    pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
        args.base_model_path,
        controlnet=controlnet,
        torch_dtype=dtype,
    )
    pipe.to(device)

    input_image = load_image(args.input_image)
    mask_image = load_image(args.mask_image)

    if args.seed == -1:
        generator = None
    else:
        generator = torch.manual_seed(args.seed)

    image = pipe(
        args.prompt,
        control_image=input_image,
        control_mask=mask_image,
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
