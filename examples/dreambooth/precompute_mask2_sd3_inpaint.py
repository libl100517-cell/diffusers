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
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Precompute mask2 files for SD3 inpaint LoRA training.")
    parser.add_argument("--train_data_list", type=str, required=True, help="Path to a text file listing images.")
    parser.add_argument("--train_data_root", type=str, required=True, help="Root directory for the listed images.")
    parser.add_argument("--image_folder_name", type=str, default="images")
    parser.add_argument("--mask_folder_name", type=str, default="masks")
    parser.add_argument("--mask_extension", type=str, default=".png")
    parser.add_argument("--mask2_folder_name", type=str, default="masks2")
    parser.add_argument("--mask2_extension", type=str, default=".png")
    parser.add_argument("--mask2_count", type=int, default=4)
    parser.add_argument("--mask_threshold", type=int, default=0)
    parser.add_argument("--mask_dilate_radius", type=int, default=5)
    parser.add_argument("--mask2_rotate", type=float, default=15.0)
    parser.add_argument("--mask2_translate", type=float, default=0.15)
    parser.add_argument("--mask2_scale_min", type=float, default=0.8)
    parser.add_argument("--mask2_scale_max", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=None)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.mask_extension and not args.mask_extension.startswith("."):
        args.mask_extension = f".{args.mask_extension}"
    if args.mask2_extension and not args.mask2_extension.startswith("."):
        args.mask2_extension = f".{args.mask2_extension}"

    return args


def replace_path_part(path_value, source_name, target_name):
    normalized_value = path_value.replace("\\", "/")
    parts = list(Path(normalized_value).parts)
    if source_name in parts:
        source_index = parts.index(source_name)
        parts[source_index] = target_name
        return str(Path(*parts))
    raise ValueError(f"Expected '{source_name}' to be a path component in '{path_value}' for mask derivation.")


def build_train_mask(crack_mask, args):
    if crack_mask.mode != "L":
        crack_mask = crack_mask.convert("L")

    mask_array = np.array(crack_mask) > args.mask_threshold
    crack_mask = Image.fromarray(mask_array.astype(np.uint8) * 255)

    if args.mask2_rotate > 0 or args.mask2_translate > 0 or args.mask2_scale_min != 1.0:
        angle = random.uniform(-args.mask2_rotate, args.mask2_rotate)
        translate = (
            int(random.uniform(-args.mask2_translate, args.mask2_translate) * crack_mask.size[0]),
            int(random.uniform(-args.mask2_translate, args.mask2_translate) * crack_mask.size[1]),
        )
        scale = random.uniform(args.mask2_scale_min, args.mask2_scale_max)
        mask2 = tvf.affine(
            crack_mask,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=0.0,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
    else:
        mask2 = crack_mask.copy()

    if args.mask_dilate_radius > 0:
        kernel_size = args.mask_dilate_radius * 2 + 1
        dilated = crack_mask.filter(ImageFilter.MaxFilter(kernel_size))
    else:
        dilated = crack_mask

    mask2_array = np.array(mask2) > args.mask_threshold
    dilated_array = np.array(dilated) > args.mask_threshold
    train_mask_array = mask2_array & (~dilated_array)

    if not train_mask_array.any():
        train_mask_array = mask2_array

    return Image.fromarray(train_mask_array.astype(np.uint8) * 255)


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    with open(args.train_data_list, "r", encoding="utf-8") as handle:
        rel_paths = [line.strip() for line in handle if line.strip()]
    rel_paths = [path.replace("\\", "/") for path in rel_paths]

    for rel_path in rel_paths:
        mask_rel_path = replace_path_part(rel_path, args.image_folder_name, args.mask_folder_name)
        mask_rel_path = str(Path(mask_rel_path).with_suffix(args.mask_extension))
        mask_path = Path(args.train_data_root) / mask_rel_path
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask file: {mask_path}")

        crack_mask = Image.open(mask_path).convert("L")

        mask2_base = replace_path_part(rel_path, args.image_folder_name, args.mask2_folder_name)
        mask2_base = Path(mask2_base)
        output_dir = Path(args.train_data_root) / mask2_base.parent
        os.makedirs(output_dir, exist_ok=True)

        for mask2_index in range(args.mask2_count):
            train_mask = build_train_mask(crack_mask, args)
            output_path = output_dir / f"{mask2_base.stem}_mask2_{mask2_index}{args.mask2_extension}"
            train_mask.save(output_path)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
