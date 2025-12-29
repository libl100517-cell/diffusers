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
import copy
import gc
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_nn_functional
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf
from tqdm.auto import tqdm
from transformers import PretrainedConfig

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.37.0.dev0")

logger = get_logger(__name__)


def load_text_encoders(class_one, class_two, class_three, args):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    if model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    raise ValueError(f"{model_class} is not supported.")


def _encode_prompt_with_t5(text_encoder, tokenizer, max_sequence_length, prompt, device=None):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    prompt_embeds = prompt_embeds.view(prompt_embeds.shape[0], seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt: str, device=None):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    prompt_embeds = prompt_embeds.view(prompt_embeds.shape[0], seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length, device=None):
    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoder=text_encoders[-1],
        tokenizer=tokenizers[-1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        device=device,
    )

    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    return prompt_embeds, pooled_prompt_embeds


class CrackInpaintDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        with open(args.train_data_list, "r", encoding="utf-8") as handle:
            rel_paths = [line.strip() for line in handle if line.strip()]
        rel_paths = [path.replace("\\", "/") for path in rel_paths]

        def replace_path_part(path_value, source_name, target_name):
            normalized_value = path_value.replace("\\", "/")
            parts = list(Path(normalized_value).parts)
            if source_name in parts:
                source_index = parts.index(source_name)
                parts[source_index] = target_name
                return str(Path(*parts))
            raise ValueError(
                f"Expected '{source_name}' to be a path component in '{path_value}' for mask derivation."
            )

        self.image_paths = [
            str(Path(args.train_data_root) / rel_path).replace("\\", "/") for rel_path in rel_paths
        ]
        mask_rel_paths = []
        for rel_path in rel_paths:
            mask_path = replace_path_part(rel_path, args.image_folder_name, args.mask_folder_name)
            mask_path = str(Path(mask_path).with_suffix(args.mask_extension))
            mask_rel_paths.append(mask_path)
        self.mask_paths = [
            str(Path(args.train_data_root) / rel_path).replace("\\", "/") for rel_path in mask_rel_paths
        ]
        self.args = args
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=InterpolationMode.NEAREST),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def _generate_low_frequency_noise(self, height, width):
        downsample = self.args.noise_downsample
        noise_height = max(1, height // downsample)
        noise_width = max(1, width // downsample)
        noise = np.random.rand(noise_height, noise_width, 3).astype(np.float32)
        noise_image = Image.fromarray((noise * 255).astype(np.uint8))
        noise_image = noise_image.resize((width, height), resample=Image.BILINEAR)
        return np.array(noise_image).astype(np.float32)

    def _apply_masked_noise(self, image, mask):
        if mask.mode != "L":
            mask = mask.convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, resample=Image.NEAREST)
        image_array = np.array(image.convert("RGB")).astype(np.float32)
        mask_array = np.array(mask) > self.args.mask_threshold

        noise_array = self._generate_low_frequency_noise(*image_array.shape[:2])
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

    def _build_train_mask(self, crack_mask):
        if crack_mask.mode != "L":
            crack_mask = crack_mask.convert("L")

        mask_array = np.array(crack_mask) > self.args.mask_threshold
        crack_mask = Image.fromarray(mask_array.astype(np.uint8) * 255)

        if self.args.mask2_rotate > 0 or self.args.mask2_translate > 0 or self.args.mask2_scale_min != 1.0:
            angle = random.uniform(-self.args.mask2_rotate, self.args.mask2_rotate)
            translate = (
                int(random.uniform(-self.args.mask2_translate, self.args.mask2_translate) * crack_mask.size[0]),
                int(random.uniform(-self.args.mask2_translate, self.args.mask2_translate) * crack_mask.size[1]),
            )
            scale = random.uniform(self.args.mask2_scale_min, self.args.mask2_scale_max)
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

        if self.args.mask_dilate_radius > 0:
            kernel_size = self.args.mask_dilate_radius * 2 + 1
            dilated = crack_mask.filter(ImageFilter.MaxFilter(kernel_size))
        else:
            dilated = crack_mask

        mask2_array = np.array(mask2) > self.args.mask_threshold
        dilated_array = np.array(dilated) > self.args.mask_threshold
        train_mask_array = mask2_array & (~dilated_array)

        if not train_mask_array.any():
            train_mask_array = mask2_array

        return Image.fromarray(train_mask_array.astype(np.uint8) * 255)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")

        train_mask = self._build_train_mask(mask)
        masked_image = self._apply_masked_noise(image, train_mask)

        pixel_values = self.image_transforms(image)
        masked_pixel_values = self.image_transforms(masked_image)
        train_mask_tensor = self.mask_transforms(train_mask)

        prompt = self.args.default_prompt
        return {
            "pixel_values": pixel_values,
            "masked_pixel_values": masked_pixel_values,
            "mask": train_mask_tensor,
            "prompt": prompt,
        }


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    masked_pixel_values = torch.stack([example["masked_pixel_values"] for example in examples])
    mask = torch.stack([example["mask"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    return {
        "pixel_values": pixel_values,
        "masked_pixel_values": masked_pixel_values,
        "mask": mask,
        "prompts": prompts,
    }


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="SD3 inpainting LoRA training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="sd3-inpaint-lora")
    parser.add_argument("--train_data_list", type=str, required=True)
    parser.add_argument("--train_data_root", type=str, required=True)
    parser.add_argument("--image_folder_name", type=str, default="images")
    parser.add_argument("--mask_folder_name", type=str, default="masks")
    parser.add_argument("--mask_extension", type=str, default=".png")
    parser.add_argument("--mask_threshold", type=int, default=0)
    parser.add_argument("--mask_dilate_radius", type=int, default=5)
    parser.add_argument("--mask2_rotate", type=float, default=15.0)
    parser.add_argument("--mask2_translate", type=float, default=0.15)
    parser.add_argument("--mask2_scale_min", type=float, default=0.8)
    parser.add_argument("--mask2_scale_max", type=float, default=1.2)
    parser.add_argument("--noise_downsample", type=int, default=8)
    parser.add_argument("--mask_loss_weight", type=float, default=1.0)
    parser.add_argument("--background_loss_weight", type=float, default=0.1)
    parser.add_argument("--default_prompt", type=str, default="")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.mask_extension and not args.mask_extension.startswith("."):
        args.mask_extension = f".{args.mask_extension}"

    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder"
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, args
    )

    tokenizer_one = transformers.CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    tokenizer_two = transformers.CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )
    tokenizer_three = transformers.T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_3", revision=args.revision
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                weights.pop()

            StableDiffusion3Pipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)
        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    "Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}."
                )

        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, transformer.parameters())),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = CrackInpaintDataset(args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=0,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("sd3-inpaint-lora", config=vars(args))

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    transformer.train()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                masked_pixel_values = batch["masked_pixel_values"].to(dtype=vae.dtype)
                masked_latents = vae.encode(masked_pixel_values).latent_dist.sample()
                masked_latents = (masked_latents - vae.config.shift_factor) * vae.config.scaling_factor
                masked_latents = masked_latents.to(dtype=weight_dtype)

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal",
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                mask = batch["mask"].to(dtype=weight_dtype)
                mask = torch_nn_functional.interpolate(
                    mask, size=(model_input.shape[-2], model_input.shape[-1]), mode="nearest"
                )
                noisy_model_input = noisy_model_input * mask + masked_latents * (1.0 - mask)

                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders,
                    tokenizers,
                    batch["prompts"],
                    max_sequence_length=256,
                    device=accelerator.device,
                )
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)

                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                weighting = compute_loss_weighting_for_sd3(weighting_scheme="logit_normal", sigmas=sigmas)
                target = noise - model_input

                loss_weights = (
                    mask * args.mask_loss_weight + (1.0 - mask) * args.background_loss_weight
                ).to(dtype=target.dtype)
                loss = torch.mean(
                    (weighting.float() * loss_weights * (model_pred.float() - target.float()) ** 2).reshape(
                        target.shape[0], -1
                    ),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    accelerator.log({"loss": loss.detach().item()}, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save_state(args.output_dir)

    accelerator.end_training()
    gc.collect()


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
