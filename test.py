import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.utils import logging as dlogging
from PIL import Image, ImageOps
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


dlogging.set_verbosity_error()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SD3 mask-conditioned inpaint inference.")
    parser.add_argument(
        "--base",
        type=str,
        default="/home/libaoluo/sam2/diffusers-main/models/stable-diffusion-3-medium/" ,
        help="Path to the diffusers SD3 base model directory.",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default="/home/libaoluo/sam2/diffusers-main/examples/dreambooth/sd3-dreambooth/checkpoint-18000/",
        help="Directory containing LoRA weights and mask_encoder.bin.",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="/home/libaoluo/dataset/opensourse_datasets/BCL11k/images/c1.jpg",
        help="Path to background image.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default="/home/libaoluo/dataset/opensourse_datasets/BCL11k/masks/c1.png",
        help="Path to mask image (white=inpaint region).",
    )
    parser.add_argument("--output", type=str, default="out_sd3_maskcond.png")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a high-resolution inspection photo of crack pattern on a material surface",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="text, logo, watermark, people, objects, blur, lowres, painting, cartoon",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--mask_condition_scale", type=float, default=1.0)
    parser.add_argument("--precondition_outputs", action="store_true")
    parser.add_argument("--invert_mask", action="store_true")
    parser.add_argument("--grid_output", type=str, default="out_sd3_maskcond_grid.png")
    return parser.parse_args()


def build_mask_encoder(latent_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, latent_channels, kernel_size=3, padding=1),
        nn.SiLU(),
        nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
    )


def encode_prompt(tokenizer1, tokenizer2, tokenizer3, te1, te2, te3, prompt: str, device: str, dtype: torch.dtype):
    t1 = tokenizer1(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(
        device
    )
    out1 = te1(t1, output_hidden_states=True)
    pooled1 = out1[0]
    hid1 = out1.hidden_states[-2]

    t2 = tokenizer2(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(
        device
    )
    out2 = te2(t2, output_hidden_states=True)
    pooled2 = out2[0]
    hid2 = out2.hidden_states[-2]

    clip_hid = torch.cat([hid1, hid2], dim=-1)
    pooled = torch.cat([pooled1, pooled2], dim=-1)

    t3 = tokenizer3(
        prompt, padding="max_length", max_length=77, truncation=True, add_special_tokens=True, return_tensors="pt"
    ).input_ids.to(device)
    t5 = te3(t3)[0]

    if clip_hid.shape[-1] < t5.shape[-1]:
        clip_hid = F.pad(clip_hid, (0, t5.shape[-1] - clip_hid.shape[-1]))
    prompt_embeds = torch.cat([clip_hid, t5], dim=-2)
    return prompt_embeds.to(dtype), pooled.to(dtype)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (
        torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    tokenizer1 = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer")
    tokenizer2 = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer_2")
    tokenizer3 = T5TokenizerFast.from_pretrained(args.base, subfolder="tokenizer_3")

    te1 = CLIPTextModelWithProjection.from_pretrained(args.base, subfolder="text_encoder").to(device, dtype=dtype)
    te2 = CLIPTextModelWithProjection.from_pretrained(args.base, subfolder="text_encoder_2").to(device, dtype=dtype)
    te3 = T5EncoderModel.from_pretrained(args.base, subfolder="text_encoder_3").to(device, dtype=dtype)

    vae = AutoencoderKL.from_pretrained(args.base, subfolder="vae").to(device, dtype=torch.float32)
    transformer = SD3Transformer2DModel.from_pretrained(args.base, subfolder="transformer").to(device, dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base, subfolder="scheduler")

    from diffusers import StableDiffusion3Pipeline
    from diffusers.utils import convert_unet_state_dict_to_peft
    from peft import LoraConfig, set_peft_model_state_dict

    lora_state = StableDiffusion3Pipeline.lora_state_dict(args.lora_dir)
    target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]
    transformer.add_adapter(
        LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, init_lora_weights="gaussian", target_modules=target_modules)
    )
    t_state = {k.replace("transformer.", ""): v for k, v in lora_state.items() if k.startswith("transformer.")}
    t_state = convert_unet_state_dict_to_peft(t_state)
    set_peft_model_state_dict(transformer, t_state, adapter_name="default")

    mask_encoder = build_mask_encoder(vae.config.latent_channels).to(device, dtype=dtype)
    mask_encoder_path = os.path.join(args.lora_dir, "mask_encoder.bin")
    if os.path.exists(mask_encoder_path):
        mask_encoder.load_state_dict(torch.load(mask_encoder_path, map_location="cpu"))
    else:
        raise FileNotFoundError(f"mask_encoder.bin not found at: {mask_encoder_path}")

    image = Image.open(args.background).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    if args.invert_mask:
        mask = ImageOps.invert(mask)

    image = image.resize((args.resolution, args.resolution), resample=Image.BICUBIC)
    mask = mask.resize((args.resolution, args.resolution), resample=Image.NEAREST)
    mask = mask.point(lambda p: 255 if p > 0 else 0)

    img = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    img = img * 2.0 - 1.0
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

    m = torch.from_numpy(np.array(mask)).float()
    if m.max() <= 1.0:
        m = m.clamp(0, 1)
    else:
        m = (m > 127).float()
    m = m.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)

    prompt_embeds, pooled = encode_prompt(
        tokenizer1, tokenizer2, tokenizer3, te1, te2, te3, args.prompt, device, dtype
    )
    negative_prompt = args.negative_prompt or ""
    negative_embeds, negative_pooled = encode_prompt(
        tokenizer1, tokenizer2, tokenizer3, te1, te2, te3, negative_prompt, device, dtype
    )

    latents = vae.encode(img).latent_dist.sample(generator=generator)
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    latents = latents.to(dtype)

    scheduler.set_timesteps(args.steps, device=device)
    timesteps = scheduler.timesteps.to(device)
    sigmas_all = scheduler.sigmas.to(device=device, dtype=latents.dtype)

    batch_size = latents.shape[0]
    m_latent = F.interpolate(m, size=latents.shape[-2:], mode="nearest").to(latents.dtype)
    x_bg = latents
    noise0 = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
    x0 = (1 - m_latent) * x_bg + m_latent * noise0
    x1_hat = x0.clone()

    feat = mask_encoder(m_latent) if args.mask_condition_scale != 0 else None

    for i, t in enumerate(timesteps):
        sigma = sigmas_all[i].view(1, 1, 1, 1)
        xt = (1.0 - sigma) * x0 + sigma * x1_hat

        if feat is not None:
            xt = xt + args.mask_condition_scale * feat

        if args.cfg != 1.0:
            xt_in = torch.cat([xt, xt], dim=0)
            t_in = torch.cat([t.expand(batch_size), t.expand(batch_size)], dim=0)
            enc = torch.cat([negative_embeds, prompt_embeds], dim=0)
            pool = torch.cat([negative_pooled, pooled], dim=0)
        else:
            xt_in = xt
            t_in = t.expand(batch_size)
            enc = prompt_embeds
            pool = pooled

        pred = transformer(
            hidden_states=xt_in,
            timestep=t_in,
            encoder_hidden_states=enc,
            pooled_projections=pool,
            return_dict=False,
        )[0]

        if args.precondition_outputs:
            pred = pred * (-sigma) + xt_in

        if args.cfg != 1.0:
            pred_uncond, pred_text = pred.chunk(2, dim=0)
            pred_cfg = pred_uncond + args.cfg * (pred_text - pred_uncond)
        else:
            pred_cfg = pred

        x1_hat = x0 + pred_cfg
        x1_hat = (1 - m_latent) * x_bg + m_latent * x1_hat

    latents = x1_hat

    latents = latents.to(torch.float32)
    latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    out = vae.decode(latents).sample
    out = (out / 2 + 0.5).clamp(0, 1)
    out = (out[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(out).save(args.output)
    print("Saved:", args.output)

    mask_vis = np.array(mask.convert("L"))
    overlay = image.copy()
    overlay_arr = np.array(overlay).astype("float32")
    mask_color = np.zeros_like(overlay_arr)
    mask_color[..., 0] = 255
    mask_alpha = (mask_vis.astype("float32") / 255.0) * 0.5
    overlay_arr = overlay_arr * (1 - mask_alpha[..., None]) + mask_color * mask_alpha[..., None]
    overlay = Image.fromarray(overlay_arr.astype("uint8"))

    grid = Image.new("RGB", (args.resolution * 2, args.resolution * 2))
    grid.paste(image, (0, 0))
    grid.paste(mask.convert("RGB"), (args.resolution, 0))
    grid.paste(overlay, (0, args.resolution))
    grid.paste(Image.fromarray(out), (args.resolution, args.resolution))
    os.makedirs(os.path.dirname(args.grid_output) or ".", exist_ok=True)
    grid.save(args.grid_output)
    print("Saved grid:", args.grid_output)


if __name__ == "__main__":
    main()
