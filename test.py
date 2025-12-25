import os
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageOps
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from transformers import CLIPTokenizer, T5TokenizerFast, CLIPTextModelWithProjection, T5EncoderModel
from diffusers.utils import logging as dlogging

dlogging.set_verbosity_error()

# ===== 1) 你的路径配置 =====
BASE = "/home/libaoluo/sam2/diffusers-main/models/stable-diffusion-3-medium/"  # ✅必须是 diffusers 目录，不是 .safetensors 单文件
LORA_DIR = "/home/libaoluo/sam2/diffusers-main/examples/dreambooth/sd3-dreambooth/checkpoint-18000/"  # 训练输出目录，里面有 pytorch_lora_weights.safetensors 和 mask_encoder.bin
LORA_NAME = "pytorch_lora_weights.safetensors"  # 默认这个名字
MASK_ENCODER_BIN = os.path.join(LORA_DIR, "mask_encoder.bin")

BG_PATH = "/home/libaoluo/dataset/opensourse_datasets/BCL11k/images/c1.jpg"
MASK_PATH = "/home/libaoluo/dataset/opensourse_datasets/BCL11k/masks/c1.png"
OUT_PATH = "./out_sd3_maskcond.png"

# ===== 2) 生成参数 =====
PROMPT = "a high-resolution inspection photo of crack pattern on a material surface"
NEG = "text, logo, watermark, people, objects, blur, lowres, painting, cartoon"
SEED = 42
STEPS = 30
CFG = 5.0
RES = 512  # 训练用的分辨率保持一致
MIXED = "bf16"  # 与训练一致
MASK_COND_SCALE = 1.0  # 对齐 args.mask_conditioning_scale
PRECONDITION_OUTPUTS = 1  # 对齐 args.precondition_outputs
INVERT_MASK = False  # mask 黑白反了就 True（白=重绘区）

device = "cuda"
dtype = torch.bfloat16 if MIXED == "bf16" else (torch.float16 if MIXED == "fp16" else torch.float32)

# ===== 3) 你训练里用的 mask_encoder 结构（必须一致）=====
def build_mask_encoder(latent_channels: int):
    return nn.Sequential(
        nn.Conv2d(1, latent_channels, kernel_size=3, padding=1),
        nn.SiLU(),
        nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
    )

# ===== 4) 文本编码（对齐 SD3: 2个CLIP + 1个T5）=====
def encode_prompt(tokenizer1, tokenizer2, tokenizer3, te1, te2, te3, prompt: str):
    # CLIP1
    t1 = tokenizer1(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    out1 = te1(t1, output_hidden_states=True)
    pooled1 = out1[0]
    hid1 = out1.hidden_states[-2]

    # CLIP2
    t2 = tokenizer2(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    out2 = te2(t2, output_hidden_states=True)
    pooled2 = out2[0]
    hid2 = out2.hidden_states[-2]

    clip_hid = torch.cat([hid1, hid2], dim=-1)
    pooled = torch.cat([pooled1, pooled2], dim=-1)

    # T5
    t3 = tokenizer3(prompt, padding="max_length", max_length=77, truncation=True, add_special_tokens=True, return_tensors="pt").input_ids.to(device)
    t5 = te3(t3)[0]  # [B, seq, d]

    # pad clip hidden to t5 dim then concat on seq dim（与你训练代码一致）
    if clip_hid.shape[-1] < t5.shape[-1]:
        clip_hid = F.pad(clip_hid, (0, t5.shape[-1] - clip_hid.shape[-1]))
    prompt_embeds = torch.cat([clip_hid, t5], dim=-2)
    return prompt_embeds.to(dtype), pooled.to(dtype)

@torch.no_grad()
def main():
    torch.manual_seed(SEED)
    g = torch.Generator(device=device).manual_seed(SEED)

    # 1) load tokenizers & text encoders
    tokenizer1 = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer")
    tokenizer2 = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer_2")
    tokenizer3 = T5TokenizerFast.from_pretrained(BASE, subfolder="tokenizer_3")

    te1 = CLIPTextModelWithProjection.from_pretrained(BASE, subfolder="text_encoder").to(device, dtype=dtype)
    te2 = CLIPTextModelWithProjection.from_pretrained(BASE, subfolder="text_encoder_2").to(device, dtype=dtype)
    te3 = T5EncoderModel.from_pretrained(BASE, subfolder="text_encoder_3").to(device, dtype=dtype)

    # 2) load vae / transformer / scheduler
    vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae").to(device, dtype=torch.float32)  # 你训练里 VAE 用 fp32
    transformer = SD3Transformer2DModel.from_pretrained(BASE, subfolder="transformer").to(device, dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(BASE, subfolder="scheduler")

    # 3) load LoRA（用 SD3Transformer2DModel 的 adapter/peft 权重）
    # 直接用 Pipeline 的 load_lora_weights 最方便，但这里我们只用 transformer，
    # 因此用 diffusers 的 lora_state_dict 读取再 set 到 transformer。
    from diffusers import StableDiffusion3Pipeline
    lora_state = StableDiffusion3Pipeline.lora_state_dict(LORA_DIR)
    # 只取 transformer.* 并去掉前缀
    from diffusers.utils import convert_unet_state_dict_to_peft
    from peft import set_peft_model_state_dict

    # 给 transformer 加一个默认 adapter（和训练一致）
    from peft import LoraConfig
    target_modules = [
        "attn.add_k_proj","attn.add_q_proj","attn.add_v_proj","attn.to_add_out",
        "attn.to_k","attn.to_out.0","attn.to_q","attn.to_v",
    ]
    transformer.add_adapter(LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, init_lora_weights="gaussian", target_modules=target_modules))

    t_state = {k.replace("transformer.", ""): v for k, v in lora_state.items() if k.startswith("transformer.")}
    t_state = convert_unet_state_dict_to_peft(t_state)
    set_peft_model_state_dict(transformer, t_state, adapter_name="default")

    # 4) load mask_encoder.bin
    mask_encoder = build_mask_encoder(vae.config.latent_channels).to(device, dtype=dtype)
    if os.path.exists(MASK_ENCODER_BIN):
        mask_encoder.load_state_dict(torch.load(MASK_ENCODER_BIN, map_location="cpu"))
    else:
        raise FileNotFoundError(f"mask_encoder.bin not found at: {MASK_ENCODER_BIN}")

    # 5) load background + mask
    image = Image.open(BG_PATH).convert("RGB")
    mask = Image.open(MASK_PATH).convert("L")
    if INVERT_MASK:
        mask = ImageOps.invert(mask)

    image = image.resize((RES, RES), resample=Image.BICUBIC)
    mask = mask.resize((RES, RES), resample=Image.NEAREST)
    mask = mask.point(lambda p: 255 if p > 127 else 0)

    # to tensor in [-1, 1]
    img = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    img = img * 2.0 - 1.0
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

    m = torch.from_numpy(np.array(mask)).float() / 255.0
    m = (m > 0.5).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)  # dtype=bf16/fp16

    # 6) encode prompt
    prompt_embeds, pooled = encode_prompt(tokenizer1, tokenizer2, tokenizer3, te1, te2, te3, PROMPT)

    # 7) encode to latents (与你训练一致：shift_factor & scaling_factor)
    latents = vae.encode(img).latent_dist.sample(generator=g)
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    latents = latents.to(dtype)

    # 8) mask_latents -> same spatial as latents
    mask_latents = F.interpolate(m, size=latents.shape[-2:], mode="nearest")
    if MASK_COND_SCALE != 0:
        mask_latents = mask_latents.to(dtype=next(mask_encoder.parameters()).dtype, device=device)
        feat = mask_encoder(mask_latents)
        latents = latents + MASK_COND_SCALE * feat

    # 9) 采样（flow-matching scheduler：用其 timesteps + sigmas 方式）
    scheduler.set_timesteps(STEPS, device=device)
    timesteps = scheduler.timesteps.to(device)
    sigmas_all = scheduler.sigmas.to(device=device, dtype=latents.dtype)

    B = latents.shape[0]
    x = torch.randn(latents.shape, generator=g, device=latents.device, dtype=latents.dtype)

    # mask feature 只算一次
    mask_latents = F.interpolate(m, size=latents.shape[-2:], mode="nearest").to(latents.dtype)
    feat = mask_encoder(mask_latents) if MASK_COND_SCALE != 0 else None

    for i, t in enumerate(timesteps):
        t_batch = t.expand(B)  # 1D [B]
        sigma = sigmas_all[i].view(1, 1, 1, 1)

        zt = (1.0 - sigma) * latents + sigma * x
        if feat is not None:
            zt = zt + MASK_COND_SCALE * feat  # ✅ 对齐训练：每步注入

        model_pred = transformer(
            hidden_states=zt,
            timestep=t_batch,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled,
            return_dict=False,
        )[0]

        if PRECONDITION_OUTPUTS:
            model_pred = model_pred * (-sigma) + zt

        x = model_pred

    # 10) decode latents
    x = x.to(torch.float32)
    x = x / vae.config.scaling_factor + vae.config.shift_factor
    out = vae.decode(x).sample
    out = (out / 2 + 0.5).clamp(0, 1)
    out = (out[0].permute(1,2,0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(out).save(OUT_PATH)
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
