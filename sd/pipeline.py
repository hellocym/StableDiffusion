import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(
    prompt: str,
    uncond_prompt: str, # negative prompt or ""
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):

    with torch.no_grad():
        if not (0 <= strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        clip = models['clip']
        clip.to(device)

        if do_cfg:
            # Convert the prompt to tokens
            cond_tokens = tokenizer.batch_encode_plus([prompt], max_length=77, padding="max_length").input_ids
            # Convert tokens into tensor
            # (Batch_size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_size, Seq_Len) -> (Batch_size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], max_length=77, padding="max_length").input_ids
            # (Batch_size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_size, Seq_Len) -> (Batch_size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            # (2, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context], dim=1)
        else:
            # Convert the prompt to tokens
            tokens = tokenizer.batch_encode_plus([prompt], max_length=77, padding="max_length").input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, Seq_Len, Dim)
            context = clip(tokens)
        # offload model if necessary
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")
        
        latent_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            
            input_image_tensor = rescale(input_image_tensor, [0, 255], [-1, 1])
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Sample the noise
            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
            # pass the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)


            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            # offload model if necessary
            to_idle(encoder)
        else:
            latents = torch.randn(latent_shape, generator=generator, device=device)
        
        diffusion = models['diffusion']
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents

            if do_cfg:
                # (Batch_size, 4, Height, Width) -> (2 * Batch_size, 4, Height, Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(timestep, latents, model_output)

        # offload model if necessary
        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, [-1, 1], [0, 255], clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to('cpu', torch.uint8).numpy()

        return images[0]

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)

    return x

def get_time_embedding(timestep):
    # (160, )
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32))
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 320)
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)