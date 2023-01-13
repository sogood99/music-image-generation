import cv2
import torch
from tqdm import tqdm
from utils.config import config
from PIL import Image
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from transformers import logging
import numpy as np
logging.set_verbosity_error()

model_id = 'CompVis/stable-diffusion-v1-4'
model_id_trans = 'openai/clip-vit-large-patch14'


def get_vae():
    vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae')
    vae = vae.to(config['device'])
    return vae


def get_unet():
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder='unet')
    unet = unet.to(config['device'])
    return unet


def get_tokenizer():
    tokenizer = CLIPTokenizer.from_pretrained(model_id_trans)
    return tokenizer


def get_text_encoder():
    text_encoder = CLIPTextModel.from_pretrained(model_id_trans)
    text_encoder = text_encoder.to(config['device'])
    return text_encoder


def get_scheduler():
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)
    return scheduler


def get_text_embeds(tokenizer, text_encoder, prompt):
    # Tokenize text and get embeddings
    text_input = tokenizer(
        prompt, padding='max_length', max_length=tokenizer.model_max_length,
        truncation=True, return_tensors='pt')
    with torch.no_grad():
        text_embeddings = text_encoder(
            text_input.input_ids.to(config['device']))[0]

    # Do the same for unconditional embeddings
    uncond_input = tokenizer(
        [''] * len(prompt), padding='max_length',
        max_length=tokenizer.model_max_length, return_tensors='pt')
    with torch.no_grad():
        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(config['device']))[0]

    # Cat for final embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings


def produce_latents(unet, scheduler, text_embeddings, height=512, width=512,
                    num_inference_steps=50, guidance_scale=7.5, latents=None):
    if latents is None:
        latents = torch.randn((text_embeddings.shape[0] // 2, unet.in_channels,
                               height // 8, width // 8))
    latents = latents.to(config['device'])

    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.sigmas[0]

    with torch.autocast(config['device']):
        for i, t in tqdm(enumerate(scheduler.timesteps), total=num_inference_steps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * \
                (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)['prev_sample']

    return latents


def decode_img_latents(vae, latents):
    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        imgs = vae.decode(latents)

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in imgs]
    return pil_images


def prompt_to_img(prompts, height=512, width=512, num_inference_steps=50,
                  guidance_scale=7.5, latents=None):
    if isinstance(prompts, str):
        prompts = [prompts]

    # Prompts -> text embeds
    text_embeds = get_text_embeds(prompts)

    # Text embeds -> img latents
    latents = produce_latents(
        text_embeds, height=height, width=width, latents=latents,
        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

    # Img latents -> imgs
    imgs = decode_img_latents(latents)

    return imgs

# interpolation method


def interpolateTextEmbed_linear(embed_1, embed_2, k):
    embed_shape = embed_1.shape
    v_interpolate = (1-k)*embed_1 + k*embed_2
    return v_interpolate.view(embed_shape)


def interpolateTextEmbed_square_correction(embed_1, embed_2, k, correction_model):
    embed_shape = embed_1.shape
    v_correction = correction_model(
        embed_1[1].view(1, -1), embed_2[1].view(1, -1))
    v_correction = torch.cat(
        [embed_1[0][None, :], v_correction[None, :].view(1, *embed_shape[1:])], dim=0)
    v_interpolate = (1-k)*embed_1 + k*embed_2 + 4*k*(1-k)*v_correction
    return v_interpolate.view(embed_shape)


def interpolateTextEmbed_linear_correction(embed_1, embed_2, k, correction_model):
    embed_shape = embed_1.shape
    v_correction = correction_model(
        embed_1[1].view(1, -1), embed_2[1].view(1, -1))
    v_correction = torch.cat(
        [embed_1[0][None, :], v_correction[None, :].view(1, *embed_shape[1:])], dim=0)
    if k < 0.5:
        v_interpolate = (1-k)*embed_1 + k*embed_2 + 2*k*v_correction
    else:
        v_interpolate = (1-k)*embed_1 + k*embed_2 + 2*(1-k)*v_correction
    return v_interpolate.view(embed_shape)


def interpolateTextEmbed_sphere(embed_1, embed_2, k):
    embed_shape = embed_1.shape
    embed_1 = embed_1.view(1, -1)
    embed_2 = embed_2.view(1, -1)

    inner_product = (embed_1 * embed_2).sum(dim=1)
    a_norm = embed_1.pow(2).sum(dim=1).pow(0.5)
    b_norm = embed_2.pow(2).sum(dim=1).pow(0.5)
    embed_angle = torch.acos(inner_product / (a_norm * b_norm))

    v_interpolate = torch.sin((1-k)*embed_angle)/torch.sin(embed_angle) * embed_1 \
        + torch.sin(k*embed_angle)/torch.sin(embed_angle) * embed_2
    return v_interpolate.view(embed_shape)


def interpolateLatentSpace_linear(latent_1, latent_2, k):
    latent_shape = latent_1.shape
    v_interpolate = (1-k) * latent_1 + k * latent_2
    return v_interpolate.view(latent_shape)


def interpolateLatentSpace_sqrt(latent_1, latent_2, k):
    latent_shape = latent_1.shape
    v_interpolate = np.sqrt(1-k) * latent_1 + np.sqrt(k) * latent_2
    return v_interpolate.view(latent_shape)


def interpolateLatentSpace_sphere(latent_1, latent_2, k):
    latent_shape = latent_1.shape
    latent_1 = latent_1.view(1, -1)
    latent_2 = latent_2.view(1, -1)

    inner_product = (latent_1 * latent_2).sum(dim=1)
    a_norm = latent_1.pow(2).sum(dim=1).pow(0.5)
    b_norm = latent_2.pow(2).sum(dim=1).pow(0.5)
    latent_angle = torch.acos(inner_product / (a_norm * b_norm))

    v_interpolate = torch.sin((1-k)*latent_angle)/torch.sin(latent_angle) * latent_1 \
        + torch.sin(k*latent_angle)/torch.sin(latent_angle) * latent_2
    return v_interpolate.view(latent_shape)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid


def images_to_video(images, fps, output_path):
    h, w, c = np.array(images[0]).shape
    frame_size = (h, w)

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*"DIVX"), 15, frame_size)

    for img in images:
        new_img = np.array(img)
        out.write(cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))

    out.release()


def generate_video(prompts, unet, scheduler, vae, tokenizer, text_encoder, num_between=10):
    # test out perturbing latent
    num_inference_steps = 20
    guidance_scale = 7.5

    height, width = 512, 512

    prompt_2 = [prompts[0]]
    text_embed_2 = get_text_embeds(tokenizer, text_encoder, prompt_2)
    latents_2 = torch.randn((text_embed_2.shape[0] // 2, unet.in_channels,
                             height // 8, width // 8))

    lmd = np.linspace(0, 1, num=num_between)

    images = []
    # Text embeds -> img latents

    for it in range(1, len(prompts)):
        prompt_1 = prompt_2
        prompt_2 = [prompts[it]]

        text_embed_1 = get_text_embeds(tokenizer, text_encoder, prompt_1)
        text_embed_2 = get_text_embeds(tokenizer, text_encoder, prompt_2)

        latents_1 = latents_2
        latents_2 = torch.randn((text_embed_2.shape[0] // 2, unet.in_channels,
                                height // 8, width // 8))

        for i, k in enumerate(lmd):
            latents = produce_latents(unet, scheduler,
                                      interpolateTextEmbed_linear(
                                          text_embed_1, text_embed_2, k),
                                      height=height, width=width,
                                      num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                                      latents=interpolateLatentSpace_sphere(latents_1, latents_2, k))

            # Img latents -> imgs
            imgs = decode_img_latents(vae, latents)
            images += [imgs[0]]
            print("Finished {:.3f} of {}".format((i+1)/num_between, it+1))
    return images
