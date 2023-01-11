import torch
from tqdm import tqdm
from utils.video_utils import torch_to_cv2
from utils.config import config
from PIL import Image
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from transformers import logging
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


def feedforward_gan(model, class_vectors, noise_vectors, batch_size, truncation):
    """
    Feedd-fowards the GAN and creates a collection of images
    :param model:
    :param class_vectors:
    :param noise_vectors:
    :param batch_size:
    :param class_ids:
    :param truncation:
    :return:
    """

    images = []
    n_batches = int(len(class_vectors) / batch_size)

    print("Generating GAN content...")
    for i in tqdm(range(n_batches)):
        cur_noise = torch.from_numpy(
            noise_vectors[i * batch_size:(i + 1) * batch_size]).to(config['device'])
        cur_class = torch.from_numpy(
            class_vectors[i * batch_size:(i + 1) * batch_size]).to(config['device'])

        with torch.no_grad():
            output = model(cur_noise, cur_class, truncation)

        images.append(output.cpu().numpy())

    if n_batches * batch_size < len(class_vectors):
        cur_noise = torch.from_numpy(
            noise_vectors[n_batches * batch_size:]).to(config['device'])
        cur_class = torch.from_numpy(
            class_vectors[n_batches * batch_size:]).to(config['device'])

        with torch.no_grad():
            output = model(cur_noise, cur_class, truncation)

        images.append(output.cpu().numpy())

    images = torch_to_cv2(images)
    return images
