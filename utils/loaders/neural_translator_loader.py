from PIL import Image
from utils.models import get_pretrained_mobile_net
from torch.utils.data import Dataset
import pickle
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from utils.loaders.neural_translator_utils import get_translator_data, get_classes
from utils.diffusion_utils import *
from utils.config import config
from os.path import join
import pandas as pd


class DiffuserOnlineHelper(Dataset):

    def __init__(self, image_sentiment_model=join(config['basedir'], "models/image_sentiment.model"), in_batch=5):
        # Load Diffusion
        self.vae, self.unet, self.scheduler, self.tokenizer, self.text_encoder = get_vae(
        ), get_unet(), get_scheduler(), get_tokenizer(), get_text_encoder()

        # Load image sentiment analysis model
        self.sentiment_model = get_pretrained_mobile_net()
        self.sentiment_model.to(config['device'])
        self.sentiment_model.load_state_dict(torch.load(image_sentiment_model))

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_image = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor(), normalize, ])

        self.height = 512
        self.width = 512
        self.num_inference_steps = 20
        self.guidance_scale = 7.5

        self.prompts = get_classes()
        self.n_class = len(self.prompts)

        self.in_batch = in_batch
        self.n_iters = int(self.n_class / in_batch)

    def __len__(self):
        return self.n_iters

    def __getitem__(self, idx):
        if idx > self.n_iters:
            raise StopIteration

        # Sample the space
        idx = np.random.randint(0, self.n_class, self.in_batch)
        conditional_vectors = get_text_embeds(
            self.tokenizer, self.text_encoder, list(self.prompts[idx]))
        noise_vectors = torch.randn((conditional_vectors.shape[0] // 2, self.unet.in_channels,
                                     self.height // 8, self.width // 8))
        conditional_vectors = conditional_vectors.to(config['device'])
        noise_vectors = noise_vectors.to(config['device'])

        with torch.no_grad():
            latents = produce_latents(
                self.unet, self.scheduler, conditional_vectors, height=self.height, width=self.width, latents=noise_vectors,
                num_inference_steps=self.num_inference_steps, guidance_scale=self.guidance_scale)
            output = decode_img_latents(self.vae, latents)

        # Convert to PIL Image
        # output = output.detach().cpu().numpy()
        # output = np.uint8(np.clip(((output + 1) / 2.0) * 256, 0, 255))
        # output = output.transpose((0, 2, 3, 1))
        images = []

        # Pre-process each image to feed them into image sentiment analysis model
        for i in range(len(output)):
            # cur_img = Image.fromarray(output[i])
            cur_img = output[i]
            cur_img = self.transform_image(cur_img)
            images.append(cur_img)
        images = torch.stack(images).to(config['device'])

        # Feed-forward image sentiment analysis
        sentiment = self.sentiment_model(images)

        conditional_vectors = conditional_vectors[self.in_batch:2*self.in_batch]
        conditional_vectors = conditional_vectors.cpu().detach().numpy()
        noise_vectors = noise_vectors.cpu().detach().numpy()
        sentiment = sentiment.cpu().detach().numpy()

        return conditional_vectors, noise_vectors, sentiment


def create_collection(epochs=1, dataset_path=join(config['basedir'], 'data/neural_translation_dataset.pickle')):
    """
    Sample latent space of Stable Diffusion
    """
    s = DiffuserOnlineHelper(in_batch=5)
    cond_vecs, noise_vecs, sent_vecs = [], [], []

    for i in range(epochs):
        print("Epoch: ", i)
        for cur_cond, cur_noise, cur_sentiment in tqdm(s):
            cond_vecs.append(cur_cond)
            noise_vecs.append(cur_noise)
            sent_vecs.append(cur_sentiment)
        print(cond_vecs)
    cond_vecs = np.concatenate(np.float16(cond_vecs))
    noise_vecs = np.concatenate(np.float16(noise_vecs))
    sent_vecs = np.concatenate(np.float16(sent_vecs))

    with open(dataset_path, "wb") as f:
        pickle.dump(cond_vecs, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(noise_vecs, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(sent_vecs, f, protocol=pickle.HIGHEST_PROTOCOL)


class NeuralTranslatorLoader(Dataset):

    def __init__(self, dataset_path=join(config['basedir'], 'data/neural_translation_dataset.pickle'),
                 train=False, deploy=True):

        cond_vecs, noise_vecs, sent_vecs = get_translator_data(
            dataset_path=dataset_path)

        self.cond_vecs = cond_vecs
        self.noise_vecs = noise_vecs
        self.sent_vecs = sent_vecs

        # Use the 80\% of the dataset of training
        thres = int(0.8 * len(self.cond_vecs))

        if train and not deploy:
            self.cond_vecs = self.cond_vecs[:thres]
            self.noise_vecs = self.noise_vecs[:thres]
            self.sent_vecs = self.sent_vecs[:thres]
        elif not train and not deploy:
            self.cond_vecs = self.cond_vecs[thres:]
            self.noise_vecs = self.noise_vecs[thres:]
            self.sent_vecs = self.sent_vecs[thres:]

    def __len__(self):
        return len(self.cond_vecs)

    def __getitem__(self, idx):
        # The valence-arousal is the input to the model
        data = torch.tensor(np.float32(self.sent_vecs[idx]))

        # The corresponding latent input that leads to the sentiment target
        target_cond = torch.tensor(np.float32(self.cond_vecs[idx]))
        target_noise = torch.tensor(np.float32(self.noise_vecs[idx]))

        return data, (target_cond, target_noise,)


def get_train_loaders(batch_size=32):
    train_dataset = NeuralTranslatorLoader(train=True, deploy=False)
    val_dataset = NeuralTranslatorLoader(train=False, deploy=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size)
    return train_loader, val_loader


def get_deploy_loaders(batch_size=32):
    train_dataset = NeuralTranslatorLoader(train=True, deploy=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
