from pytorch_pretrained_biggan import (truncated_noise_sample)
from utils.models import NeuralTranslator
import torch
from utils.loaders.neural_translator_loader import get_classes
from utils.config import config
from os.path import join


class NeuralSentimentTranslator:

    def __init__(self, translator_path=join(config['basedir'], "models/neural_translator_0.model")):

        # Load the class
        self.class_name = get_classes()

        self.net = NeuralTranslator()
        self.net.to(config['device'])
        self.net.load_state_dict(torch.load(translator_path))

    def translate_sentiment(self, audio_sentiment):
        song_words = []

        audio_sentiment = torch.tensor(audio_sentiment).to(config['device'])
        class_vectors = self.net(audio_sentiment)
        class_vectors = torch.softmax(class_vectors, dim=1)

        class_ids = class_vectors.argmax(1).cpu().detach().numpy()
        for i in class_ids:
            song_words.append(self.class_name[i])

        class_vectors = class_vectors.cpu().detach().numpy()
        return class_vectors, song_words
