import torch
from utils.models import SoundMLP
from utils.sound_features import extract_features_from_file
import numpy as np
from utils.config import config
from os.path import join


class SoundSentimentExtractor:

    def __init__(self, sound_model_path=join(config['basedir'], "models/mlp.model"),
                 space_statistics_path=join(config['basedir'], "models/space_statistics.pickle")):
        # Step 1: Load the sound to sentiment model
        self.model = SoundMLP()
        self.model.to(config['device'])
        self.model.eval()
        self.model.load_neural_model(sound_model_path)
        self.model.load_neural_space_statistics(space_statistics_path)

        # Features are extracted every 0.5sec
        self.period = 0.5

    def extract_sentiment(self, audio_path, smoothing_window):
        # Step 0: Load the file
        data, power_features = extract_features_from_file(audio_path)

        # Step 1: Feed-forward through the network
        data = torch.tensor(np.float32(data)).to(config['device'])
        predictions_np = self.model.get_np_features(data)

        # Step 2: Down-sample and smooth the sentiment signal
        predictions_np = smooth_audio_sentiment(
            predictions_np, smoothing_window)
        return predictions_np, power_features


def smooth_audio_sentiment(sentiment_features, smoothing_window=10):
    n_samples = int(len(sentiment_features) /
                    float(smoothing_window)) * smoothing_window
    sentiment_features = sentiment_features[:n_samples].reshape(
        (-1, smoothing_window, 2))
    sentiment_features = np.mean(sentiment_features, 1)
    return sentiment_features
