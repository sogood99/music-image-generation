import pickle
from sklearn.cluster import KMeans
import collections
import numpy as np
import pandas as pd
import os
from utils.config import config


def get_translator_data(dataset_path='./data/neural_translation_dataset.pickle'):
    with open(dataset_path, "rb") as f:
        cond_vecs = pickle.load(f)
        noise_vecs = pickle.load(f)
        sent_vecs = pickle.load(f)

    # Normalize sentiments similarly to the normalization performed for the audio model
    sent_vecs = (sent_vecs - np.mean(sent_vecs, axis=0)) / \
        np.std(sent_vecs, axis=0)

    return cond_vecs, noise_vecs, sent_vecs


def get_classes():
    df = pd.read_csv(os.path.join(config['basedir'], "data/oasis/OASIS.csv"))
    themes = df.Theme.to_numpy()
    themes = np.array(list(map(lambda s: s[:-2].strip(), themes)))
    prompts = np.unique(themes)
    return prompts
