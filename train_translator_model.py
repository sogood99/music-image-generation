import torch
from utils.models import NeuralTranslator
from utils.loaders.neural_translator_loader import create_collection, get_train_loaders
from utils.train_utils import train_eval, train_model, eval_network
from tqdm import tqdm
from utils.config import config


def create_dataset(epochs=100):
    """
    Sample latent space, get inverse
    """
    create_collection(epochs=epochs)


def train_and_evaluate():
    net = NeuralTranslator()
    net.to(config['device'])
    train_loader, test_loader = get_train_loaders()

    train_model(net, train_loader, epochs=200,
                lr=0.001, train_type='translator')
    eval_network(net, train_loader, train_type='translator')
    eval_network(net, test_loader, train_type='translator')


def train_deploy(seed=1):
    train_loader, test_loader = get_train_loaders()

    net = NeuralTranslator()
    net.to(config['device'])

    torch.manual_seed(seed)
    train_eval(net, train_loader, test_loader, epochs=500,
               lr=0.001, train_type='translator')
    torch.save(net.state_dict(), "models/neural_translator_" +
               str(seed) + ".model")

    net = NeuralTranslator()
    net.to(config['device'])
    net.load_state_dict(torch.load(
        "models/neural_translator_" + str(seed) + ".model"))
    eval_network(net, train_loader, train_type='translator')


if __name__ == '__main__':
    train_deploy(seed=0)
