from utils.models import get_pretrained_mobile_net
from utils.loaders.oasis_loaders import get_oasis_dataset_loaders
from utils.train_utils import train_eval, train_model, eval_network
import torch
from utils.config import config


def train_and_evaluate():
    net = get_pretrained_mobile_net(pretrained=True)
    net.to(config['device'])
    train_loader, test_loader = get_oasis_dataset_loaders()

    train_model(net, train_loader, epochs=5, lr=0.0001, train_type='image')
    eval_network(net, test_loader, train_type='image')


def train_deploy():
    train_loader, test_loader = get_oasis_dataset_loaders()

    net = get_pretrained_mobile_net(pretrained=True)
    net.to(config['device'])

    train_eval(net, train_loader, test_loader,
               epochs=50, lr=0.0001, train_type='image')
    torch.save(net.state_dict(), "models/image_sentiment.model")

    model = get_pretrained_mobile_net(pretrained=True)
    model.to(config['device'])
    model.load_state_dict(torch.load("models/image_sentiment.model"))
    eval_network(model, test_loader, train_type='image')


if __name__ == '__main__':
    train_deploy()
