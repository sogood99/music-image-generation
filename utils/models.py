import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNetV2
import pickle
from utils.config import config
from os.path import join


class SoundMLP(nn.Module):
    def __init__(self, input_size=9156):
        super(SoundMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 2)

        self.mean = 0
        self.std = 1

    def forward(self, x):
        x = F.dropout(x, training=self.training)
        out = self.fc1(x)
        out = torch.sigmoid(out)

        out = self.fc2(out)
        return out

    def get_np_features(self, x):
        x = self.forward(x)
        x = x.cpu().detach().numpy()
        x = (x - self.mean) / self.std

        return x

    def load_neural_space_statistics(self, path=join(config['basedir'], "models/space_statistics.pickle")):
        with open(path, "rb") as f:
            self.mean = pickle.load(f)
            self.std = pickle.load(f)

    def load_neural_model(self, path=join(config['basedir'], "models/mlp.model")):
        self.load_state_dict(torch.load(
            path, map_location=torch.device(config['device'])))

    def save_neural_space_statistics(self, path=join(config['basedir'], "models/space_statistics.pickle")):
        with open(path, "wb") as f:
            pickle.dump(self.mean, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.std, f, protocol=pickle.HIGHEST_PROTOCOL)


class NeuralTranslator(nn.Module):

    def __init__(self, input_size=2):
        super(NeuralTranslator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4_cond = nn.Linear(1024, 248)
        # self.fc4_noise = nn.Linear(4096, 16384)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        selected_cond = self.fc4_cond(out)
        # selected_noise = self.fc4_noise(out)
        # return (selected_cond, selected_noise)
        return selected_cond


def get_pretrained_mobile_net(pretrained=True):
    if pretrained:
        pretrained_net = mobilenet_v2(weights='DEFAULT')

    net = MobileNetV2(num_classes=2)
    if pretrained:
        state_dict = pretrained_net.state_dict()
        del state_dict['classifier.1.weight']
        del state_dict['classifier.1.bias']
        net.load_state_dict(state_dict, strict=False)

    return net


class CorrectionNet(nn.Module):
    def __init__(self):
        super(CorrectionNet, self).__init__()
        self.fc1 = nn.Linear(59136, 1024)
        self.fc2 = nn.Linear(59136, 1024)

        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 59136)

    def forward(self, x0, x1):
        x0 = F.relu(self.fc1(x0))
        x1 = F.relu(self.fc1(x1))
        x = torch.cat((x0, x1), dim=1)

        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
