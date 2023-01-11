import torch.optim as optim
import torch
import numpy as np
import pickle
import torch.nn.functional as F
from tqdm import tqdm
from utils.config import config


def train_model(net, train_loader, epochs=1, lr=0.001, train_type='sound'):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()

    for epoch in range(epochs):
        total_count = 0.0
        total_loss = 0.0

        for data, targets in train_loader:
            if train_type == 'sound':
                data = data.view((-1, data.size(2))).to(config['device'])
                targets = targets.view(
                    (-1, targets.size(2))).to(config['device'])
            elif train_type == 'image':
                data = data.to(config['device'])
                targets = targets.to(config['device'])
                targets = targets.view(-1, 2)
            elif train_type == 'translator':
                data = data.to(config['device'])
                N = data.shape[0]

                cond_target, noise_target = targets
                cond_target = cond_target.view(N, -1).to(config['device'])
                noise_target = noise_target.view(N, -1).to(config['device'])

            optimizer.zero_grad()
            outputs = net(data)
            if train_type == 'translator':
                output_cond, output_noise = outputs
                loss = torch.sum((noise_target - output_noise) ** 2) + \
                    torch.sum((cond_target - output_cond) ** 2)
            else:
                loss = torch.sum((outputs - targets) ** 2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_count += data.size(0)
        print("Epoch loss = ", total_loss / total_count)


def eval_network(net, test_loader,  train_type='sound'):
    predictions = []
    eval_targets = []
    net.eval()

    for data, targets in test_loader:
        if train_type == 'sound':
            data = data.view((-1, data.size(2))).to(config['device'])
            targets = targets.view((-1, targets.size(2))).to(config['device'])
        elif train_type == 'image':
            data = data.to(config['device'])
            targets = targets.to(config['device'])
            targets = targets.view(-1, 2)
        elif train_type == 'translator':
            data = data.to(config['device'])
            cond_target, noise_target = targets
            cond_target = cond_target.view(-1,
                                           config['cond_size']).detach().cpu().numpy()
            noise_target = noise_target.view(-1,
                                             config['noise_size']).detach().cpu().numpy()

            targets = cond_target

        outputs = net(data)

        if train_type == 'translator':
            cond_output, noise_output = outputs
            cond_output = cond_output.view(-1,
                                           config['cond_size']).detach().cpu().numpy()
            noise_output = noise_output.view(-1,
                                             config['noise_size']).detach().cpu().numpy()

            outputs = cond_output
        else:
            predictions.append(outputs.detach().cpu().numpy())
            eval_targets.append(targets.detach().cpu().numpy())

    if train_type != 'translator':
        predictions = np.concatenate(predictions)
        eval_targets = np.concatenate(eval_targets)

    if train_type == 'translator':
        mse_noise = np.mean(np.sum((noise_output - noise_target) ** 2, axis=1))
        mse_cond = np.mean(np.sum((cond_output - cond_target) ** 2, axis=1))

        print(mse_noise, mse_cond)

        return mse_noise, mse_cond
    else:

        mse_arousal = np.mean((predictions[:, 0] - eval_targets[:, 0]) ** 2)
        mse_valence = np.mean((predictions[:, 1] - eval_targets[:, 1]) ** 2)

        arousal_acc = np.sum(
            (predictions[:, 0] * eval_targets[:, 0]) > 0) / len(predictions)
        valence_acc = np.sum(
            (predictions[:, 1] * eval_targets[:, 1]) > 0) / len(predictions)

        print(mse_arousal, mse_valence, 100 * arousal_acc, 100 * valence_acc)

        return mse_arousal, mse_valence, arousal_acc, valence_acc


def get_features(model, loader):
    model.eval()
    features = []

    for data, labels in loader:
        data = data.to(config['device'])
        cur_feats = model(data)
        features.append(cur_feats.cpu().detach().numpy())

    features = np.concatenate(features)

    return features
