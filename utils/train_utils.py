import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils.config import config
from torch.utils.tensorboard import SummaryWriter
import os


def train_eval(net, train_loader, test_loader, epochs=1, lr=0.001, train_type='sound', offset=0):
    writer = SummaryWriter(os.path.join("outputs", "logdir", train_type))
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    net.train()

    for epoch in range(offset, offset+epochs):
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
                cond_target = cond_target.view(N).long().to(config['device'])
                noise_target = noise_target.view(N, -1).to(config['device'])

            optimizer.zero_grad()
            outputs = net(data)
            if train_type == 'translator':
                criterion = torch.nn.CrossEntropyLoss(reduction="sum")
                output_cond = outputs
                loss = criterion(output_cond, cond_target)
            else:
                loss = torch.sum((outputs - targets) ** 2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_count += data.size(0)

            writer.add_scalar("train/loss", total_loss /
                              total_count, global_step=epoch)

        if epoch % 5 == 0:
            test_loss, test_acc = eval_network(
                net, test_loader, train_type=train_type)
            writer.add_scalar("test/loss", test_loss, global_step=epoch)
            writer.add_scalar("test/acc", test_acc, global_step=epoch)

        scheduler.step(total_loss/total_count)
        print("Epoch {} loss = {}".format(epoch, total_loss / total_count))
    writer.close()


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
            N = data.shape[0]

            cond_target, noise_target = targets
            cond_target = cond_target.view(N).long().to(config['device'])
            noise_target = noise_target.view(N, -1).to(config['device'])

            targets = cond_target

        outputs = net(data)

        if train_type == 'translator':
            criterion = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = criterion(outputs, cond_target).detach().cpu().numpy()
            outputs = outputs.argmax(dim=1)

        predictions.append(outputs.detach().cpu().numpy())
        eval_targets.append(targets.detach().cpu().numpy())

    predictions = np.concatenate(predictions)
    eval_targets = np.concatenate(eval_targets)

    if train_type == 'translator':
        acc = np.sum(predictions == eval_targets) / len(predictions)
    else:
        loss = np.mean(np.sum((predictions - eval_targets) ** 2, axis=1))

        acc = np.sum(np.bitwise_and((predictions[:, 0] * eval_targets[:, 0])
                     > 0,  predictions[:, 1] * eval_targets[:, 1] > 0)) / len(predictions)

    print("Loss = {} Accuracy = {}".format(loss, 100 * acc))
    return loss, acc


def get_features(model, loader):
    model.eval()
    features = []

    for data, labels in loader:
        data = data.to(config['device'])
        cur_feats = model(data)
        features.append(cur_feats.cpu().detach().numpy())

    features = np.concatenate(features)

    return features


def train_model(net, train_loader, epochs=1, lr=0.001, train_type='sound', offset=0):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    net.train()

    for epoch in range(offset, offset+epochs):
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
                cond_target = cond_target.view(N).long().to(config['device'])
                noise_target = noise_target.view(N, -1).to(config['device'])

            optimizer.zero_grad()
            outputs = net(data)
            if train_type == 'translator':
                criterion = torch.nn.CrossEntropyLoss()
                output_cond = outputs
                loss = criterion(output_cond, cond_target)
            else:
                loss = torch.sum((outputs - targets) ** 2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_count += data.size(0)
        scheduler.step(total_loss/total_count)
        print("Epoch {} loss = {}".format(epoch, total_loss / total_count))
