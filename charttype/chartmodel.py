import torch
import torch.nn as nn
import torchvision
import numpy as np
import time
from tqdm import tqdm


def get_model():
    model = torchvision.models.resnet152(pretrained=True, progress=True)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(in_features, 1024), nn.Linear(1024,8))
    return model

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, n_epochs):
    model.to(device)
    valid_loss_min = np.Inf
    patience = 10
    p = 0

    lst_loss_valid = []
    lst_loss_train = []
    lst_epochs = []

    # количество эпох
    for epoch in range(1, n_epochs + 1):
        print(time.ctime(), 'Epoch:', epoch)
        train_loss = []
        for batch_i, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        # Run validation
        model.eval()
        val_loss = []
        for batch_i, (data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss.append(loss.item())

        loss_valid = np.mean(val_loss)
        loss_train = np.mean(train_loss)

        lst_loss_valid.append(loss_valid)
        lst_loss_train.append(loss_train)
        lst_epochs.append(epoch)

        print(f'Epoch {epoch}, train loss: {loss_train:.4f}, valid loss: {loss_valid:.4f}.')


        scheduler.step(loss_valid)
        if loss_valid <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                loss_valid))
            torch.save(model.state_dict(), '../data/chart_type/model.pt')
            valid_loss_min = loss_valid
            p = 0

        # Checking data
        if loss_valid > valid_loss_min:
            p += 1
            print(f'{p} epochs of increasing val loss')
            if p > patience:
                print('Stopping training')
                stop = True
                break

    return model, train_loss, val_loss, lst_loss_valid, lst_loss_train, lst_epochs

