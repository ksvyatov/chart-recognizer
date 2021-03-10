import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
import albumentations
from albumentations.pytorch import ToTensorV2 as AT
import dataset
import chartmodel
import matplotlib.pyplot as plt

if __name__ == '__main__':

    PATH = '../data/chart_type/yandex/'
    train_path = list()
    for directory in os.listdir(PATH):
        train_path.append(os.path.join(PATH, directory))

    test_path = ("../data/chart_type/test/")

    train_list = list()
    for directory in train_path:
        for pic in os.listdir(directory):
            train_list.append(directory + '/' + pic)

    test_list = list()
    for pic in os.listdir(test_path):
        test_list.append(test_path + pic)
    print(len(train_list), len(test_list))


    batch_size = 64
    num_workers = os.cpu_count()
    img_size = 256

    data_transforms = albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.CLAHE(),
        albumentations.ChannelShuffle(),
        albumentations.Downscale(),
        albumentations.Cutout(),
        albumentations.ShiftScaleRotate(),
        albumentations.Normalize(),
        AT()
    ])

    data_transforms_test = albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.Normalize(),
        AT()
    ])

    trainset = dataset.ChartsDataset('/', train_list, transform=data_transforms)
    testset = dataset.ChartsDataset('/', test_list, transform=data_transforms_test, mode="test")
    valid_size = int(len(train_list) * 0.1)
    train_set, valid_set = torch.utils.data.random_split(trainset, (len(train_list) - valid_size, valid_size))

    trainloader = torch.utils.data.DataLoader(train_set, pin_memory=True,
                                              batch_size=batch_size, shuffle=True,  # Remember to shuffle data
                                              num_workers=num_workers)

    validloader = torch.utils.data.DataLoader(valid_set, pin_memory=True,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Translate learning to the GPU for faster learning
    print(device)
    model = chartmodel.get_model()
    model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0007)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3, )

    model_resnet, train_loss, val_loss, lst_valid, lst_train, lst_epochs = chartmodel.train_model(model, trainloader, validloader, criterion,
                                                                                                            optimizer, scheduler, device, n_epochs=80)
    fig, ax = plt.subplots()
    ax.plot(lst_epochs, lst_train, lst_epochs, lst_valid)

    ax.set(xlabel='epoch', ylabel='loss',
           title='learning curve')
    ax.grid()

    fig.savefig("../data/chart_type/train_valid_loss.png")
    plt.show()
