import torch
import os
import chartmodel
from torch.utils.data import Dataset
import albumentations
from albumentations.pytorch import ToTensorV2 as AT

from charttype import dataset

batch_size = 32
num_workers = 4

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    img_size = 256
    data_transforms_test = albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.Normalize(),
        AT()
    ])

    test_list = list()
    test_path = os.path.dirname(os.path.abspath(__file__)) + "../data/chart_type/test/images/"
    for pic in os.listdir(test_path):
        test_list.append(test_path + pic)
    testset = dataset.ChartsDataset('/', test_list, transform=data_transforms_test, mode="test")
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers)

    model = chartmodel.get_model()
    model.load_state_dict(torch.load('../data/chart_type/model.pt', map_location=device))
    model.eval()
    for img, img_filename in testloader:
        with torch.no_grad():
            img = img.to(device)
            output = model(img)
            pred = torch.argmax(output, dim=1).cpu().numpy()
            types = dataset.ChartsDataset.get_class_names(pred)
            for i in range(len(img_filename)):
                print(f'filename: {os.path.basename(img_filename[i])}; type: {types[i]}; label: {pred[i]}')
