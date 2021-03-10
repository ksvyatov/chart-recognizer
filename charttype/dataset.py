from torch.utils.data import Dataset
import cv2

class ChartsDataset(Dataset):
    classes = {
        "bar_chart": 1,
        "diagram": 2,
        "flow_chart": 3,
        "graph": 4,
        "growth_chart": 5,
        "pie_chart": 6,
        "table": 7
    }

    def __init__(self, path, img_list, transform=None, mode='train'):
        self.path = path
        self.img_list = img_list
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_name = self.img_list[idx]

        fname = image_name
        if image_name.split(".")[-1] == "gif":
            gif = cv2.VideoCapture(fname)
            _, image = gif.read()
        else:
            image = cv2.imread(fname)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Distribution of pictures into categories
        label = 0
        for key, value in self.classes.items():
            if key in image_name:
                label = value
                break

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.mode == "train":
            return image, label
        else:
            return image, image_name

    @staticmethod
    def get_class_name(label):
        for key, value in ChartsDataset.classes.items():
            if value == int(label):
                return key
        return None

    @staticmethod
    def get_class_names(labels):
        return [ChartsDataset.get_class_name(label) for label in labels]
