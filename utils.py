from torch.utils.data import Dataset
import os
from PIL import Image

class CustomMNIST_Dataset(Dataset):
    def __init__(self, root='poisoned_data', train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        folder = 'train' if train else 'test'
        self.data_dir = os.path.join(root, folder)

        self.images = []
        self.labels = []

        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith('.png'):
                self.images.append(os.path.join(self.data_dir, filename))
                label = int(filename.split('_label')[1].split('.')[0])
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
