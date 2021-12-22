import os
from PIL import Image
import random

from abc import abstractmethod
from torch.utils.data import Dataset

class UDA(Dataset):
    def __init__(self, x, dir_path, transform):
        super(UDA, self).__init__()
        self.x = x
        self.dir_path = dir_path
        self.transform = transform
    
    def __len__(self):
        return len(self.x)

    @abstractmethod
    def __getitem__(self, index):
        pass

    def transform_data(self, path):
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img

class SuperviedData(UDA):
    def __init__(self, x, y, dir_path, transform):
        super(SuperviedData, self).__init__(x, dir_path, transform)
        self.y = y
    
    def __getitem__(self, index):
        return self.transform_data(os.path.join(self.dir_path, self.x[index])), self.y[index]

class UnsuperviedData(UDA):
    def __init__(self, x, dir_path, transform, augmenter):
        '''
            x: list files names
        '''
        super(UnsuperviedData, self).__init__(x, dir_path, transform)
        self.augmenter = augmenter

    def __getitem__(self, index):
        unlabel_path = os.path.join(self.dir_path, self.x[index])
        unlabel_data = self.transform_data(unlabel_path)

        aug_img = Image.open(unlabel_path).convert('RGB')
        aug_img = self.augmenter(aug_img)
        aug_data = self.transform_data(aug_img)

        return unlabel_data, aug_data
    


