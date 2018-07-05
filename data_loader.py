# coding: utf-8
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class COS_DES(Dataset):
    def __init__(self, data, opt, transform=None):
        super(COS_DES, self).__init__()
        self.data = data
        self.opt = opt
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_path, label = self.data[index]
        label = np.array(label)
        label = np.eye(self.opt.NUM_CLASSES)[label]
        img = np.array(Image.open(data_path))
        if img.shape[2] == 1:
            img = np.append(img, img, 0)
            img = np.append(img, img, 0)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        if self.transform:
            sample = self.transform(img)
        else:
            sample = img
        return sample, label.astype(np.float32)


class DouDat(Dataset):
    def __init__(self, data, opt, transform=None):
        super(DouDat, self).__init__()
        self.data = data
        self.opt = opt
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = []
        cos_path, ani_path, label = self.data[index]
        label = np.array(label)
        label = np.eye(self.opt.NUM_CLASSES)[label]
        cos_img = np.array(Image.open(cos_path))
        ani_img = np.array(Image.open(ani_path))

        if cos_img.shape[2] == 1:
            cos_img = np.append(cos_img, cos_img, 0)
            cos_img = np.append(cos_img, cos_img, 0)
        elif cos_img.shape[2] > 3:
            cos_img = cos_img[:, :, :3]

        if ani_img.shape[2] == 1:
            ani_img = np.append(ani_img, ani_img, 0)
            ani_img = np.append(ani_img, ani_img, 0)
        elif ani_img.shape[2] > 3:
            ani_img = cos_img[:, :, :3]

        if self.transform:
            sample.append(self.transform(cos_img))
            sample.append(self.transform(ani_img))
        else:
            sample.append(cos_img)
            sample.append(ani_img)
        return sample, label.astype(np.float32)