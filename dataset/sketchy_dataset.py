import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import os
import numpy as np
from glob import glob


class TrainDataset(Dataset):
    def __init__(self,
                 ph_root,
                 sk_root,
                 ph_name_txt,
                 dict_path):
        norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        with open(ph_name_txt, 'r') as f:
            self.ph_paths = f.read().splitlines()

        self.ph_root = ph_root
        self.sk_root = sk_root
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(), norm
        ])
        self.label_dict = np.load(dict_path, allow_pickle=True).item()

    def __getitem__(self, index):
        ph_path = self.ph_paths[index]
        ph_full_path = os.path.join(self.ph_root, ph_path)
        ph = Image.open(ph_full_path)
        ph = self.transform(ph)
        ph_class_name = ph_path.split('/')[1]
        ph_class = torch.tensor(self.label_dict[ph_class_name])

        label = ph_path.split('/')[-1].split('.')[0]
        sk_regex = os.path.join(self.sk_root, 'train')
        sk_regex = os.path.join(sk_regex, ph_class_name)
        sk_regex = os.path.join(sk_regex, label + '-*')
        sk_paths = glob(sk_regex)
        random.shuffle(sk_paths)
        sk_path = sk_paths[0]
        sk = Image.open(sk_path)
        sk = self.transform(sk)
        return ph, sk, ph_class

    def __len__(self):
        length = len(self.ph_paths)
        return length


class TestPhDataset(Dataset):
    def __init__(self, ph_root, ph_txt):
        self.ph_root = ph_root
        with open(ph_txt) as f:
            self.ph_names = f.read().splitlines()
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.ToTensor(), norm])

    def __getitem__(self, index: int):
        ph_name = self.ph_names[index]
        ph_label = ph_name.split('.')[0]
        ph_path = os.path.join(self.ph_root, ph_name)
        ph = Image.open(ph_path)
        ph = self.transform(ph)

        return ph, ph_label

    def __len__(self):
        return len(self.ph_names)


class TestSkDataset(Dataset):
    def __init__(self, sk_root, sk_txt):
        self.sk_root = sk_root
        with open(sk_txt) as f:
            self.sk_names = f.read().splitlines()
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.ToTensor(), norm])

    def __getitem__(self, index: int):
        sk_name = self.sk_names[index]
        sk_label = sk_name.split('.')[0]
        sk_path = os.path.join(self.sk_root, sk_name)
        sk = Image.open(sk_path)
        sk = self.transform(sk)

        return sk, sk_label

    def __len__(self):
        return len(self.sk_names)