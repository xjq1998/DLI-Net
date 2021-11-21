import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


class TrainDataset(Dataset):
    def __init__(self,ph_root,sk_root,ph_txt):
        self.ph_root = ph_root
        self.sk_root = sk_root
        with open(ph_txt,'r') as f:
            self.ph_names=f.read().splitlines()
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose([
            transforms.Resize(288),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            norm
        ])

    def __getitem__(self, index: int):
        ph_name=self.ph_names[index]
        ph_path = os.path.join(self.ph_root, ph_name)
        ph = Image.open(ph_path)
        if len(ph.split())==1:
            ph=Image.merge('RGB',(ph,ph,ph))
        ph = self.transform(ph)

        sk_name=ph_name.split('.')[0]+'.png'
        sk_path = os.path.join(self.sk_root, sk_name)
        sk = Image.open(sk_path)
        sk = Image.merge('RGB', (sk, sk, sk))
        sk = self.transform(sk)

        return ph, sk

    def __len__(self):
        return len(self.ph_names)


class TestPhDataset(Dataset):
    def __init__(self,ph_root,ph_txt):
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
        if len(ph.split())==1:
            ph=Image.merge('RGB',(ph,ph,ph))
        ph = self.transform(ph)

        return ph, ph_label

    def __len__(self):
        return len(self.ph_names)


class TestSkDataset(Dataset):
    def __init__(self,sk_root,sk_txt):
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
        sk = Image.merge('RGB', (sk, sk, sk))
        sk = self.transform(sk)

        return sk, sk_label

    def __len__(self):
        return len(self.sk_names)