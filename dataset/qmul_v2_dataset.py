from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from glob import glob
import random


class TrainDataset(Dataset):
    def __init__(self, ph_root, ph_txt, sk_root):
        self.ph_root = ph_root
        self.sk_root = sk_root
        with open(ph_txt) as f:
            self.ph_names = f.read().splitlines()
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose(
            [
                transforms.Resize(288),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                norm,
            ]
        )

    def __getitem__(self, index: int):
        ph_name = self.ph_names[index]
        ph_label = ph_name.split(".")[0]
        ph_path = os.path.join(self.ph_root, ph_name)
        ph = Image.open(ph_path)
        ph = self.transform(ph)

        sk_regex = os.path.join(self.sk_root, ph_label + "_*")
        sk_paths = glob(sk_regex)
        random.shuffle(sk_paths)
        sk_path = sk_paths[0]
        sk = Image.open(sk_path)
        sk = sk.split()[3]
        sk = Image.merge("RGB", (sk, sk, sk))
        sk = self.transform(sk)

        return ph, sk

    def __len__(self):
        return len(self.ph_names)


class TestPhDataset(Dataset):
    def __init__(self, ph_root, ph_txt):
        self.ph_root = ph_root
        with open(ph_txt) as f:
            self.ph_names = f.read().splitlines()
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose(
            [transforms.Resize(256), transforms.ToTensor(), norm]
        )

    def __getitem__(self, index: int):
        ph_name = self.ph_names[index]
        ph_label = ph_name.split(".")[0]
        # ph_label=ph_name
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
            [transforms.Resize(256), transforms.ToTensor(), norm]
        )

    def __getitem__(self, index: int):
        sk_name = self.sk_names[index]
        sk_label = sk_name.split(".")[0]
        sk_path = os.path.join(self.sk_root, sk_name)
        sk = Image.open(sk_path)
        sk = sk.split()[3]
        sk = Image.merge("RGB", (sk, sk, sk))
        sk = self.transform(sk)

        return sk, sk_label

    def __len__(self):
        return len(self.sk_names)
