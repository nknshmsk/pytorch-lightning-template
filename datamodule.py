import json
from pathlib import Path
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl


class image_data_set_by_json(Dataset):
    def __init__(self, json_path, role, transform=None):
        super().__init__()
        self.__dict__.update(locals())

        with open(json_path) as f:
            df = json.load(f)
        self.data = df[self.role]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]["path"])
        path = Path(self.data[idx]["path"])
        return self.transform(image), self.data[idx]["label"], f"{path.parents[0].name}_{path.stem}"


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, transform, batch_size=128):
        super().__init__()
        self.__dict__.update(locals())

    def setup(self, stage):
        self.mnist_train = MNIST("MNIST", train=True, download=True, transform=self.transform)
        self.mnist_test = MNIST("MNIST", train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class MVTecDataModule(pl.LightningDataModule):
    def __init__(self, json_path, train_batch_size=128, validation_batch_size=1, num_workers=12, input_image_size=128):
        super().__init__()
        self.__dict__.update(locals())
        self.transform = transforms.Compose([
            transforms.Resize((self.input_image_size, self.input_image_size)),
            transforms.ToTensor()
        ])

    def setup(self, stage):
        self.mvtec_train = image_data_set_by_json(self.json_path, role="train", transform=self.transform)
        self.mvtec_test = image_data_set_by_json(self.json_path, role="test", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mvtec_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.mvtec_test,
            batch_size=self.validation_batch_size,
            num_workers=self.num_workers)
