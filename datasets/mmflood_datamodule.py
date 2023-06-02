import os
import torchvision.transforms as transforms

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets.mmflood_dataset import MMFloodDataset, TemporalMMFloodDataset


class MMFloodDataModule(LightningDataModule):

    def __init__(self, data_dir, batch_size=32, num_workers=16, height=1497, width=1385, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.height = height
        self.width = width
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.setup()

    def setup(self, stage=None):
        transform = transforms.Compose([
                    transforms.Resize((self.height, self.width)),
                    transforms.ToTensor(),
                ])
        self.train_dataset = MMFloodDataset(
                root=os.path.join(self.data_dir, 'train'),
                transform=None
        )
        self.val_dataset = MMFloodDataset(
                root=os.path.join(self.data_dir, 'val'),
                transform=None
        )
        self.test_dataset = MMFloodDataset(
                root=os.path.join(self.data_dir, 'test'),
                transform=None
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

class TemporalMMFloodDataModule(LightningDataModule):

    def __init__(self, data_dir, batch_size=32, num_workers=16, height=1497, width=1385, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.height = height
        self.width = width
        self.seed = seed

        self.setup()

    def setup(self, stage=None):
        transform = transforms.Compose([
                    transforms.Resize((self.height, self.width)),
                    transforms.ToTensor(),
                ])
        self.dataset = TemporalMMFloodDataset(
                root=self.data_dir,
                transform=None
        )

    def dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )