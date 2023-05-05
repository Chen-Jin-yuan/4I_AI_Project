import csv
from typing import Optional, Dict

import numpy as np
import torch
from pytorch_lightning.core import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split


class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [torch.randn(3, 299, 299) for i in range(size)]
        self.labels = [torch.randint(0, 3, (1,))[0] for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DummyDataModule(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        # 加载训练集、验证集、测试集
        self.train_dataset = DummyDataset(2000)
        self.val_dataset = DummyDataset(500)
        self.test_dataset = DummyDataset(500)

    def train_dataloader(self):
        # 创建训练集的DataLoader
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers = 8)

    def val_dataloader(self):
        # 创建验证集的DataLoader
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        # 创建测试集的DataLoader
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)


def load_labels(csv_fp: str) -> Dict[int, int]:
    """
    Load labels from csv file

    :param csv_fp: csv label file path
    :return: dict of labels
    """
    with open(csv_fp, mode="r") as f:
        reader = csv.reader(f)
        _ = next(reader)  # skip header
        # Note that file extension is removed
        return {int(row[0][:-4]): int(row[1]) for row in reader}


class CSVMMAPDataset(Dataset):
    """
    MMAP accessed numpy dataset with csv label file
    """
    data: np.memmap
    labels: Dict[int, int]
    total_len: int

    def __init__(self, csv_fp: str, npy_fp: str):
        self.labels = load_labels(csv_fp)
        self.total_len = len(self.labels)
        self.data = np.memmap(npy_fp, dtype=np.float32, mode="c", shape=(self.total_len, 3, 299, 299))

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


class CSVDataModule(LightningDataModule):
    """DataModule on CSV labelled npy dataset"""
    dataset: CSVMMAPDataset
    batch_size: int
    seed: int
    label_count: Dict[int, int]
    train: Optional[Dataset]
    val: Optional[Dataset]
    test: Optional[Dataset]

    def __init__(self, csv_fp: str = "data/argument.csv", npy_fp: str = "data/argument.npy", batch_size: int = 16,
                 seed: int = 42):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.save_hyperparameters()

        self.dataset = CSVMMAPDataset(csv_fp, npy_fp)
        self.label_count = label_count(self.dataset.labels)
        self.train, self.val, self.test = None, None, None

    def setup(self, stage: str) -> None:
        if self.train is not None:
            return

        train, val, test = random_split(self.dataset, [0.8, 0.1, 0.1],
                                        generator=torch.Generator().manual_seed(self.seed))
        train.label_count = self.label_count
        val.label_count = self.label_count
        test.label_count = self.label_count

        self.train, self.val, self.test = train, val, test

    def train_dataloader(self):
        # REMARK better not to use parallel data loading to avoid high memory usage
        # Also not of much use since data is mmaped
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


def label_count(labels: Dict[int, int]) -> Dict[int, int]:
    """
    Get counts of each category in the dataset

    :param labels: dict of labels
    :return: dict of category counts
    """
    d = dict()
    for _, cat in labels.items():
        d[cat] = d.get(cat, 0) + 1
    return d
