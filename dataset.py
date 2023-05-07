import csv
from typing import Optional, Dict, List, Tuple

import numpy as np
import numpy.lib.format as npfmt
import torch
from pytorch_lightning.core import LightningDataModule
from torch.utils.data import Dataset, DataLoader


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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        # 创建验证集的DataLoader
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        # 创建测试集的DataLoader
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)


def load_labels(csv_fp: str) -> List[Tuple[str, int]]:
    """
    Load labels from csv file

    :param csv_fp: csv label file path
    :return: dict of labels
    """
    with open(csv_fp, mode="r") as f:
        reader = csv.reader(f)
        _ = next(reader)  # skip header
        # Note that file extension is removed
        return [(row[0], int(row[1])) for row in reader]


class CSVMMAPDataset(Dataset):
    """
    MMAP accessed numpy dataset with csv label file
    """
    data: np.memmap
    labels: List[Tuple[str, int]]
    total_len: int

    def __init__(self, csv_fp: str, npy_fp: str):
        self.labels = load_labels(csv_fp)
        self.total_len = len(self.labels)
        self.data = npfmt.open_memmap(npy_fp, dtype=np.float32, mode="c", shape=(self.total_len, 3, 299, 299))

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        return self.data[item], self.labels[item][1]


class CSVDataModule(LightningDataModule):
    """DataModule on CSV labelled npy dataset"""
    train_dataset: Optional[CSVMMAPDataset]
    val_dataset: Optional[CSVMMAPDataset]
    train_csv: str
    val_csv: str
    train_npy: str
    val_npy: str
    batch_size: int
    seed: int
    label_count: Dict[int, int]
    train: Optional[Dataset]
    val: Optional[Dataset]
    test: Optional[Dataset]

    def __init__(self, train_csv: str = "data/argument.csv", train_npy: str = "data/argument.npy",
                 val_csv: str = "data/val_split.csv", val_npy: str = "data/val_split.npy",
                 batch_size: int = 16,
                 seed: int = 42):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.save_hyperparameters()

        self.train_csv, self.train_npy = train_csv, train_npy
        self.val_csv, self.val_npy = val_csv, val_npy

        self.train_dataset, self.val_dataset = None, None

    def setup(self, stage: str) -> None:
        if self.train_dataset is not None:
            return

        self.train_dataset = CSVMMAPDataset(self.train_csv, self.train_npy)
        self.train_dataset.label_count = label_count(self.train_dataset.labels)
        self.val_dataset = CSVMMAPDataset(self.val_csv, self.val_npy)
        self.val_dataset.label_count = label_count(self.val_dataset.labels)

    def train_dataloader(self):
        # REMARK better not to use parallel data loading to avoid high memory usage
        # Also not of much use since data is mmaped
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


def label_count(labels: List[Tuple[str, int]]) -> Dict[int, int]:
    """
    Get counts of each category in the dataset

    :param labels: dict of labels
    :return: dict of category counts
    """
    d = dict()
    for _, cat in labels:
        d[cat] = d.get(cat, 0) + 1
    return d
