import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core import LightningDataModule

class MyDataset(Dataset):
    def __init__(self, size ):
        self.size = size
        self.data = [torch.randn(3, 299, 299) for i in range(size)]
        self.labels = [torch.randint(0, 3, (1,))[0] for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class MyDataModule(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        # 加载训练集、验证集、测试集
        self.train_dataset = MyDataset(2000)
        self.val_dataset = MyDataset(500)
        self.test_dataset = MyDataset(500)

    def train_dataloader(self):
        # 创建训练集的DataLoader
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers = 8)

    def val_dataloader(self):
        # 创建验证集的DataLoader
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = 8)

    def test_dataloader(self):
        # 创建测试集的DataLoader
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = 8)


