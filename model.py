import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torchmetrics import Accuracy, CohenKappa


# 定义 lightning 模型
class Inception_V3(pl.LightningModule):
    def __init__(self, data_len_list=[1, 1, 1], lr=0.001, model_path=None):
        super().__init__()
        self.num_classes = 3
        self.lr = lr

        # 定义加权交叉熵损失函数权重
        # weights : https://discuss.pytorch.org/t/what-is-the-weight-values-mean-in-torch-nn-crossentropyloss/11455/10
        # 一般两种方式：
        # 1. 使用类别数据量的倒数，可能会导致准确率下降
        # 2. 使用最大的类别尺寸除以每个类别的尺寸，maxSize / curSize
        weights = []
        maxSize = max(data_len_list)
        for curSize in data_len_list:
            weights.append(maxSize / curSize)
        class_weights = torch.FloatTensor(weights)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # 计算准确率
        self.train_acc = Accuracy(task='multiclass', num_classes=3)
        self.train_k = CohenKappa(task='multiclass', num_classes=3)
        self.val_acc = Accuracy(task='multiclass', num_classes=3)
        self.val_k = CohenKappa(task='multiclass', num_classes=3)
        self.test_acc = Accuracy(task='multiclass', num_classes=3)
        self.test_k = CohenKappa(task='multiclass', num_classes=3)

        # 加载模型
        if model_path != None:
            if os.path.exists(model_path):
                # 加载本地保存的模型
                self.model = torch.load(model_path)
            else:
                print("[error]: not find path of model!")
        else:
            # 加载预训练模型
            self.model = torchvision.models.inception_v3(pretrained = True)

            # # 如果要冻结参数
            # for param in model.parameters():
            #     param.requires_grad = False

            # 替换全连接层
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)  # 对于Inception_v3，self(x)返回一个InceptionOutputs对象，是二元组
        loss = self.loss_fn(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.train_acc(preds, y)
        self.train_k(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_k", self.train_k, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.val_acc(preds, y)
        self.val_k(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_k", self.val_k, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.test_acc(preds, y)
        self.test_k(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_k", self.test_k, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # 定义优化器
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def save_model(self, model_path):
        # 保存整个网络及参数
        torch.save(self.model, model_path)
