from model import Inception_V3
from dataset import MyDataModule
import pytorch_lightning as pl

if __name__ == "__main__":
    model = Inception_V3()
    # 实例化DataModule类
    data = MyDataModule(batch_size=32)

    # 定义训练器
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=100, check_val_every_n_epoch=5,
                         auto_lr_find=True, auto_scale_batch_size=True)

    trainer.fit(model, data)