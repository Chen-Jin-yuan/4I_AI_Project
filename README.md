# 环境

* python3.10
* pytorch2.0
* pytorch-lightning2.0.0以上
* cuda 11.6以上

# 数据集

* 新建一个`data`文件夹，将数据集保存到该文件夹中，如下：

```
├─data
│  ├─1. Original Images
│  │  ├─a. Training Set
│  │  └─b. Testing Set
│  └─2. Groundtruths
```

* 运行`argument_and_prepare.sh`脚本，会进行数据增强并产生最终数据集。若您无法使用`bash`，可以按顺序执行以下命令来准备数据：
  * 1.python split.py
  * 2.python argument.py
  * 3.python prepare.py
  * 4.python prepare.py data/val_split.csv data/1.\ Original\ Images/a.\ Training\ Set data/val_split.npy

# 训练

我们使用`pytorch-lightning`来进行训练。

1. 配置超参数文件`config.example.yaml`；
2. 执行`python train.py fit --config config.example.yaml`。

模型权重和损失准确率等信息会保存在`lightning_logs`文件中。

# 推理

执行以下命令：

* `python predict.py your_checkpoints_path`
* `your_checkpoints_path`是指保存的模型权重文件。

产生的输出文件在`data`文件夹中，名称为`prediction.csv`。
