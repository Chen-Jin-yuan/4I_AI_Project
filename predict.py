import argparse
import csv
import os
from typing import List

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm

from model import Inception_V3
from prepare import load_transform


class PredictSet(Dataset):
    root_dir: str
    filelist: List[str]
    transform: callable

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.filelist = os.listdir(root_dir)
        self.transform = load_transform()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item):
        img = read_image(os.path.join(self.root_dir, self.filelist[item]))

        img = img.float() / 255  # rescale to [0, 1]
        img = img.expand(3, *img.shape[1:])  # expand to 3 channels

        img = self.transform(img)  # resize & normalize

        return self.filelist[item], img


transform = None


def load_image(fp: str) -> Tensor:
    global transform
    if transform is None:
        transform = load_transform()

    img = read_image(fp)

    img = img.float() / 255  # rescale to [0, 1]
    img = img.expand(3, *img.shape[1:])  # expand to 3 channels

    img = transform(img)  # resize & normalize

    return img


def predict(ckpt: str, root_dir: str, save_csv: str, batch_size: int = 16):
    model = Inception_V3.load_from_checkpoint(ckpt)
    model.eval()

    filelist = os.listdir(root_dir)

    with open(save_csv, mode="w", newline="") as f:
        wt = csv.writer(f)
        wt.writerow(["case", "class", "P0", "P1", "P2"])
        for batch in tqdm(range(0, len(filelist), batch_size)):
            batch_filelist = filelist[batch:batch + batch_size]
            batch_imgs = [load_image(os.path.join(root_dir, fp)) for fp in batch_filelist]
            batch_imgs = torch.stack(batch_imgs)
            batch_imgs = batch_imgs.to(model.device)

            with torch.no_grad():
                y_hat = model(batch_imgs)
                probs = torch.softmax(y_hat, dim=1)
            preds = torch.argmax(y_hat, dim=1)

            for fp, p, prob in zip(batch_filelist, preds, probs):
                wt.writerow([fp, p.item(), *prob.tolist()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("root_dir", type=str, nargs="?", default="data/1. Original Images/b. Testing Set",
                        help="folder containing images")
    parser.add_argument("save_csv", type=str, nargs="?", default="data/prediction.csv", help="csv file to save")
    parser.add_argument("--batch_size", type=int, nargs="?", default=32, help="batch size")
    args = parser.parse_args()
    predict(args.ckpt, args.root_dir, args.save_csv, args.batch_size)


if __name__ == "__main__":
    main()
