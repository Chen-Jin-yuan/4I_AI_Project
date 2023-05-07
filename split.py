import argparse
import csv
from random import Random

from dataset import load_labels


def split(src_csv: str, train_csv: str, val_csv: str, ratio: float = 0.8, seed: int = 42):
    """Randomly split src dataset into train and val set"""
    labels = load_labels(src_csv)

    rand = Random(seed)
    rand.shuffle(labels)

    train_len = int(len(labels) * ratio)
    train_items = labels[:train_len]
    val_items = labels[train_len:]

    for set_fp, set_items in [(train_csv, train_items), (val_csv, val_items)]:
        with open(set_fp, "w") as f:
            wt = csv.writer(f)
            wt.writerow(["image name", "image quality level"])
            for k, v in set_items:
                wt.writerow([k, v])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_csv", type=str, nargs="?",
                        default="data/2. Groundtruths/a. DRAC2022_ Image Quality Assessment_Training Labels.csv",
                        help="csv file containing labels")
    parser.add_argument("train_csv", type=str, nargs="?", default="data/train_split.csv", help="train csv file")
    parser.add_argument("val_csv", type=str, nargs="?", default="data/val_split.csv", help="val csv file")
    args = parser.parse_args()
    split(args.src_csv, args.train_csv, args.val_csv)


if __name__ == '__main__':
    main()
