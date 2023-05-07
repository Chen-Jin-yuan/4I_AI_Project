import argparse
import csv
import multiprocessing as mp
import os
from functools import partial
from itertools import chain, repeat
from typing import Iterable, Tuple, List, Optional

from torchvision import transforms
from torchvision.io import read_image, write_jpeg
from tqdm import tqdm

from dataset import load_labels


def argument_transform():
    return transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Grayscale()  # must be grayscale
    ])


def fn(transform, src_root: str, dst_root: str, t):
    """
    Auxiliary function for argument.
    This function is the loop body of the parallel process.

    :param transform: transform function
    :param src_root: source folder
    :param dst_root: destination folder
    :param t: tuple of (idx, ((img_fp, label), apply))
    :return: tuple of (image name, image quality level)
    """
    # idx: image id, img_fp: image path, label: image label, apply: whether to apply transform
    idx, ((img_fp, label), apply) = t

    img = read_image(os.path.join(src_root, img_fp))
    img = transform(img) if apply else img
    write_jpeg(img, os.path.join(dst_root, f"{idx}.jpg"))

    return [f"{idx}.jpg", label]  # to be written to csv


def argument(src_csv_fp: str, src_root: str, dst_csv_fp: str, dst_root: str, factors: Optional[List[int]] = None):
    """
    Perform argument on images in src_root and save to dst_root.

    :param src_csv_fp: source csv file path
    :param src_root: source folder
    :param dst_csv_fp: destination csv file path
    :param dst_root: destination folder
    :param factors: data augmentation factor
    """
    if factors is None:
        factors = [10, 5, 1]

    transform = argument_transform()

    # Create dst_root if not exists
    if not os.path.exists(dst_root):
        os.mkdir(dst_root)

    labels = load_labels(src_csv_fp)  # first load labels from csv file
    with open(dst_csv_fp, mode="w", newline="") as f2:
        writer = csv.writer(f2)
        writer.writerow(["image name", "image quality level"])

        # Repeat each row in reader `factor` times.
        arg_rd: Iterable[Tuple[str, str]] = chain.from_iterable(
            zip(repeat((x, label), factors[label]),  # file index
                # Whether to apply transform: we ensure there's at least one original image in the final dataset
                [False] + [True] * (factors[label] - 1))
            for x, label in labels)

        with mp.Pool() as pool:
            f_ = partial(fn, transform, src_root, dst_root)
            for row in tqdm(pool.imap_unordered(f_, enumerate(arg_rd))):
                writer.writerow(row)  # write label to csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_csv_fp", type=str, nargs="?", default="data/train_split.csv",
                        help="csv file containing image file path and label")
    parser.add_argument("src_root", type=str, nargs="?", default="data/1. Original Images/a. Training Set",
                        help="folder containing images")
    parser.add_argument("dst_csv_fp", type=str, nargs="?", default="data/argument.csv", help="csv file to save")
    parser.add_argument("dst_root", type=str, nargs="?", default="data/argument", help="folder to save images")
    parser.add_argument("--factors", type=lambda s: [int(item) for item in s.split(',')], default="10,5,1",
                        help="argumentation factors")
    args = parser.parse_args()
    argument(args.src_csv_fp, args.src_root, args.dst_csv_fp, args.dst_root, args.factors)


if __name__ == '__main__':
    main()
