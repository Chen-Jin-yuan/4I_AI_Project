import argparse
import os

import numpy as np
import numpy.lib.format as npfmt
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm


def load_transform():
    return transforms.Compose([
        transforms.Resize((299, 299), antialias=True),
        # required by pretrained model
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])


def prepare(src_root: str, npy_fp: str):
    """
    Convert all images in src_root to tensor and save to npy_fp npy (use memmap)

    :param src_root: source folder
    :param npy_fp: destination npy file path
    """
    transform = load_transform()

    if os.path.exists(npy_fp):
        os.remove(npy_fp)

    fps = list(os.listdir(src_root))
    # Create mmap file to save data
    f: np.memmap = npfmt.open_memmap(npy_fp, dtype=np.float32, mode="w+", shape=(len(fps), 3, 299, 299))
    for img_fp in tqdm(fps):
        img = read_image(os.path.join(src_root, img_fp))

        img = img.float() / 255  # rescale to [0, 1]
        img = img.expand(3, *img.shape[1:])  # expand to 3 channels

        img = transform(img)  # resize & normalize
        f[int(img_fp[:-4])] = img  # save to npy

    f.flush()  # flush to disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_root", type=str, nargs="?", default="data/argument", help="folder containing images")
    parser.add_argument("npy_fp", type=str, nargs="?", default="data/argument.npy", help="npy file to save")
    args = parser.parse_args()
    prepare(args.src_root, args.npy_fp)


if __name__ == '__main__':
    main()
