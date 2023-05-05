from pytorch_lightning.cli import LightningCLI

from dataset import CSVDataModule
from model import Inception_V3


def cli_main():
    cli = LightningCLI(Inception_V3, CSVDataModule)


if __name__ == "__main__":
    cli_main()
