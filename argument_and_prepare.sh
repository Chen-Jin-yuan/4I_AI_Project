#!/usr/bin/env bash
python3 split.py
python3 argument.py
python3 prepare.py
python3 prepare.py data/val_split.csv data/1.\ Original\ Images/a.\ Training\ Set data/val_split.npy
