#!/bin/bash

echo "##### Pip install..."
python3 -m pip install --upgrade pip
pip install -q -e .

echo "##### wandb login ..."
wandb login b18a65ed03a0379d9c70d2de641e5fa80bf6f3b8
wandb online

echo "##### Python script"
python3 scripts/check_imports.py
python3 scripts/download_tiny_shakespear.py
python3 scripts/train.py

echo "##### Stopping pod"
runpodctl remove pod $RUNPOD_POD_ID