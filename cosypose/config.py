import cosypose
import os
import yaml

from pathlib import Path

LOCAL_DATA_DIR = LOCAL_DATA_DIR=Path("/home/ubuntu/gezp/dataset/cosypose/local_data")


assert LOCAL_DATA_DIR.exists()

BOP_DS_DIR = LOCAL_DATA_DIR / 'bop_datasets'
EXP_DIR = LOCAL_DATA_DIR / 'experiments'
