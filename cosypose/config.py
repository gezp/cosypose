import cosypose
import os
import yaml

from pathlib import Path
from ament_index_python.packages import get_package_share_directory

# local_data_path = os.path.join(get_package_share_directory('cosypose'), 'local_data')
local_data_path = "/home/ubuntu/gezp/dataset/cosypose/local_data"

LOCAL_DATA_DIR = Path(local_data_path)


assert LOCAL_DATA_DIR.exists()

BOP_DS_DIR = LOCAL_DATA_DIR / 'bop_datasets'
EXP_DIR = LOCAL_DATA_DIR / 'experiments'
