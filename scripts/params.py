from pathlib import Path
from pydoc import locate

import yaml


PARAMETERS_FILE = "params.yaml"


with open(PARAMETERS_FILE) as file:
    print(f"Parsing parameters from \"{PARAMETERS_FILE}\"")
    params = yaml.safe_load(file)


ROOT_DIR = Path(params["root_dir"])

# Dataset parameters
RAW_DATASET_DIR = ROOT_DIR / params["data"]["download"]["subdir"]
RAW_DATASET_VAL_DIR = ROOT_DIR / params["data"]["download_val"]["subdir"]
DATASET_DIR = ROOT_DIR / params["data"]["dataset"]["subdir"]


# Train parameters
TRAIN_DIR = ROOT_DIR / params["train"]["subdir"]
