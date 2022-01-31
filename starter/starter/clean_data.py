"""
This script Cleans the raw data
Author: Jonathan Cabreira
Date: January 2022
"""
import sys
from pathlib import Path
import yaml
from box import Box
import os

STARTER_ROOT = str(Path(__file__).resolve().parents[1])
# Adds cwd path to the sys.path list in the first position so 
# that python will search the cwd first 
sys.path.insert(0,STARTER_ROOT)

# Project Directories
from ml.data import clean_raw_data
CONFIG_FILEPATH = os.path.join(STARTER_ROOT, "config.yaml")

# Load config file
with open(CONFIG_FILEPATH, "r", encoding="utf-8") as ymlfile:
  config = Box(yaml.safe_load(ymlfile))

RAW_DATA_FILE_PTH = os.path.join(STARTER_ROOT,config.data.raw.filepath)
CLEANED_DATA_FILE_PTH = os.path.join(STARTER_ROOT, config.data.cleaned.filepath)

def run_clean_data():
  cleaned_df = clean_raw_data(filepath = RAW_DATA_FILE_PTH)
  cleaned_df.to_csv(CLEANED_DATA_FILE_PTH)

if __name__ == '__main__':
  run_clean_data()