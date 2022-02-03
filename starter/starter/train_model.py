"""
This script Trains the machine learning model 
Author: Jonathan Cabreira
Date: January 2022
"""

# Add the necessary imports for the starter code.
import os
import json
import yaml
from box import Box
import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import pickle
import numpy as np

# Prepare logging
STARTER_ROOT = str(Path(__file__).resolve().parents[1])
LOG_FOLDER = os.path.join(STARTER_ROOT, 'logs')
CONFIG_FILEPATH = os.path.join(STARTER_ROOT, "config.yaml")
LOG_FILE_PTH = os.path.join(LOG_FOLDER, 'train_model.log')

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)
print(CONFIG_FILEPATH)
with open(CONFIG_FILEPATH, 'r', encoding='UTF-8') as configfile:
    config = Box(yaml.safe_load(configfile))

CLEANED_DATA_FILEPATH = os.path.join(
    STARTER_ROOT, config.data.cleaned.filepath)

logging.basicConfig(
    filename=LOG_FILE_PTH,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='w',
    level=logging.INFO
)


def run_train_model():
    logging.info('Train model log')
    # Add code to load in the data.
    data = pd.read_csv(CLEANED_DATA_FILEPATH).drop(["Unnamed: 0"], axis=1)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data,
                                   test_size=config.preprocessing.test_size,
                                   random_state=config.model.random_seed)

    # Proces the train data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=config.preprocessing.cat_features,
        label=config.preprocessing.label,
        training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=config.preprocessing.cat_features,
        label=config.preprocessing.label,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Train the model
    model = train_model(X_train, y_train, config)

    # Save model and encoder
    pickle.dump(model, open(os.path.join(
        STARTER_ROOT, config.model.GradientBoostingClassifier.output_file_pth), "wb"))
    pickle.dump(encoder, open(os.path.join(
        STARTER_ROOT, config.preprocessing.encoder_file_path), "wb"))
    pickle.dump(lb, open(os.path.join(
        STARTER_ROOT, config.preprocessing.binarizer_file_path), "wb"))

    # Save processes data
    with open(os.path.join(STARTER_ROOT, config.preprocessing.train_filepath), "wb") as f:
        np.save(f, X_train)
        np.save(f, y_train)

    with open(os.path.join(STARTER_ROOT, config.preprocessing.test_filepath), "wb") as f:
        np.save(f, X_test)
        np.save(f, y_test)


if __name__ == '__main__':
    run_train_model()
