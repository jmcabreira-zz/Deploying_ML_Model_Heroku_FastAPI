import os
import logging
import pytest
from pathlib import Path
import yaml
from box import Box
import pandas as pd

STARTER_ROOT = str(Path(__file__).resolve().parents[2])
LOG_FOLDER = os.path.join(STARTER_ROOT, "logs")
CONFIG_FILEPATH = os.path.join(STARTER_ROOT, "config.yaml")
LOG_FILE_PTH = os.path.join(LOG_FOLDER, "unit_test.log")

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)
print(CONFIG_FILEPATH)
with open(CONFIG_FILEPATH, "r", encoding="UTF-8") as configfile:
    config = Box(yaml.safe_load(configfile))

CLEANED_DATA_FILEPATH = os.path.join(STARTER_ROOT, config.data.cleaned.filepath)


@pytest.fixture
def data():
    """
    Load the cleaned dataset
    """
    df = pd.read_csv(os.path.join(STARTER_ROOT, "data/census_cleaned.csv")).drop(
        ["Unnamed: 0"], axis=1
    )
    return df


def test_dataframe_shape(data):
    """
    test dataframe shape
    """
    try:
        assert len(data.shape[0]) > 0
        assert len(data.shape[1]) == 15
        logging.info(
            f"Testing the cleaned data: The cleaned dataset has {data.shape[0]} rows and {data.shape[1]} columns as expected."
        )
    except AssertionError as e:
        logging.error(
            f"Testing the cleaned data: The cleaned dataset has {data.shape[0]} rows and {data.shape[1]}."
        )
        raise e


def test_columns_name(data):
    """
    test the columns dataset name
    """
    right_columns = [
        "age",
        "workclass",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
        "salary",
    ]

    data_columns = data.columns.values
    try:
        assert list(right_columns) == list(data_columns)
        logging.info("Testing Cleaned data: Column names are as expected")
    except AssertionError as e:
        logging.error(
            f"Testing cleaned data: Column names are not as expected. Dataset columnsa: {data_columns}"
        )
        raise e


def test_missing_values(data):

    try:
        assert data.isnull().sum() == 0
        logging.info("Testing cleaned data: No rows with null values.")
    except AssertionError as e:
        logging.error(
            f"Testing cleaned data: Expected no null values but found {data.isnull().sum()} missing values"
        )
        raise e


def test_question_mark(data):
    """
    Check whether the data has a question mark in it
    """
    try:
        assert "?" not in data.values
        logging.info("Testing the cleaned data: No question mark in cleaned data")
    except AssertionError as e:
        logging.error(f"Testing cleaned data: There is still ? in the clenaed data")
        raise e
