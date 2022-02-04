import os
import logging
import pytest
from pathlib import Path
import yaml
from box import Box

STARTER_ROOT = str(Path(__file__).resolve().parents[2])
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


@pytest.fixture
def data():
    """
    Load the cleaned dataset
    """
    df = starter.helper_function.import_data("data/census_cleaned.csv")
    return df


def test_import_data(data):
    '''
    test import_data 
    '''
    assert data.shape[0] > 0
    assert data.shape[1] > 0


def test_process_data(data):
    """
    test process_data function
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_test, y_test, _, _ = starter.helper_function.process_data(
        data, categorical_features=cat_features, label="salary", training=True)

    assert len(X_test) == len(y_test)


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
        "salary"
    ]

    obtained_columns = data.columns.values
    assert list(right_columns) == list(obtained_columns)
