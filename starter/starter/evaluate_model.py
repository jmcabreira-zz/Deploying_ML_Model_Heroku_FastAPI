import pickle
import sys
from pathlib import Path
import os

import numpy as np
import pandas as pd
import yaml
from box import Box
from sklearn.model_selection import train_test_split
import ml
from ml.model import compute_model_metrics, inference
from ml.data import process_data

STARTER_ROOT = str(Path(__file__).resolve().parents[1])
CONFIG_FILEPATH = os.path.join(STARTER_ROOT, "config.yaml")

with open(CONFIG_FILEPATH, 'r', encoding='UTF-8') as configfile:
    config = Box(yaml.safe_load(configfile))

CAT_FEATURES = config.preprocessing.cat_features
CLEANED_DATA_FILEPATH = os.path.join(
    STARTER_ROOT, config.data.cleaned.filepath)
SLICE_OUTPUT_FILEPATH = os.path.join(
    STARTER_ROOT, config.model.metrics.slice_filepath)
ALL_TRAINING_OUTPUT_FILEPATH = os.path.join(
    STARTER_ROOT, config.model.metrics.all_train_filepath)
ALL_TEST_OUTPUT_FILEPATH = os.path.join(
    STARTER_ROOT, config.model.metrics.all_test_filepath)


def run_evaluate_all():

    train_data_filepath = config.preprocessing.train_filepath
    test_data_filepath = config.preprocessing.test_filepath
    model_filepath = config.model.GradientBoostingClassifier.output_file_pth

    with open(os.path.join(STARTER_ROOT, train_data_filepath), "rb") as f:
        X_train = np.load(f)
        y_train = np.load(f)

    with open(os.path.join(STARTER_ROOT, test_data_filepath), "rb") as f:
        X_test = np.load(f)
        y_test = np.load(f)

    model = pickle.load(open(os.path.join(STARTER_ROOT, model_filepath), "rb"))

    # Predictions and score
    train_preds = inference(model=model, X=X_train)
    test_preds = inference(model=model, X=X_test)
    train_scores = compute_model_metrics(y=y_train, preds=train_preds)
    test_scores = compute_model_metrics(y=y_test, preds=test_preds)

    # Save results
    train_results = {
        "precision": train_scores[0],
        "recall": train_scores[1],
        "fbeta": train_scores[2],
    }

    test_results = {
        "precision": test_scores[0],
        "recall": test_scores[1],
        "fbeta": test_scores[2],
    }

    with open(ALL_TRAINING_OUTPUT_FILEPATH, "w") as f:
        f.write(str(train_results))
    with open(ALL_TEST_OUTPUT_FILEPATH, "w") as f:
        f.write(str(test_results))


def run_evaluate_slice_scores():
    categorical_features = config.preprocessing.cat_features
    label = config.preprocessing.label
    ohe_encoder_filepath = config.preprocessing.encoder_file_path
    binarizer_filepath = config.preprocessing.binarizer_file_path
    model_filepath = config.model.GradientBoostingClassifier.output_file_pth

    clean_df = pd.read_csv(CLEANED_DATA_FILEPATH).drop(["Unnamed: 0"], axis=1)
    ohe_encoder = pickle.load(
        open(os.path.join(STARTER_ROOT, ohe_encoder_filepath), "rb"))
    label_binarizer = pickle.load(
        open(os.path.join(STARTER_ROOT, binarizer_filepath), "rb"))
    model = pickle.load(
        open(os.path.join(STARTER_ROOT, model_filepath), "rb"))

    all_scores_df = pd.DataFrame(
        columns=[
            "feature",
            "category",
            "num_samples",
            "precision",
            "recall",
            "fbeta",
        ]
    )

    _, test = train_test_split(
        clean_df, test_size=0.20, random_state=config.model.random_seed
    )

    for feature in categorical_features:
        for category in test[feature].unique():
            filtered_df = test[test[feature] == category]
            num_samples = len(filtered_df)

            # Process filtered data
            X_test, y_test, _, _ = process_data(
                X=filtered_df,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=ohe_encoder,
                lb=label_binarizer,
            )

            # Predictions and score
            preds = inference(model=model, X=X_test)
            scores = compute_model_metrics(y=y_test, preds=preds)
            scores_list = [
                feature,
                category,
                num_samples,
                scores[0],
                scores[1],
                scores[2],
            ]
            scores_series = pd.Series(scores_list, index=all_scores_df.columns)

            # Add scores to DataFrame
            all_scores_df = all_scores_df.append(
                scores_series, ignore_index=True
            )

    all_scores_df.to_csv(SLICE_OUTPUT_FILEPATH)


if __name__ == "__main__":
    run_evaluate_slice_scores()
    run_evaluate_all()
