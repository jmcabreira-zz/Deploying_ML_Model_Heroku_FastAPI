# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
import pickle
from typing import Literal
import pandas as pd
import yaml
from box import Box

from starter.ml.model import inference
from starter.ml.data import process_data
from starter.CensusClass.Census_Class import CensusData, Prediction

app = FastAPI()

ROOT = os.getcwd()
CONFIG_FILEPATH = os.path.join(ROOT, "config.yaml")

with open(CONFIG_FILEPATH, "r", encoding="utf-8") as ymlfile:
    config = Box(yaml.safe_load(ymlfile))

ENCODER_FILEPATH = os.path.join(ROOT, config.preprocessing.encoder_file_path)
BINARIZER_FILEPATH = os.path.join(ROOT, config.preprocessing.binarizer_file_path)
MODEL_FILEPATH = os.path.join(
    ROOT, config.model.GradientBoostingClassifier.output_file_pth
)

app = FastAPI()

categorical_features = config.preprocessing.cat_features
label = config.preprocessing.label

model = pickle.load(open(MODEL_FILEPATH, "rb"))
encoder = pickle.load(open(ENCODER_FILEPATH, "rb"))
binarizer = pickle.load(open(BINARIZER_FILEPATH, "rb"))


@app.get("/")
def root():
    return {"Greeting": "Hello and Welcome :)"}


@app.post("/predict/", response_model=Prediction, status_code=200)
async def make_predictions(request_data: CensusData):
    request_df = pd.DataFrame(
        {k: v for k, v in request_data.dict(by_alias=True).items()}, index=[0]
    )

    X, _, _, _ = process_data(
        X=request_df,
        categorical_features=categorical_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=binarizer,
    )
    prediction = inference(model=model, X=X)

    prediction = binarizer.inverse_transform(prediction).tolist()[0]

    print("Predicted Income:", prediction)
    return {"prediction": prediction}
