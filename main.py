# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
import pickle
from typing import Literal
import pandas as pd
import yaml
from box import Box
from pathlib import Path
import sys
import uvicorn

root = str(Path(__file__).resolve().parents[0])
sys.path.insert(0, os.path.join(root, "starter"))

from starter.ml.model import inference
from starter.ml.data import process_data
from starter.CensusClass.Census_Class import CensusData, Prediction

app = FastAPI()


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# if "DYNO" in os.environ and os.path.isdir(".dvc"):
#     os.system("dvc config core.no_scm true")
#     os.system("dvc remote add -d s3-bucket s3://census-bureau-data-model/data_storage/)
#     if os.system("dvc pull") != 0:
#         exit("dvc pull failed")
#     os.system("rm -r .dvc .apt/usr/lib/dvc")

CONFIG_FILEPATH = os.path.join(root, "starter", "config.yaml")

with open(CONFIG_FILEPATH, "r", encoding="utf-8") as ymlfile:
    config = Box(yaml.safe_load(ymlfile))

ENCODER_FILEPATH = os.path.join(root, "starter", config.preprocessing.encoder_file_path)
BINARIZER_FILEPATH = os.path.join(
    root, "starter", config.preprocessing.binarizer_file_path
)
MODEL_FILEPATH = os.path.join(
    root, "starter", config.model.GradientBoostingClassifier.output_file_pth
)
print("ENCODER_FILEPATH:", ENCODER_FILEPATH)
print("BINARIZER_FILEPATH:", BINARIZER_FILEPATH)
print("MODEL_FILEPATH:", MODEL_FILEPATH)


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


# if __name__ == "__main__":

#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# def make_predictions(request_data: CensusData):
#     request_df = pd.DataFrame({k: v for k, v in request_data.items()}, index=[0])

#     X, _, _, _ = process_data(
#         X=request_df,
#         categorical_features=categorical_features,
#         label=None,
#         training=False,
#         encoder=encoder,
#         lb=binarizer,
#     )
#     prediction = inference(model=model, X=X)

#     # prediction = binarizer.inverse_transform(prediction).tolist()[0]
#     # prediction = [ ">50K" if pred==1 else  pre==0 "<50K"]
#     print(prediction)
#     if prediction == 1:
#         prediction = ">50K"
#     elif prediction == 0:
#         prediction = "<=50K"
#     print("Predicted Income:", prediction)
#     return {"prediction": prediction}


# if __name__ == "__main__":
#     request_data = {
#         "age": 41,
#         "workclass": "Private",
#         "fnlgt": 45781,
#         "education": "Masters",
#         "education-num": 14,
#         "marital-status": "Married-civ-spouse",
#         "occupation": "Prof-specialty",
#         "relationship": "Not-in-family",
#         "race": "White",
#         "sex": "Male",
#         "capital-gain": 2020,
#         "capital-loss": 0,
#         "hours-per-week": 50,
#         "native-country": "United-States",
#     }
#     make_predictions(request_data)
