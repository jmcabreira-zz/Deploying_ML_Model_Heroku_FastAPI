from fastapi.testclient import TestClient
from pathlib import Path
import sys
import logging
import pytest

APP_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, APP_ROOT)
print(APP_ROOT)
from main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@pytest.fixture
def client():
    with TestClient(app) as clt:
        yield clt


def test_welcome(client):
    req = client.get("/")
    assert req.status_code == 200, "Status code is not 200"
    assert req.json() == {"Greeting": "Hello and Welcome :)"}, "Wrong json output"


def test_below_50k_pred(client):
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    with TestClient(app) as client:
        response = client.post("/predict/", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}, "Wrong json output"


def test_above_50k_pred(client):
    data_ = {
        "age": 41,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2020,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
    }
    with TestClient(app) as client:
        response = client.post("/predic/", json=data_)
        print(response.text)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}, "Wrong prediction"
