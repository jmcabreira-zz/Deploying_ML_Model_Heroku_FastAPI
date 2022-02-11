import requests
import json

data = {
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
r = requests.post("http://127.0.0.1:8000/predict/", data=json.dumps(data))
print(r)
print(r.json())
