import requests
import json

# hhtp = 'http://127.0.0.1:8000/predict/'
url = "https://cabreira-census-data-app.herokuapp.com/predict/"
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
headers = {"content-type": "application/json"}

# response = requests.get(url)
# response = requests.post(url, data=json.dumps(data))
response = requests.post(url, data=json.dumps(data), headers=headers)
print(response)
print(response.json())
