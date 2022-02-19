import json

from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


def test_api_get_root():
    r = client.get("/")
    cont = r.json()
    assert 'welcome' in cont.keys()
    assert cont['welcome'] == 'Welcome, this is the API of a model for the prediction of salary classes'
    assert r.status_code == 200


def test_api_post_predict_response_code():
    data = [{"age": 35, "workclass": "Without-pay", "fnlgt": 94638,
            "education": "Masters",
            "education_num": 15,
            "marital_status": "Widowed",
            "occupation": "Tech-support",
            "relationship": "Not-in-family",
            "race": "Amer-Indian-Eskimo",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "Germany"}]

    r = client.post("/predict", data=json.dumps(data))

    assert r.status_code == 200


def test_api_post_predict_prediction():
    data = [{
        "age": 65, "workclass": "Private", "fnlgt": 94638,
         "education": "Masters",
         "education_num": 15,
         "marital_status": "Widowed",
         "occupation": "Tech-support",
         "relationship": "Not-in-family",
         "race": "Amer-Indian-Eskimo",
         "sex": "Male",
         "capital_gain": 100000,
         "capital_loss": 0,
         "hours_per_week": 40,
         "native_country": "United-States"},
        {"age": 35, "workclass": "Without-pay", "fnlgt": 94638,
         "education": "Masters",
         "education_num": 15,
         "marital_status": "Widowed",
         "occupation": "Tech-support",
         "relationship": "Not-in-family",
         "race": "Amer-Indian-Eskimo",
         "sex": "Male",
         "capital_gain": 0,
         "capital_loss": 0,
         "hours_per_week": 40,
         "native_country": "Germany"}
    ]

    r = client.post("/predict", data=json.dumps(data))
    assert all([i in [0, 1] for i in r.json()['prediction']])
