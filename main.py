import json
import pickle
import os
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.data import process_data


dir = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(dir, "model/logistic_regression_model.pkl"), "rb"))
cat_features = pickle.load(open(os.path.join(dir, "model/cat_features.pkl"), "rb"))
encoder = pickle.load(open(os.path.join(dir, "model/encoder.pkl"), "rb"))
lb = pickle.load(open(os.path.join(dir, "model/lb.pkl"), "rb"))


class InputDataRecord(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True
        response_model_by_alias = True


# Instantiate the app.
app = FastAPI()



# GET on the root gives a welcome message
@app.get("/")
async def welcome_message():
    return {"welcome": "Welcome, this is the API of a model for the prediction of salary classes"}


@app.post("/predict")
async def predict(input_data: List[InputDataRecord]):
    values = [item.dict().values() for item in input_data]
    X = pd.DataFrame(values, columns=input_data[0].dict(by_alias=True).keys())
    X_transformed, _, _, _ = process_data(
        X, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
    )
    return {"prediction": model.predict(X_transformed).tolist()}

