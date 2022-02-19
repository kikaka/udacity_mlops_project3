# Script to train machine learning model.
import os
import pickle

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, slice_data, performance_sliced_data

# load data from census_cleaned.csv
data = pd.read_csv('../data/census_cleaned.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
model = train_model(X_train, y_train)
pickle.dump(model, open('../model/logistic_regression_model.pkl', 'wb'))
pickle.dump(cat_features, open('../model/cat_features.pkl', "wb"))
pickle.dump(encoder, open('../model/encoder.pkl', "wb"))
pickle.dump(lb, open('../model/lb.pkl', "wb"))


predictions = inference(model, X_test)
model_precision, model_recall, model_fbeta = compute_model_metrics(y_test, predictions)
print('precision: ', model_precision)
print('recall: ', model_recall)
print('fbeta: ', model_fbeta)

# performance of the model on slices of data
for category in cat_features:
    performance_sliced_data(test, category, cat_features=cat_features, encoder=encoder, lb=lb, model=model)


