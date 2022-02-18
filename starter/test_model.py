import pandas as pd
import pytest
from starter.ml.model import train_model, compute_model_metrics, inference


def test_train_model():
    model = train_model([[1, 32, 3], [1, 5, 6], [5, 7, 0]], [0, 0, 1])
    assert str(type(model)) == "<class 'sklearn.linear_model._logistic.LogisticRegression'>"

    pred = model.predict([[1, 3, 4], [11, 13, 14], [5, 6, 0]])
    assert all(pd.Series(pred).between(0, 1))


def test_compute_model_metrics():
    mm = compute_model_metrics([0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                               [1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    expected_result = [0.4, 2.0/3.0, 0.5]
    assert mm == pytest.approx(expected_result)


def test_inference():
    model = train_model([[12, 2, 32, 3], [18, 1, 5, 6], [30, 5, 7, 0],
                         [30, 15, 17, 20], [3, 7, 9, 2], [10, 25, 1, 1]],
                        [1, 1, 0, 1, 0, 1])
    predictions = inference(model, [[1, 34, 12, 12], [5, 5, 5, 5]])
    assert all(pd.Series(predictions).between(0, 1))
