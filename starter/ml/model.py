from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
from starter.ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Init logistic regression model
    model = LogisticRegression()
    # train model
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def slice_data(df, feature):
    """ Function to get slices of the data for each class of the feature

    Inputs
    ------
    df : pd.DataFrame
         input data frame
    feature : str
         name of the categorical feature the data frame should split on
    Returns
    -------
    df_dict : dict(pd.DataFrame)
              dictionary of data frame parts
    """
    df_dict = dict()

    for cls in df[feature].unique():
        df_tmp = df[df[feature] == cls]
        df_dict[cls] = df_tmp

    return df_dict


def performance_sliced_data(df, category, cat_features, encoder, lb, model):
    df_slices = slice_data(df, category)
    print(f'### Model performance on slices of data for category {category}: ###')
    for class_name in df_slices.keys():
        X_test_crt, y_test_crt, _, _ = process_data(
            df_slices[class_name], categorical_features=cat_features, label="salary", training=False, encoder=encoder,
            lb=lb
        )
        pred_crt = inference(model, X_test_crt)
        crt_precision, crt_recall, crt_fbeta = compute_model_metrics(y_test_crt, pred_crt)
        print(f"Class {class_name}: beta={crt_fbeta}, recall={crt_recall}, precision={crt_precision}")
    print('#####################################################################')
