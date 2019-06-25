import pandas as pd
import numpy as np
from typing import Any
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost.sklearn import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import logging
import re
from prettytable import PrettyTable


def train_NB(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict
) -> BernoulliNB:
    """Fits a Bernoulli Naive Bayes model.

    Args:
        X_train: training features.
        y_train: training classes.

    Returns:
        Naive Bayes classifier.

    """
    clf_NB = BernoulliNB()
    clf_NB.fit(X_train, y_train)
    return clf_NB


def train_LR(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict
) -> LogisticRegression:
    """Fits a logistic regression.

    Args:
        X_train: training features.
        y_train: training classes.

    Returns:
        Logistic regression classifier.

    """
    clf_LR = LogisticRegression(**parameters["models"]["LR"])
    clf_LR.fit(X_train, y_train)
    return clf_LR


def train_SVM(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> SVC:
    """Fits a Support Vector Machine.

    Args:
        X_train: training features.
        y_train: training classes.

    Returns:
        SVM classifier.

    """
    clf_SVM = SVC(**parameters["models"]["SVM"])
    clf_SVM.fit(X_train, y_train)
    return clf_SVM


def train_RF(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict
) -> RandomForestClassifier:
    """Fits a Random Forest classifier.

    Args:
        X_train: training features.
        y_train: training classes.

    Returns:
        RF classifier.

    """
    clf_RF = RandomForestClassifier(**parameters["models"]["RF"])
    clf_RF.fit(X_train, y_train)
    return clf_RF


def train_AB(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict
) -> AdaBoostClassifier:
    """Fits an AdaBoost classifier.

    Args:
        X_train: training features.
        y_train: training classes.

    Returns:
        Adaboost classifier.

    """
    clf_AB = AdaBoostClassifier(**parameters["models"]["AB"])
    clf_AB.fit(X_train, y_train)

    return clf_AB


def train_XG(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict
) -> AdaBoostClassifier:
    """Fits an XGBoost classifier.

    Args:
        X_train: training features.
        y_train: training classes.

    Returns:
        XGBoost classifier.

    """
    clf_XG = XGBClassifier(**parameters["models"]["XG"])
    clf_XG.fit(X_train, y_train)

    return clf_XG


def train_NN(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> Sequential:
    """Fits an artificial neural network classifier.

    Args:
        X_train: training features.
        y_train: training classes.

    Returns:
        Neural network.

    """

    # Build NN structure
    N_features = X_train.shape[1]
    parameters = parameters["models"]["NN"]
    clf_NN = Sequential()
    for _ in range(parameters["layers"]["depth"]):
        clf_NN.add(Dense(N_features, **parameters["layers"]["dense"]))
    clf_NN.add(Dense(1, **parameters["layers"]["output"]))

    # Compile and train NN
    sgd = SGD()
    clf_NN.compile(optimizer=sgd, **parameters["compile"])
    y_train = np.array(y_train)
    clf_NN.fit(X_train, y_train, **parameters["fit"])

    return clf_NN


def get_type(clf: Any) -> str:
    """Get the type of a classifier as a string.

    Args:
        clf: a classifier object.

    Returns:
        The type of the classifier as a string.

    """

    pattern = re.compile("\.([^.]*)'")
    result = re.search(pattern, str(type(clf)))
    name = result.group(1)
    # name = " ".join(re.findall("[A-Z]+(?!=[A-Z])[^A-Z]*", name))
    # name = name.replace("NB", "Naive Bayes")
    # name = name.replace(" Classifier", "")
    # if name == "SVC":
    #     name = "Support Vector Machine"
    # elif name == "Sequential":
    #     name += " Neural Network"
    # elif name == "Voting":
    #     name = "Ensemble Voting ({})".format(clf.get_params()["voting"])
    if name == "VotingClassifier":
        name += " ({})".format(clf.get_params()["voting"])
    return name


def train_voting(X_train: pd.DataFrame, y_train: pd.Series, *clfs) -> VotingClassifier:
    """Fits hard and soft ensemble voting classifiers.

    Args:
        X_train: training features.
        y_train: training classes.
        clfs: classifiers to form the voting panel.

    Returns:
        A list of hard and soft voting classifiers.

    """
    estimators = [(get_type(clf), clf) for clf in clfs]
    hard = VotingClassifier(estimators=estimators, voting="hard")
    hard.fit(X_train, y_train)
    soft = VotingClassifier(estimators=estimators, voting="soft")
    soft.fit(X_train, y_train)
    return [hard, soft]


def train_NN(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> Sequential:
    """Fits an artificial neural network classifier.

    Args:
        X_train: training features.
        y_train: training classes.

    Returns:
        Neural network.

    """

    # Build NN structure
    N_features = X_train.shape[1]
    parameters = parameters["models"]["NN"]
    clf_NN = Sequential()
    for _ in range(parameters["layers"]["depth"]):
        clf_NN.add(Dense(N_features, **parameters["layers"]["dense"]))
    clf_NN.add(Dense(1, **parameters["layers"]["output"]))

    # Compile and train NN
    sgd = SGD()
    clf_NN.compile(optimizer=sgd, **parameters["compile"])
    y_train = np.array(y_train)
    clf_NN.fit(X_train, y_train, **parameters["fit"])

    return clf_NN


def evaluate_models(X_valid: pd.DataFrame, y_valid: pd.Series, *clfs) -> None:
    """Logs model evaluation metrics.

    Args:
        X_valid: validation features.
        y_valid: validation classes.
        clfs: model classifier objects.

    """
    results = PrettyTable(
        ["Model", "Accuracy", "Precision", "Recall", "F1 score", "AUC"]
    )
    DIGITS = 2

    for clf in clfs:
        # Get name of classifier
        name = get_type(clf)

        # Generate predicted values
        if type(clf) == Sequential:
            y_pred = clf.predict_classes(X_valid)
        else:
            y_pred = clf.predict(X_valid)

        # Generate performance metrics
        results.add_row(
            [
                name,
                round(accuracy_score(y_valid, y_pred), DIGITS),
                round(precision_score(y_valid, y_pred), DIGITS),
                round(recall_score(y_valid, y_pred), DIGITS),
                round(f1_score(y_valid, y_pred), DIGITS),
                round(roc_auc_score(y_valid, y_pred), DIGITS),
            ]
        )

    # Log performance metrics
    log = logging.getLogger(__name__)
    title = "Model evaluation metrics:"
    size = "Validation size: {}".format(len(y_valid))
    output = "\n".join([title, "", size, "", str(results), ""])
    log.info(output)


def make_predictions(
    clf: Any, edges_test: list, X_test: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """Generates predictions on the test data.

    Args:
        clf: a classifier object.
        edges_test: list of test edges.
        X_test: test features.

    Returns:
        A Pandas DataFrame with the predictions.
    """

    if parameters["predict"]["probability"] and type(clf) != Sequential:
        y_pred = [prob[1] for prob in clf.predict_proba(X_test)]
    else:
        y_pred = clf.predict(X_test)
    row_list = [dict(Id=i + 1, Predictions=y_pred[i]) for i in range(len(edges_test))]
    output = pd.DataFrame(row_list)
    return output
