import logging
import pandas as pd
import numpy as np
from typing import Any
from itertools import compress
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import logging
import re
from prettytable import PrettyTable
from twitter_network.nodes.setup import red, blue


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


def train_ET(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict
) -> ExtraTreesClassifier:
    """Fits an extra trees classifier.

    Args:
        X_train: training features.
        y_train: training classes.

    Returns:
        ET classifier.

    """
    clf_ET = ExtraTreesClassifier(**parameters["models"]["ET"])
    clf_ET.fit(X_train, y_train)
    return clf_ET


def train_GB(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict
) -> GradientBoostingClassifier:
    """Fits a gradient boosting classifier.

    Args:
        X_train: training features.
        y_train: training classes.

    Returns:
        GB classifier.

    """
    clf_GB = GradientBoostingClassifier(**parameters["models"]["GB"])
    clf_GB.fit(X_train, y_train)
    return clf_GB


def train_IF(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict
) -> IsolationForest:
    """Fits an isolation forest classifier.

    Args:
        X_train: training features.
        y_train: training classes.

    Returns:
        IF classifier.

    """
    clf_IF = IsolationForest(**parameters["models"]["IF"])
    clf_IF.fit(X_train, y_train)
    return clf_IF


def train_HGB(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict
) -> HistGradientBoostingClassifier:
    """Fits a Histogram-based Gradient Boosting forest classifier.

    Args:
        X_train: training features.
        y_train: training classes.

    Returns:
        HGB classifier.

    """

    log = logging.getLogger(__name__)

    # # Recursive feature elimination on HGB
    # clf_HGB = HistGradientBoostingClassifier(**parameters["models"]["HGB"])
    # rfe = RFE(estimator=clf_HGB, n_features_to_select=30, step=1)
    # rfe.fit(X_train, y_train)
    # print()
    #
    # # Get selected variables
    # features = list(X_train.columns)
    # mask = rfe.support_
    # included = list(compress(features, mask))
    # excluded = list(set(features) - set(included))
    # log.info(blue("Included {} variables: {}".format(len(included), included)))
    # log.info(blue("Excluded {} variables: {}".format(len(excluded), excluded)))

    clf_HGB = HistGradientBoostingClassifier(**parameters["models"]["HGB"])
    clf_HGB.fit(X_train, y_train)

    return clf_HGB


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
    non_stand_X_train: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: dict,
) -> XGBClassifier:
    """Fits an XGBoost classifier.

    Args:
        X_train: training features.
        y_train: training classes.

    Returns:
        XGBoost classifier.

    """
    log = logging.getLogger(__name__)

    # # Parameters for grid search
    # param_grid = {
    #     "max_depth": [3, 5],
    #     "subsample": [0.8, 1],
    #     "min_samples_split": [50, 100, 150],
    #     "colsample_bytree": [0.8, 1],
    #     "learning_rate": [0.05, 0.1, 0.15],
    #     "gamma": [0, 0.0001],
    #     "importance_type": ["gain", "cover"],
    #     "n_estimators": [50, 100, 150],
    # }
    #
    # default_params = {
    #     "n_jobs": -1,
    #     "random_state": 1234,
    #     "verbosity": 0,
    #     "objective": "binary:logistic",
    # }

    # # run random search
    # log.info(blue("Tuning XGBoost hyperparameters..."))
    # xgb_search = XGBClassifier(**default_params)
    # xgb_random = RandomizedSearchCV(
    #     xgb_search, param_distributions=param_grid, cv=5, iid=False
    # )
    # xgb_random.fit(X_train, y_train)
    # log.info(blue("Best estimator:\n" + str(xgb_random.best_estimator_)))

    # Recursive feature elimination on XG
    # log.info(blue("Running recursive feature elimination..."))
    tuned_params = dict(
        base_score=0.5,
        booster="gbtree",
        colsample_bylevel=1,
        colsample_bynode=1,
        colsample_bytree=1,
        gamma=0.0001,
        learning_rate=0.1,
        max_delta_step=0,
        max_depth=5,
        min_child_weight=1,
        min_samples_split=50,
        missing=None,
        n_estimators=150,
        n_jobs=-1,
        nthread=None,
        objective="binary:logistic",
        random_state=1234,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        seed=None,
        silent=None,
        subsample=1,
        verbosity=0,
    )
    xgb_tuned = XGBClassifier(**tuned_params)
    xgb_tuned.fit(X_train, y_train)
    # rfe = RFE(estimator=xgb_tuned, n_features_to_select=30, step=1, verbose=2)
    # rfe.fit(X_train, y_train)

    # # Get selected variables
    # features = list(non_stand_X_train.columns)
    # mask = rfe.support_
    # included = list(compress(features, mask))
    # excluded = list(set(features) - set(included))
    # log.info(blue("Included {} variables: {}".format(len(included), included)))
    # log.info(blue("Excluded {} variables: {}".format(len(excluded), excluded)))

    return xgb_tuned


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
    paras = parameters["models"]["NN"]
    clf_NN = Sequential()
    for _ in range(paras["layers"]["depth"]):
        clf_NN.add(
            Dense(N_features, **paras["layers"]["dense"], input_shape=(N_features,))
        )
        if paras["layers"]["dropout"]:
            clf_NN.add(Dropout(rate=paras["layers"]["dropout"]))
    clf_NN.add(Dense(1, **paras["layers"]["output"]))

    # Compile and train NN
    sgd = SGD()
    clf_NN.compile(optimizer=sgd, **paras["compile"])
    y_train = np.array(y_train)
    clf_NN.fit(X_train, y_train, **paras["fit"])

    return clf_NN


class Jacob:
    def __init__(self, threshold, target) -> None:
        self.threshold = threshold
        self.target = target

    def smooth_soft(self, x: float) -> float:
        if x > self.threshold:
            return 1.0
        else:
            return x

    def smooth_hard(self, x: float) -> float:
        if x > self.threshold:
            return 1.0
        else:
            return 0.0

    def predict_proba(self, X_test: pd.DataFrame) -> list:
        return [self.smooth_soft(x) for x in X_test[self.target].values]

    def predict(self, X_test: pd.DataFrame) -> list:
        return [self.smooth_hard(x) for x in X_test[self.target].values]


def train_Jacob(X_train: pd.DataFrame, parameters: dict) -> Jacob:
    # Set threshold parameter
    threshold = parameters["models"]["Jacob"]["threshold"]
    target = parameters["models"]["Jacob"]["target"]
    if threshold == "median":
        # Get the median value of the target variable
        threshold = np.median(X_train[target])
    elif type(threshold) == float:
        pass
    else:
        raise TypeError("Jacob's threshold must be one of: float or 'median'.")
    # Initialise instance of Jacob
    clf_Jacob = Jacob(threshold=threshold, target=target)
    return clf_Jacob


def binarize(x, threshold):
    if x <= threshold:
        return 0
    else:
        return 1


def IF_predict(clf, X):
    y_pred = [1 + y for y in clf.score_samples(X)]
    y_mean = sum(y_pred) / len(y_pred)
    y_pred = [binarize(y, y_mean) for y in y_pred]
    return y_pred


def evaluate_models(
    X_valid: pd.DataFrame, y_valid: pd.Series, non_stand_X_valid: pd.DataFrame, *clfs
) -> None:
    """Logs model evaluation metrics.

    Args:
        X_valid: validation features.
        y_valid: validation classes.
        non_stand_X_test: unstandardised test features for Jacob classifier.
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
        elif type(clf) == Jacob:
            y_pred = clf.predict(non_stand_X_valid)

        elif type(clf) == IsolationForest:
            y_pred = IF_predict(clf, X_valid)
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
    clf: Any,
    edges_test: list,
    X_test: pd.DataFrame,
    non_stand_X_test: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Generates predictions on the test data.

    Args:
        clf: a classifier object.
        edges_test: list of test edges.
        X_test: test features.

    Returns:
        A Pandas DataFrame with the predictions.
    """
    if type(clf) == Jacob:
        y_pred = clf.predict(non_stand_X_test)
    elif type(clf) == IsolationForest:
        y_pred = IF_predict(clf, X_test)
    elif parameters["predict"]["probability"]:
        y_pred = [prob[1] for prob in clf.predict_proba(X_test)]
    else:
        y_pred = clf.predict(X_test)

    if type(clf) == Sequential:
        row_list = [
            dict(Id=i + 1, Predictions=y_pred[i][0]) for i in range(len(edges_test))
        ]
    else:
        row_list = [
            dict(Id=i + 1, Predictions=y_pred[i]) for i in range(len(edges_test))
        ]

    output = pd.DataFrame(row_list)
    return output
