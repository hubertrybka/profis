import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score, confusion_matrix


def evaluate(model, test_X, test_y):
    """
    Evaluates the SVC model performance on the test set.
    Args:
        model (sklearn.svm.SVC): trained model
        test_X: test set features
        test_y: test set labels
    Returns:
        metrics (dict): dictionary containing accuracy, ROC AUC and confusion matrix metrics
    """
    predictions = model.predict_proba(test_X)[:, 1]
    df = pd.DataFrame()
    df["pred"] = predictions
    df["label"] = test_y.values
    df["pred"] = df["pred"].apply(lambda x: 1 if x > 0.5 else 0)
    accuracy = df[df["pred"] == df["label"]].shape[0] / df.shape[0]
    try:
        roc_auc = roc_auc_score(df["label"], df["pred"])
    except ValueError:
        print(
            "ROC AUC score could not be calculated. Only one class present in the test set."
        )
        roc_auc = 0
    try:
        tn, fp, fn, tp = confusion_matrix(df["label"], df["pred"]).ravel()
    except ValueError:
        print(
            "Confusion matrix could not be calculated. Only one class present in the test set."
        )
        tn, fp, fn, tp = 0, 0, 0, 0
    metrics = {
        "accuracy": round(accuracy, 4),
        "roc_auc": round(roc_auc, 4),
        "true_positive": round(tp / df.shape[0], 4),
        "true_negative": round(tn / df.shape[0], 4),
        "false_positive": round(fp / df.shape[0], 4),
        "false_negative": round(fn / df.shape[0], 4),
    }
    return metrics


def cross_evaluate(model, X, y, n_splits=10):
    """
    Cross-evaluates the SVC model performance on the training set.
    Args:
        model (sklearn.svm.SVC): trained model
        X: test set features
        y: test set labels
        n_splits (int): number of splits for cross-validation
    Returns:
        metrics (dict): dictionary containing accuracy and ROC AUC metrics
    """
    skf = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
    accuracies = []
    roc_aucs = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        predictions = model.predict_proba(X_test)[:, 1]
        df = pd.DataFrame()
        df["pred"] = predictions
        df["label"] = y_test
        df["pred"] = df["pred"].apply(lambda x: 1 if x > 0.5 else 0)
        accuracy = df[df["pred"] == df["label"]].shape[0] / df.shape[0]
        accuracies.append(accuracy)
        roc_auc = roc_auc_score(df["label"], df["pred"])
        roc_aucs.append(roc_auc)

    metrics = {
        "accuracy": round(np.mean(accuracies), 4),
        "accuracy_std": round(np.std(accuracies), 4),
        "roc_auc": round(np.mean(roc_aucs), 4),
        "roc_auc_std": round(np.std(roc_aucs), 4),
    }
    return metrics


def grid_search(
    model, X, y, param_grid, n_splits=10, n_jobs=-1, scoring="roc_auc", verbose=False
):
    """
    Grid search for the best hyperparameters of the SVC model.
    Args:
        model (sklearn.svm.SVC): model to be evaluated
        X: data features
        y: data labels
        param_grid (dict): dictionary containing the hyperparameters to be evaluated
        n_splits (int): number of splits for cross-validation
        n_jobs (int): number of parallel jobs
        scoring (str): scoring metric
        verbose (bool): verbosity
    Returns:
        best_params (dict): dictionary containing the best hyperparameters
    """
    skf = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
    grid_search = sklearn.model_selection.GridSearchCV(
        model,
        param_grid,
        cv=skf,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=2 if verbose else 0,
    )
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    results = grid_search.cv_results_
    return best_params, results
