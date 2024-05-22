import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score
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


def nested_CV(
        model, X, y, param_grid, n_splits=5, n_jobs=-1, scoring="roc_auc", optimize=True, verbose=False
):
    """
    Nested CV grid search for the best hyperparameters of the SVC model.
    Args:
        model (sklearn.svm.SVC): model to be evaluated
        X: data features
        y: data labels
        param_grid (dict): dictionary containing the hyperparameters to be evaluated
        n_splits (int): number of splits for cross-validation
        n_jobs (int): number of parallel jobs
        scoring (str): scoring metric
        optimize (bool): whether to perform hyperparameter optimization
        verbose (bool): verbosity
    Returns:
        best_params (dict): dictionary containing the best hyperparameters
    """

    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    outer_cv = KFold(n_splits=n_splits, shuffle=True)

    if optimize:
        # inner loop: parameter search
        model = GridSearchCV(
            model,
            param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=2 if verbose else 0,
        )

    # outer loop: model evaluation
    test_scores = cross_val_score(model,
                                  X,
                                  y,
                                  cv=outer_cv,
                                  n_jobs=n_jobs,
                                  scoring=scoring)

    return model, test_scores
