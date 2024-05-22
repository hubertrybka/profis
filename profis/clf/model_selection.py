import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np


def nested_CV(
        model, X, y, param_grid, n_splits=5, n_jobs=-1, scoring="roc_auc", optimize=True, verbose=False
):
    """
    Nested CV grid search for the best hyperparameters of the SVC model.
    Args:
        model: model to be evaluated
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
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True)

    if optimize:
        # inner loop: parameter search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=2 if verbose else 0,
            refit=True
        )
    else:
        grid_search = model

    # outer loop: model evaluation
    test_scores = []
    eval_scores = []
    models = []

    for train_index, test_index in outer_cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        grid_search.fit(X_train, y_train)
        pred_test = grid_search.predict_proba(X_test)
        auc_test = roc_auc_score(y_test, pred_test[:, 1])
        test_scores.append(auc_test)
        if optimize:
            eval_scores.append(grid_search.best_score_)
            models.append(grid_search.best_estimator_)

    if optimize:
        best_model = models[eval_scores.index(max(eval_scores))]
    else:
        best_model = model

    return best_model, np.array(test_scores)