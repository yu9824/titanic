"""Cross-validation helpers and Kaggle submission file writing."""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .config import DIR_SUBMISSIONS, RANDOM_STATE
from .logging import get_child_logger

__all__ = ("evaluate", "fit_predict", "save_submission")

_logger = get_child_logger(__name__)

# `test.csv` always holds exactly this many passengers.
N_ROWS_SUBMISSION = 418


def _describe(estimator: BaseEstimator) -> str:
    """Name an estimator, unwrapping a `~sklearn.pipeline.Pipeline`.

    Without this every pipeline would just log as ``"Pipeline"``.
    """
    if isinstance(estimator, Pipeline):
        return type(estimator.steps[-1][1]).__name__
    return type(estimator).__name__


def evaluate(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: "Union[pd.Series[int], np.ndarray]",
    n_splits: int = 5,
    scoring: str = "accuracy",
    random_state: int = RANDOM_STATE,
) -> "dict[str, float]":
    """Score ``estimator`` with stratified k-fold cross-validation.

    Stratification keeps the ~38 % survival rate constant across folds, which
    matters because the classes are imbalanced.

    Returns
    -------
    dict[str, float]
        ``{"mean": ..., "std": ...}`` of the per-fold scores.
    """
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    result = {"mean": float(scores.mean()), "std": float(scores.std())}
    _logger.info(
        "%s: %s = %.4f +/- %.4f",
        _describe(estimator),
        scoring,
        result["mean"],
        result["std"],
    )
    return result


def fit_predict(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: "Union[pd.Series[int], np.ndarray]",
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Fit on the full training set and predict the test set labels."""
    estimator.fit(X_train, y_train)  # type: ignore[attr-defined]
    return np.asarray(estimator.predict(X_test))  # type: ignore[attr-defined]


def save_submission(
    passenger_id: "Union[pd.Series[int], np.ndarray]",
    y_pred: "Union[pd.Series[int], np.ndarray, Any]",
    name: str,
    dirpath: Optional[Path] = None,
) -> Path:
    """Write a Kaggle-ready ``PassengerId,Survived`` csv.

    Parameters
    ----------
    passenger_id : pd.Series or np.ndarray
        The ``PassengerId`` column of ``test.csv``, in its original order.
    y_pred : pd.Series or np.ndarray
        Predicted labels, cast to ``int`` before writing.
    name : str
        File stem, e.g. ``"submission_baseline"``.
    dirpath : Path, optional
        Destination directory, by default `config.DIR_SUBMISSIONS`.

    Returns
    -------
    Path
        Path of the written csv.

    Raises
    ------
    ValueError
        If the row count is wrong or a label is outside ``{0, 1}``. Kaggle
        rejects such files, so fail here rather than after the upload.
    """
    dirpath = DIR_SUBMISSIONS if dirpath is None else Path(dirpath)

    submission = pd.DataFrame(
        {
            "PassengerId": np.asarray(passenger_id).astype(int),
            "Survived": np.asarray(y_pred).astype(int),
        }
    )

    if len(submission) != N_ROWS_SUBMISSION:
        raise ValueError(
            f"A Titanic submission needs {N_ROWS_SUBMISSION} rows, "
            f"got {len(submission)}."
        )
    invalid = set(submission["Survived"].unique()) - {0, 1}
    if invalid:
        raise ValueError(
            f"'Survived' must be 0 or 1, found {sorted(invalid)}."
        )

    dirpath.mkdir(parents=True, exist_ok=True)
    filepath = dirpath / f"{name}.csv"
    submission.to_csv(filepath, index=False)
    _logger.info(
        "Wrote %s (survival rate %.3f)",
        filepath,
        submission["Survived"].mean(),
    )
    return filepath
