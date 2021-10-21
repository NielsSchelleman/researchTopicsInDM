""" Utils mainly to write code agnostic to numpy or pandas.  """
# Author: Davina Zamanzadeh <davzaman@gmail.com>

from typing import List, Union
import logging
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

ArrayLike = Union[pd.Series, np.array, List]
Matrix = Union[pd.DataFrame, np.ndarray]


def setup_logging(log_filename: str = "output.log"):
    # Ref: https://stackoverflow.com/a/46098711/1888794
    # prints to console and saves to File.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )


def standardize_uppercase(input: str) -> str:
    """Standardize string to upper case."""
    return input.upper()


def sigmoid_scores(
    wss_standardized: ArrayLike, shift_amount: float, cutoff_type: str
) -> ArrayLike:
    """
    Applies sigmoid to standardized weighted sum scores to conver to probability.
    Right: Regular sigmoid pushes larger values to have high probability,
    Left: To flip regular sigmoid across y axis, make input negative.
        This pushes smaller values to have high probability.
    We apply similar tricks for mid and tail, shifting appropriately.
    """

    cutoff_transformations = {
        "RIGHT": lambda wss_standardized, b: wss_standardized + b,
        "LEFT": lambda wss_standardized, b: -wss_standardized + b,
        "TAIL": lambda wss_standardized, b: (np.absolute(wss_standardized) - 0.75 + b),
        "MID": lambda wss_standardized, b: (-np.absolute(wss_standardized) + 0.75 + b),
    }

    return sigmoid(cutoff_transformations[cutoff_type](wss_standardized, shift_amount))


def sigmoid(X: ArrayLike) -> ArrayLike:
    return 1 / (1 + np.exp(-X))


def isnan(X: Matrix):
    # ref: https://stackoverflow.com/a/29530601/1888794
    if isinstance(X, pd.DataFrame):
        return X.isnull().values
    # else np.ndarray
    return np.isnan(X)


def isin(X: Union[ArrayLike, Matrix], list: ArrayLike) -> bool:
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.isin(list)
    return np.isin(X, list)


def is_numeric(X: Union[ArrayLike, Matrix]) -> bool:
    if isinstance(X, pd.DataFrame):
        return is_numeric_dtype(X.values)

    return is_numeric_dtype(X)


def enforce_numeric(X: Union[ArrayLike, Matrix]) -> Matrix:
    if isinstance(X, pd.DataFrame):
        X = X.apply(pd.to_numeric, errors="coerce")
    elif isinstance(X, np.ndarray):
        X = np.array(list(map(pd.to_numeric, X)))
    else:
        X = pd.to_numeric(X, errors="coerce")

    return X


def missingness_profile(X: Matrix):
    nans = isnan(X)

    def percentify(n: int, axis: int) -> float:
        """Take number and report as a percent of axis."""
        return n / X.shape[axis] * 100

    # By row
    entries_missing = nans.any(axis=1).sum()
    print(
        "Entries missing a value: "
        f"{entries_missing} ({percentify(entries_missing, 0)}%)"
    )

    # By column
    print("Features missing | %")
    feature_missing = nans.sum(axis=0)
    percent_feature_missing = percentify(feature_missing, 0)
    #print(feature_missing, 'v',percent_feature_missing)
    #print(np.concatenate(feature_missing, percent_feature_missing))


def ord_to_interval(column: pd.Series,
                    column_name: str = 'ordinal',
                    order: list = None,
                    proportion_based: bool = False,
                    relation=lambda x: x,
                    ):
    if not order:
        order = list(set(column))

    if proportion_based:
        counts = dict(column.value_counts())
    else:
        counts = dict.fromkeys(set(column), floor(len(column) / len(set(column))))

    init = 0
    ranges_start = {}
    ranges_end = {}

    for i in order:
        compute = relation(counts[i])
        ranges_start[i] = init
        ranges_end[i] = init + compute
        init = init + compute

    return pd.DataFrame([column.map(ranges_start), column.map(ranges_end)],
                        ['_start_' + column_name, '_end_' + column_name]).transpose()