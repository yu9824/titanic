"""Feature engineering for the Titanic dataset.

Every function here is pure: the input :class:`~pandas.DataFrame` is never
mutated, a new object is returned instead.

"""

from typing import Union

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = (
    "CATEGORICAL_FEATURES",
    "CATEGORICAL_FEATURES_ENGINEERED",
    "NUMERIC_FEATURES",
    "NUMERIC_FEATURES_ENGINEERED",
    "add_features",
    "build_preprocessor",
    "get_feature_names",
)

# Columns available straight from the competition csv files.
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]

# Columns produced by `add_features`.
NUMERIC_FEATURES_ENGINEERED = ["FamilySize", "IsAlone", "FarePerPerson"]
CATEGORICAL_FEATURES_ENGINEERED = ["Title", "CabinDeck", "TicketPrefix"]

# French/other spellings that mean the same thing as a common English title.
_TITLE_ALIASES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
_TITLE_COMMON = frozenset({"Mr", "Miss", "Mrs", "Master"})


def _extract_title(name: "pd.Series[str]") -> "pd.Series[str]":
    """Pull the honorific out of ``"Braund, Mr. Owen Harris"`` -> ``"Mr"``."""
    title = name.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = title.replace(_TITLE_ALIASES)
    return title.where(title.isin(_TITLE_COMMON), "Rare")


def _extract_ticket_prefix(ticket: "pd.Series[str]") -> "pd.Series[str]":
    """Keep the alphabetic part of a ticket number, e.g. ``"PC 17599"``."""
    prefix = (
        ticket.str.replace(r"[\.\/\s]", "", regex=True)
        .str.replace(r"\d+", "", regex=True)
        .str.upper()
    )
    # Purely numeric tickets have no prefix; label them explicitly.
    return prefix.replace("", "NONE")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features to a copy of ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw competition frame, i.e. the output of `data.load_train` or
        `data.load_test`.

    Returns
    -------
    pd.DataFrame
        A new frame with `NUMERIC_FEATURES_ENGINEERED` and
        `CATEGORICAL_FEATURES_ENGINEERED` appended. ``df`` is left untouched.
    """
    family_size = df["SibSp"] + df["Parch"] + 1

    return df.assign(
        Title=_extract_title(df["Name"]),
        FamilySize=family_size,
        IsAlone=(family_size == 1).astype(int),
        # `Fare` is the price of the whole ticket, not per passenger.
        FarePerPerson=df["Fare"] / family_size,
        CabinDeck=df["Cabin"].str[0].fillna("U"),
        TicketPrefix=_extract_ticket_prefix(df["Ticket"]),
    )


def get_feature_names(use_engineered: bool) -> "list[str]":
    """Return the columns consumed by `build_preprocessor`."""
    numeric, categorical = _select_columns(use_engineered)
    return numeric + categorical


def _select_columns(
    use_engineered: bool,
) -> "tuple[list[str], list[str]]":
    if use_engineered:
        return (
            NUMERIC_FEATURES + NUMERIC_FEATURES_ENGINEERED,
            CATEGORICAL_FEATURES + CATEGORICAL_FEATURES_ENGINEERED,
        )
    return list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES)


def build_preprocessor(
    use_engineered: bool = False,
    scale: bool = True,
    min_frequency: Union[int, float, None] = 5,
) -> ColumnTransformer:
    """Build the impute + encode step shared by every model.

    Parameters
    ----------
    use_engineered : bool, optional
        Include the columns created by `add_features`, by default False.
        Set to False to reproduce the baseline notebook.
    scale : bool, optional
        Standardise the numeric columns, by default True. Required for
        `~sklearn.linear_model.LogisticRegression`, useless for trees.
    min_frequency : int or float or None, optional
        Categories rarer than this are folded into a single "infrequent"
        level, by default 5. Keeps high-cardinality columns such as
        ``TicketPrefix`` from exploding into hundreds of dummies.

    Returns
    -------
    ColumnTransformer
        Outputs a dense array, so it can feed any scikit-learn estimator.
    """
    numeric, categorical = _select_columns(use_engineered)

    numeric_steps: "list[tuple[str, object]]" = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
