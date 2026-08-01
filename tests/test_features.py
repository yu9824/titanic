import numpy as np
import pandas as pd
import pytest

from titanic.features import (
    CATEGORICAL_FEATURES_ENGINEERED,
    NUMERIC_FEATURES_ENGINEERED,
    add_features,
    build_preprocessor,
    get_feature_names,
)
from titanic.modeling import N_ROWS_SUBMISSION, save_submission


@pytest.fixture
def df_raw() -> pd.DataFrame:
    """A miniature stand-in for ``train.csv``, including the awkward cases."""
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived": [0, 1, 1, 0],
            "Pclass": [3, 1, 3, 2],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Mlle. Laina",
                "Palsson, Master. Gosta Leonard",
            ],
            "Sex": ["male", "female", "female", "male"],
            "Age": [22.0, 38.0, np.nan, 2.0],
            "SibSp": [1, 1, 0, 3],
            "Parch": [0, 0, 0, 1],
            "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "349909"],
            "Fare": [7.25, 71.28, 7.925, 21.075],
            "Cabin": [None, "C85", None, None],
            "Embarked": ["S", "C", "S", "S"],
        }
    )


def test_add_features_does_not_mutate_input(df_raw: pd.DataFrame) -> None:
    before = df_raw.copy(deep=True)
    add_features(df_raw)
    pd.testing.assert_frame_equal(df_raw, before)


def test_add_features_appends_expected_columns(df_raw: pd.DataFrame) -> None:
    result = add_features(df_raw)

    expected = NUMERIC_FEATURES_ENGINEERED + CATEGORICAL_FEATURES_ENGINEERED
    assert set(expected) <= set(result.columns)
    assert list(df_raw.columns) == list(result.columns)[: len(df_raw.columns)]


def test_title_normalises_aliases_and_rare(df_raw: pd.DataFrame) -> None:
    # "Mlle" is folded into "Miss"; "Master" is kept as its own title.
    assert add_features(df_raw)["Title"].tolist() == [
        "Mr",
        "Mrs",
        "Miss",
        "Master",
    ]


def test_family_and_fare_features(df_raw: pd.DataFrame) -> None:
    result = add_features(df_raw)

    assert result["FamilySize"].tolist() == [2, 2, 1, 5]
    assert result["IsAlone"].tolist() == [0, 0, 1, 0]
    assert result["FarePerPerson"].iloc[0] == pytest.approx(7.25 / 2)


def test_cabin_deck_and_ticket_prefix(df_raw: pd.DataFrame) -> None:
    result = add_features(df_raw)

    # Missing cabins become "U" rather than NaN.
    assert result["CabinDeck"].tolist() == ["U", "C", "U", "U"]
    # A purely numeric ticket has no prefix.
    assert result["TicketPrefix"].tolist() == ["A", "PC", "STONO", "NONE"]


@pytest.mark.parametrize("use_engineered", [False, True])
def test_preprocessor_outputs_dense_array_without_nan(
    df_raw: pd.DataFrame, use_engineered: bool
) -> None:
    df = add_features(df_raw) if use_engineered else df_raw
    X = df[get_feature_names(use_engineered)]

    transformed = build_preprocessor(
        use_engineered=use_engineered, min_frequency=None
    ).fit_transform(X)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(df)
    assert not np.isnan(transformed).any()


def test_save_submission_writes_kaggle_format(tmp_path) -> None:
    passenger_id = np.arange(892, 892 + N_ROWS_SUBMISSION)
    y_pred = np.zeros(N_ROWS_SUBMISSION)

    filepath = save_submission(
        passenger_id, y_pred, name="submission_test", dirpath=tmp_path
    )

    written = pd.read_csv(filepath)
    assert list(written.columns) == ["PassengerId", "Survived"]
    assert len(written) == N_ROWS_SUBMISSION
    assert written["Survived"].dtype.kind == "i"


def test_save_submission_rejects_wrong_row_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="rows"):
        save_submission([1, 2], [0, 1], name="broken", dirpath=tmp_path)


def test_save_submission_rejects_non_binary_labels(tmp_path) -> None:
    passenger_id = np.arange(N_ROWS_SUBMISSION)
    y_pred = np.full(N_ROWS_SUBMISSION, 2)

    with pytest.raises(ValueError, match="0 or 1"):
        save_submission(passenger_id, y_pred, name="broken", dirpath=tmp_path)
