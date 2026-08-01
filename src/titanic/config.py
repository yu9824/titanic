"""Project-wide paths and competition constants.

All paths are anchored to the repository root so that notebooks under
``notebooks/`` and scripts under ``src/`` resolve to the same locations
regardless of the current working directory.

"""

from pathlib import Path

__all__ = (
    "COMPETITION",
    "DIR_DATA",
    "DIR_DATA_PROCESSED",
    "DIR_DATA_RAW",
    "DIR_MODELS",
    "DIR_PROJECT_ROOT",
    "DIR_SUBMISSIONS",
    "RANDOM_STATE",
    "TARGET",
)

# src/titanic/config.py -> src/titanic -> src -> <repository root>
DIR_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DIR_DATA = DIR_PROJECT_ROOT / "data"
DIR_DATA_RAW = DIR_DATA / "raw"
DIR_DATA_PROCESSED = DIR_DATA / "processed"
DIR_SUBMISSIONS = DIR_DATA_PROCESSED / "submissions"
DIR_MODELS = DIR_PROJECT_ROOT / "models"

# Competition URL suffix, i.e. https://www.kaggle.com/competitions/titanic
COMPETITION = "titanic"

TARGET = "Survived"
RANDOM_STATE = 0
