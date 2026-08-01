"""Download and load the Kaggle Titanic competition data.

The download is delegated to the ``kaggle`` command line client so that the
Python API and the documented shell workflow stay in sync.

"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import COMPETITION, DIR_DATA_RAW
from .helpers import is_installed
from .logging import get_child_logger

__all__ = (
    "download_competition_data",
    "load_gender_submission",
    "load_test",
    "load_train",
)

_logger = get_child_logger(__name__)

# Files shipped by the competition, extracted from ``titanic.zip``.
FILENAME_TRAIN = "train.csv"
FILENAME_TEST = "test.csv"
FILENAME_GENDER_SUBMISSION = "gender_submission.csv"


def _resolve_kaggle_command() -> "list[str]":
    """Locate the Kaggle CLI.

    ``shutil.which`` misses the client when the interpreter runs from a
    non-activated environment (a Jupyter kernel, for instance), so the
    interpreter's own ``bin/`` directory and ``python -m kaggle`` are tried
    as well.
    """
    executable = shutil.which("kaggle")
    if executable is not None:
        return [executable]

    sibling = Path(sys.executable).parent / "kaggle"
    if sibling.is_file():
        return [str(sibling)]

    if is_installed("kaggle"):
        return [sys.executable, "-m", "kaggle"]

    raise FileNotFoundError(
        "The Kaggle CLI was not found. Install it with `pip install kaggle` "
        "and place your API token at ~/.kaggle/kaggle.json "
        "(https://www.kaggle.com/settings > API > Create New Token)."
    )


def download_competition_data(
    dirpath: Optional[Path] = None, force: bool = False
) -> Path:
    """Download and unzip the competition data via the Kaggle CLI.

    ``kaggle competitions download`` (CLI v2) always writes a single zip
    archive; unlike older releases it has no ``--unzip`` flag, so the archive
    is expanded here.

    Parameters
    ----------
    dirpath : Path, optional
        Destination directory, by default `config.DIR_DATA_RAW`.
    force : bool, optional
        Re-download even if the data is already present, by default False.

    Returns
    -------
    Path
        The directory containing the extracted csv files.

    Raises
    ------
    FileNotFoundError
        If the Kaggle CLI cannot be located.
    subprocess.CalledProcessError
        If the Kaggle CLI exits with a non-zero status. Its stderr is logged
        before the exception propagates.
    """
    dirpath = DIR_DATA_RAW if dirpath is None else Path(dirpath)

    if not force and (dirpath / FILENAME_TRAIN).is_file():
        _logger.info("Already downloaded, skipping: %s", dirpath)
        return dirpath

    dirpath.mkdir(parents=True, exist_ok=True)
    command = _resolve_kaggle_command() + [
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(dirpath),
    ]
    if force:
        command.append("--force")

    _logger.info("Running: %s", " ".join(command))
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        _logger.error("Kaggle CLI failed: %s", completed.stderr.strip())
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )

    filepath_zip = dirpath / f"{COMPETITION}.zip"
    if not filepath_zip.is_file():
        raise FileNotFoundError(
            f"Expected the Kaggle CLI to produce '{filepath_zip}', "
            "but it does not exist."
        )

    with zipfile.ZipFile(filepath_zip) as zf:
        zf.extractall(dirpath)
    _logger.info("Extracted %s into %s", filepath_zip.name, dirpath)

    return dirpath


def _read_csv(filename: str, dirpath: Optional[Path] = None) -> pd.DataFrame:
    """Read one of the raw competition csv files with a helpful error."""
    dirpath = DIR_DATA_RAW if dirpath is None else Path(dirpath)
    filepath = dirpath / filename
    if not filepath.is_file():
        raise FileNotFoundError(
            f"'{filepath}' does not exist. Download the competition data "
            "first, e.g. `titanic.data.download_competition_data()` or "
            f"`kaggle competitions download -c {COMPETITION} -p {dirpath}`."
        )
    return pd.read_csv(filepath)


def load_train(dirpath: Optional[Path] = None) -> pd.DataFrame:
    """Load ``train.csv`` (891 labelled passengers)."""
    return _read_csv(FILENAME_TRAIN, dirpath)


def load_test(dirpath: Optional[Path] = None) -> pd.DataFrame:
    """Load ``test.csv`` (418 passengers to predict)."""
    return _read_csv(FILENAME_TEST, dirpath)


def load_gender_submission(dirpath: Optional[Path] = None) -> pd.DataFrame:
    """Load ``gender_submission.csv``, the official example submission."""
    return _read_csv(FILENAME_GENDER_SUBMISSION, dirpath)
