# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# %%
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


# %%
DIRPATH_INPUT = Path("/kaggle/input")
DIRPATH_WORKING = Path("/kaggle/working")
DIRPATH_TEMP = Path("/kaggle/temp")

DIRPATH_TEMP.mkdir(exist_ok=True)
dirpath_dataset = DIRPATH_INPUT / "competitions/titanic"

# %%
KEY_Y = "Survived"

df_train = pd.read_csv(dirpath_dataset / "train.csv", index_col=0)
X_train = df_train.drop(KEY_Y, axis=1)
y_train = df_train.loc[:, KEY_Y]
display(X_train.head(), y_train.to_frame().head())

# %%
X_test = pd.read_csv(dirpath_dataset / "test.csv", index_col=0)
X_test.head()

# %% [markdown]
# Embarked: 乗船地

# %%
X_train.isna().sum()

# %%
X_test.isna().sum()


# %%
def extract_title(X) -> np.ndarray:
    X = np.asarray(X)
    titles = ("Mr.", "Mrs.", "Miss")

    return np.array(
        [[title in name for title in titles] for name in X[:, 0]],
        dtype=int,
    )


title_transformer = FunctionTransformer(extract_title)

# %%
preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(), ["Sex", "Embarked", "Pclass"]),
        ("num", SimpleImputer(), ["Age", "Fare"]),
        ("title", title_transformer, ["Name"]),
    ]
)

estimator = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
estimator.fit(X_train, y_train)


# %%
y_pred_on_train = estimator.predict(X_train)
accuracy_score(y_train, y_pred_on_train)

# %%
y_pred_on_test = estimator.predict(X_test)

# %%
sr_submit = pd.Series(y_pred_on_test, index=X_test.index, name=y_train.name)
sr_submit.to_frame().head()

# %%
sr_submit.to_csv(DIRPATH_WORKING / "submission.csv")
