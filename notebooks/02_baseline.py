# %% [markdown]
# # 02. ベースラインを作って提出する
#
# **このステップの狙いは精度ではなく「提出まで一周すること」。**
# 生の列だけを使った最小構成でモデルを組み、`submission.csv` を書き出す。
#
# ここで押さえる型:
#
# 1. 前処理はすべて `Pipeline` に入れる（`fit` は学習 fold 内でのみ行う）
# 2. 交差検証 (CV) でモデルを比較する — Kaggle の提出回数は 1 日 10 回まで
# 3. 全学習データで再学習してから `test` を予測し、提出ファイルを書く

# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from titanic import logging as titanic_logging
from titanic.config import RANDOM_STATE, TARGET
from titanic.data import load_test, load_train
from titanic.features import build_preprocessor, get_feature_names
from titanic.modeling import evaluate, fit_predict, save_submission

titanic_logging.enable_default_handler()
pd.set_option("display.max_columns", 50)

# %% [markdown]
# ## 1. データと使用する列
#
# `use_engineered=False` なので、csv にそのまま入っている列だけを使う。

# %%
df_train = load_train()
df_test = load_test()

feature_names = get_feature_names(use_engineered=False)
X_train = df_train[feature_names]
y_train = df_train[TARGET]
X_test = df_test[feature_names]

print(feature_names)
X_train.head()

# %% [markdown]
# ## 2. 前処理
#
# `build_preprocessor` が返す `ColumnTransformer` の中身:
#
# - 数値列 → 中央値で補完 → 標準化
# - カテゴリ列 → 最頻値で補完 → One-Hot エンコード
#
# **これをモデルと一緒に `Pipeline` に包むのが重要。**
# 先に全データへ `fit` してしまうと、CV の各 fold に検証データの情報が
# 漏れて（リーク）、CV スコアが実力より高く出る。

# %%
preprocessor = build_preprocessor(use_engineered=False)
preprocessor

# %% [markdown]
# ## 3. CV でモデルを比較
#
# `Survived` は約 38% しか 1 が無いので、
# 層化 (`StratifiedKFold`) して fold ごとの陽性比率を揃える。

# %%
candidates = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(
        n_estimators=500, random_state=RANDOM_STATE
    ),
}

results = {}
for name, model in candidates.items():
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(use_engineered=False)),
            ("model", model),
        ]
    )
    results[name] = evaluate(pipeline, X_train, y_train)

pd.DataFrame(results).T.sort_values("mean", ascending=False)

# %% [markdown]
# ばらつき (`std`) が 0.02〜0.03 あるので、
# **0.005 程度の差は誤差**と考えたほうがよい。

# %% [markdown]
# ## 4. 全データで再学習して提出ファイルを作る
#
# CV はあくまでモデル選択のための道具。
# 最終予測は 891 件すべてを使って学習し直したモデルで行う。

# %%
best_name = max(results, key=lambda name: results[name]["mean"])
best_pipeline = Pipeline(
    [
        ("preprocessor", build_preprocessor(use_engineered=False)),
        ("model", candidates[best_name]),
    ]
)
print(f"selected: {best_name}")

y_pred = fit_predict(best_pipeline, X_train, y_train, X_test)
filepath_submission = save_submission(
    df_test["PassengerId"], y_pred, name="submission_baseline"
)
filepath_submission

# %%
pd.read_csv(filepath_submission).head()

# %% [markdown]
# ## 5. 提出
#
# ターミナル（`conda activate kaggle311`）で以下を実行する。
# `-m` のメッセージには **CV スコアと手法**を書いておくと、
# 後で Public LB と突き合わせられる。
#
# ```bash
# kaggle competitions submit -c titanic \
#   -f data/processed/submissions/submission_baseline.csv \
#   -m "baseline: RandomForest, raw features, CV acc=0.81"
#
# kaggle competitions submissions -c titanic   # スコア確認
# ```
#
# → 次は [03_feature_engineering](03_feature_engineering.ipynb) で
# 特徴量を追加し、勾配ブースティングでスコアを伸ばす。
