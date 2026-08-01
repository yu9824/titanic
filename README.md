# titanic

<!-- badges -->
[![CI](https://github.com/yu9824/titanic/actions/workflows/CI.yml/badge.svg)](https://github.com/yu9824/titanic/actions/workflows/CI.yml)
[![release-pypi](https://github.com/yu9824/titanic/actions/workflows/release.yml/badge.svg)](https://github.com/yu9824/titanic/actions/workflows/release.yml)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://github.com/python/mypy)
<!-- /badges -->

Kaggle [Titanic](https://www.kaggle.com/competitions/titanic) コンペのワークスペース。
データ取得 → EDA → モデリング → 提出までの一連の流れを、他コンペにも流用できる形で整理したもの。

手順とコマンドの全体像は **[_notes/kaggle-workflow.md](_notes/kaggle-workflow.md)** にまとめてある。

## Quick start

```bash
conda activate kaggle311
pip install -e ".[kaggle,viz,gbdt,notebook,test]"

# ~/.kaggle/kaggle.json の配置とコンペの Rules 同意が必要
kaggle competitions download -c titanic -p data/raw
unzip -o data/raw/titanic.zip -d data/raw/

jupytext --sync notebooks/*.py   # .py から .ipynb を生成
```

提出:

```bash
kaggle competitions submit -c titanic \
  -f data/processed/submissions/submission_lgbm.csv \
  -m "LGBM tuned + FE, CV acc=0.832"
```

## Structure

```
notebooks/          # Jupytext で .ipynb と .py がペア（.py が git の実体）
  01_eda.py                  データ取得・欠損・ターゲットとの関係
  02_baseline.py             生の列だけで最小構成 → submission_baseline.csv
  03_feature_engineering.py  特徴量追加 + LightGBM → submission_lgbm.csv
src/titanic/
  config.py         パス定数・コンペ名
  data.py           Kaggle CLI 経由のダウンロードと読み込み
  features.py       add_features / build_preprocessor
  modeling.py       evaluate (CV) / fit_predict / save_submission
data/               .gitignore 対象（raw/ に生データ、processed/submissions/ に提出ファイル）
_notes/             手順メモ
```

## Results

| CV (accuracy) | 手法 |
| --- | --- |
| 0.796 | LogisticRegression, 生の列のみ |
| 0.810 | RandomForest, 生の列のみ |
| 0.832 | LightGBM (tuned) + 特徴量エンジニアリング |

5-fold `StratifiedKFold`。Titanic は 891 行しかなく標準偏差が 0.02 前後あるため、
0.005 程度の差は誤差の範囲。

## Development

```bash
ruff check . && ruff format .
mypy src/
pytest -q
```
