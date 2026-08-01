# Kaggle ワークフロー（Titanic で確立した型）

コンペに参加してから提出・スコア確認までの手順。
Titanic を例にしているが、`titanic` の部分を差し替えれば他コンペにも流用できる。

前提: conda 環境 `kaggle311` を使う。`.vscode/settings.json` もこの環境を参照している。

---

## 0. 初回セットアップ

### 0-1. API トークンの配置

[Kaggle > Settings](https://www.kaggle.com/settings) > API > **Create New Token** で
`kaggle.json` がダウンロードされるので、以下に置く。

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json   # 権限が緩いと CLI が警告を出す
```

> **`kaggle.json` は絶対にコミットしない。** API キーそのもの。
> このリポジトリの `.gitignore` は `~/.kaggle` を対象にしていないが、
> リポジトリ内にコピーを置かないこと。

### 0-2. 環境構築

```bash
conda activate kaggle311
pip install -e ".[kaggle,viz,gbdt,notebook,test]"
```

| extra | 中身 | 用途 |
| --- | --- | --- |
| `kaggle` | kaggle | データ取得・提出 |
| `viz` | matplotlib, seaborn | EDA |
| `gbdt` | lightgbm | 勾配ブースティング |
| `notebook` | jupyter, jupytext | ノートブック |
| `test` | pytest | テスト |

### 0-3. 動作確認

```bash
kaggle --version
kaggle competitions files -c titanic
```

ここで `403 Forbidden` が出る場合は、**コンペページで Rules に同意していない**。
ブラウザで https://www.kaggle.com/competitions/titanic/rules を開き
"I Understand and Accept" を押す（CLI からは同意できない）。

---

## 1. データ取得

```bash
kaggle competitions download -c titanic -p data/raw
unzip -o data/raw/titanic.zip -d data/raw/
```

> **注意: Kaggle CLI v2 の `competitions download` に `--unzip` は無い。**
> （`-f/-p/-w/-o/-q` のみ。ネット上の記事の多くは旧 CLI 前提。）
> 常に `<competition>.zip` が 1 つ落ちてくるので自分で展開する。

Python から実行する場合（unzip まで面倒を見る）:

```bash
python -c "from titanic.data import download_competition_data as d; d()"
```

`data/` は `.gitignore` 済みなので、生データも提出ファイルもコミットされない。

| コマンド | 用途 |
| --- | --- |
| `kaggle competitions list` | 参加できるコンペ一覧 |
| `kaggle competitions files -c <comp>` | 配布ファイルの一覧 |
| `kaggle competitions download -c <comp> -p <dir>` | ダウンロード（zip） |
| `kaggle competitions download -c <comp> -f train.csv -p <dir>` | 単一ファイルのみ |
| `kaggle datasets download -d <user>/<dataset>` | コンペ外のデータセット |

---

## 2. 分析・モデリング

ノートブックは Jupytext で `.ipynb` と `.py` (py:percent) がペアになっている
（[jupytext.toml](../jupytext.toml)）。**`.py` の方が git の差分を読みやすい。**

```bash
# .py しか無い状態から .ipynb を作る（clone 直後など）
jupytext --sync notebooks/*.py
```

VSCode で `.ipynb` を保存すると `.py` へ自動同期される
（[.vscode/settings.json](../.vscode/settings.json) の `emeraldwalk.runonsave`）。

| ノートブック | 内容 |
| --- | --- |
| `notebooks/01_eda.py` | データ取得・欠損・ターゲットとの関係 |
| `notebooks/02_baseline.py` | 生の列だけで最小構成 → `submission_baseline.csv` |
| `notebooks/03_feature_engineering.py` | 特徴量追加 + LightGBM → `submission_lgbm.csv` |

再利用するロジックは `src/titanic/` 側に置く。

| モジュール | 役割 |
| --- | --- |
| `titanic.config` | パス定数（`DIR_DATA_RAW`, `DIR_SUBMISSIONS` など） |
| `titanic.data` | ダウンロード・読み込み |
| `titanic.features` | `add_features`, `build_preprocessor` |
| `titanic.modeling` | `evaluate`（CV）, `fit_predict`, `save_submission` |

ノートブックを非対話で流し直す場合:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

### モデリングで守る型

- **前処理は必ず `Pipeline` に入れる。** 先に全データへ `fit` すると
  CV の各 fold に検証データの情報が漏れ、スコアが実力より高く出る。
- **`StratifiedKFold` で層化する。** `Survived` は 1 が約 38% しかない。
- **CV の `std` を見る。** Titanic は 891 行しかなく `std` が 0.02〜0.03 出るので、
  0.005 程度の差は誤差。
- **提出前に検証する。** `save_submission` が行数（418）と
  ラベル（0/1）をチェックしてから書き出す。

---

## 3. 提出

```bash
kaggle competitions submit -c titanic \
  -f data/processed/submissions/submission_lgbm.csv \
  -m "LGBM tuned + FE (Title/FamilySize/CabinDeck), CV acc=0.832"
```

- `-m` は必須。**手法と CV スコアを書く。** 後で Public LB と突き合わせるため。
- **提出は 1 日 10 回まで**（Titanic の場合）。無駄撃ちせず CV で絞る。

### 結果の確認

```bash
kaggle competitions submissions -c titanic          # 自分の提出履歴とスコア
kaggle competitions leaderboard -c titanic --show   # 上位の順位表
```

### Public LB の読み方

- Titanic の Public LB は test 418 件の **50%（約 209 件）**でしか計算されない。
  1 件正解が変わるだけで 0.005 動く。**LB の小さな上下に意味は無い。**
- LB に合わせてモデルを選ぶ（LB probing）と Private LB で崩れる。
  **CV を信じ、LB は「CV と大きく乖離していないか」の確認に使う。**
- Titanic は答えが公開されているため LB 上位（1.000）はカンニング。
  実質的な上限は 0.80〜0.83 程度。

### 記録の残し方

| CV | Public LB | 手法 |
| --- | --- | --- |
| 0.810 | | baseline: RandomForest, 生の列のみ |
| 0.832 | | LGBM tuned + FE |

この表を更新していくと、CV と LB の相関が見えて次の判断がしやすくなる。

---

## 4. 他コンペへの流用

```bash
kaggle competitions list                    # 参加できるコンペを探す
kaggle competitions list -s "house prices"  # キーワード検索
```

1. ブラウザでコンペページを開き **Rules に同意**（CLI では不可）
2. `src/titanic/config.py` の `COMPETITION` を差し替え
3. `titanic.data` の `FILENAME_*` と `titanic.modeling.N_ROWS_SUBMISSION` を
   そのコンペに合わせる
4. 評価指標を確認して `evaluate(..., scoring=...)` を変更
   （accuracy / roc_auc / neg_root_mean_squared_error など）

初心者向けの定番コンペ:

| コンペ | タスク | 指標 |
| --- | --- | --- |
| `titanic` | 二値分類 | accuracy |
| `house-prices-advanced-regression-techniques` | 回帰 | RMSLE |
| `spaceship-titanic` | 二値分類 | accuracy |
| `playground-series-s4e1` 等 (Playground) | 月替わり | 各種 |

---

## 5. 開発時のチェック

```bash
ruff check . && ruff format .
mypy src/
pytest -q
```
