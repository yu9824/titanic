# kaggle 上のカーネルを使用する方法

1. コンペページ ＞ Code > 新しいノートブックの作成
  - コンペページから作成することで、データセットの attatch がされた状態からスタート。あとからでもできるみたいではある。
2. ノートブックの名前をいい感じに変える
3. `kaggle kernels pull <userid>/<notebookname> -m -p ./notebooks` # -m: メタデータもpullする。pushするときに必要。 -p: ディレクトリ指定。by default .
4. ローカルでプログラムを実行・編集
5. `kaggle kernels push -p ./notebooks`
