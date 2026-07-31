## プロジェクトの作成

```bash
sphinx-quickstart docs_src
```

```plaintext
Welcome to the Sphinx 8.2.3 quickstart utility.

Please enter values for the following settings (just press Enter to
accept a default value, if one is given in brackets).

Selected root path: docs_src

You have two options for placing the build directory for Sphinx output.
Either, you use a directory "_build" within the root path, or you separate
"source" and "build" directories within the root path.
> Separate source and build directories (y/n) [n]: n

The project name will occur in several places in the built documentation.
> Project name: kaggle-template
> Author name(s): yu9824
> Project release []: 0.0.1

If the documents are to be written in a language other than English,
you can select a language here by its language code. Sphinx will then
translate text that it generates into that language.

For a list of supported codes, see
https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-language.
> Project language [en]: en

Creating file /home/yu9824/projects/kaggle-template/docs_src/conf.py.
Creating file /home/yu9824/projects/kaggle-template/docs_src/index.rst.
Creating file /home/yu9824/projects/kaggle-template/docs_src/Makefile.
Creating file /home/yu9824/projects/kaggle-template/docs_src/make.bat.

Finished: An initial directory structure has been created.

You should now populate your master file /home/yu9824/projects/kaggle-template/docs_src/index.rst and create other documentation
source files. Use the Makefile to build the docs, like so:
   make builder
where "builder" is one of the supported builders, e.g. html, latex or linkcheck.

```

```bash
rm -r docs_src/_build
touch docs_src/_static/.gitkeep
```

## docstringから生成

```bash
sphinx-apidoc -f -o ./docs_src ./src/kaggle_template --module-first
```

## index.rst を index.md へ

````markdown
```{toctree}
:maxdepth: 2
:caption: API Reference

modules
```
````


## build

```bash
# single version
sphinx-build -b html ./docs_src ./docs
# multible version
sphinx-multiversion ./docs_src ./docs
```

## favicon, logoの設定

以下をアンコメントしてパスを適切に設定すればよい

```python
# favicon
# https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/branding.html
# html_favicon = "_static/favicon.png"

# テーマのオプション設定
# https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/branding.html
html_theme_options = {
    "show_toc_level": 2,  # TOCの表示レベル（見出しの深さ、1-3の範囲）
    # site logo
    # "logo": {
    #     "image_light": "_static/site_logo.png",
    #     "image_dark": "_static/site_logo_dark.png",
    # },
}

```
