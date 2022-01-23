# <div align=center style="line-height:1.5;font-size:40;"><b>Accelerated-TD</b></div>

加速勾配時間差学習アルゴリズム（Accelerated Gradient Temporal Difference Learning algorithm, ATD）のPython実装。

<img src="translations.svg" style="height:25px" />

![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/VEXLife/Accelerated-TD) ![GitHub](https://img.shields.io/github/license/VEXLife/Accelerated-TD) ![GitHub issues](https://img.shields.io/github/issues/VEXLife/Accelerated-TD)  ![Gitee issues](https://img.shields.io/badge/dynamic/json?label=Gitee%20Issues&query=%24.open_issues_count&url=http%3A%2F%2Fgitee.com%2Fapi%2Fv5%2Frepos%2FVEXLife%2FAccelerated-TD&logo=gitee&style=flat) ![GitHub Pull requests](https://img.shields.io/github/issues-pr/VEXLife/Accelerated-TD) ![Contribution](https://img.shields.io/static/v1?label=contribution&message=welcome&color=%23ff66ef&link=http://github.com/VEXLife/Accelerated-TD) ![GitHub Repo stars](https://img.shields.io/github/stars/VEXLife/Accelerated-TD?style=social) ![GitHub forks](https://img.shields.io/github/forks/VEXLife/Accelerated-TD?style=social) ![Gitee Repo stars](https://img.shields.io/badge/dynamic/json?label=Gitee%20Stars&query=%24.stargazers_count&url=http%3A%2F%2Fgitee.com%2Fapi%2Fv5%2Frepos%2FVEXLife%2FAccelerated-TD&logo=gitee&style=social) ![Gitee forks](https://img.shields.io/badge/dynamic/json?label=Gitee%20Forks&query=%24.forks_count&url=http%3A%2F%2Fgitee.com%2Fapi%2Fv5%2Frepos%2FVEXLife%2FAccelerated-TD&logo=gitee&style=social)

# 紹介

## エージェント

`PlainATDAgent` は ![](https://latex.codecogs.com/svg.image?\mathbf{A}) 行列を直接更新し、`SVDATDAgent` と `DiagonalizedSVDATDAgent` はそれぞれ特異値分解を更新します。これは、論文の著者（以下の論文へのリンク）にはそれほど複雑ではないように見えます。 `SVDATDAgent` と `DiagonalizedSVDATDAgent` の違いは、 `SVDATDAgent` がここで説明されている方法を採用していることです：[Brand 2006](https://pdf.sciencedirectassets.com/271586/1-s2.0-S0024379506X04573/1-s2.0-S0024379505003812/main.pdf) および `DiagonalizedSVDATDAgent` は、ここで説明した方法[Gahring 2015](https://arxiv.org/pdf/1511.08495) を採用して対角化します ![](https://latex.codecogs.com/svg.image?\mathbf{\Sigma}) 行列を使用して、行列の疑似逆行列を計算しやすくします。私はこの方法を完全には理解していませんが。
また、 `TDAgent`と呼ばれる従来の勾配時間差エージェントを実装しました。以下に説明するいくつかの環境でそれらをテストしました。

## バックエンドのサポート

PyTorch（CPU）のバックエンドをサポートして、 `numpy.ndarray`から`torch.Tensor`への変換プロセスをスキップします。このサポートを使用するには、 `atd`モジュールをインポートする前に次のコードを追加できます：
```python
import os
os.environ["ATD_BACKEND"] = "NumPy"  # 又は "PyTorch"
```

自分でテストを実行する場合は、このリポジトリのクローンを作成して、 `python algorithm_test/<random_walk 又は boyans_chain>.py`を実行します。:)

# 必須

- Python>=3.9
- NumPy>=1.19
- PyTorchをバックエンドとして使用する場合は、Torch>=1.10も必要です。
- テストスクリプトを実行する場合は、Matplotlib>=3.3.3も必要です。
- テストスクリプトを実行する場合は、Tqdmも必要です。

# テスト

## ランダムウォーク（Random Walk）

この環境は[サットンの本](http://incompleteideas.net/book/RLbook2020.pdf)からのものです。

コードファイルは[これ](https://github.com/VEXLife/Accelerated-TD/blob/main/algorithm_test/random_walk.py)で、結果は[ここ](https://github.com/VEXLife/加速-TD/blob/main/figures/random_walk.png)：
![random_walk](https://user-images.githubusercontent.com/36587232/144394107-d0cf9bd0-ea09-4e48-ba04-cb07af9e4240.png)

## ボヤンのチェーン（Boyan's Chain）

この環境は、[Boyan 1999](https://www.researchgate.net/publication/2621189_Least-Squares_Temporal_Difference_Learning) に示されています。

コードファイルは[これ](https://github.com/VEXLife/Accelerated-TD/blob/main/algorithm_test/boyans_chain.py)で、結果は[ここ](https://github.com/VEXLife/Accelerated-TD/blob/main/figures/boyans_chain.png)：
![boyans_chain](https://user-images.githubusercontent.com/36587232/144394100-dc8c48c2-1d38-433f-aea6-f202da3bbb13.png)

# 手順

アルゴリズムの実装をプロジェクトにインポートするには、慣れていない場合は次の手順に従ってください。
1. リポジトリのクローンを作成し、 `atd.py`を目的の場所にコピーします。 GitHub から .zip ファイルをダウンロードした場合は、忘れずに解凍してください。
2. 次のコードをPythonスクリプトの先頭に追加します。
   ```python
   from atd import TDAgent, SVDATDAgent, DiagonalizedSVDATDAgent, PlainATDAgent # または任意のエージェントから
   ```
3. ターゲットディレクトリが、実行するPythonメインプログラムファイルが配置されているディレクトリと異なる場合は、手順2のコードスニペットの代わりにこのコードスニペットを使用して、適切なディレクトリを環境変数に追加し、Pythonが通訳はそれを見つけることができます。または、Pythonの新しいバージョンで提供されている `importlib`を検討することもできます。
   ```python
   import sys

   sys.path.append("<atd.pyを配置したディレクトリ>")
   from atd import TDAgent, SVDATDAgent, DiagonalizedSVDATDAgent, PlainATDAgent # または任意のエージェントから
   ```
4. 次のようなエージェントを初期化すると、次のように使用できます！
   ```python
   agent = TDAgent(lr=0.01, lambd=0, observation_space_n=4, action_space_n=2)
   ```

参照：[Gahring 2016](https://arxiv.org/pdf/1611.09328.pdf)

プルリクエストについてお気軽にご連絡ください。コメントをお待ちしております。
