# BERT Classification Tutorial 2024

本実装は[hppRC/bert-classification-tutorial](https://github.com/hppRC/bert-classification-tutorial)の2024版実装です。  
実装の背景や詳細についてはこちらのリポジトリをご覧ください。

以前の実装からの主な変更点は以下の通りです。

- 実装全体をHuggingFace関連ライブラリを利用するように変更
  - データセットの構築をHuggingFace Datasetsを利用するように変更
  - 訓練をHuggingFaceのTrainerとAccelerateを利用するように変更
- 仮想環境の構築にryeを利用するよう変更


## 実行手順

```bash
# 環境構築
rye sync -f
source .venv/bin/activate

# データセット作成
python src/prepare.py

# 訓練
accelerate launch --config_file config/4-ds.json src/train.py --model_name tohoku-nlp/bert-base-japanese-v3 --experiment_name 4-ds
```

## 補足

- `config`ディレクトリに`accelerate`利用時のconfigファイルを保存してあります
  - `4-ds.json`は4GPU+DeepSpeedを利用する場合の設定ファイルです
  - `1.json`は1GPUのみ利用する場合の設定ファイルです
  - `accelerate config --config_file config/hoge.json`を実行することでお好みの設定ファイルを対話的に作成することができます
- `tensorboard --logdir ./outputs`を実行することでTensorBoardを利用して学習の進捗を確認することができます

## おわりに

本実装が研究・企業応用・個人利用問わずさまざまな方のお役に立てれば幸いです。

質問・バグ報告などがあればどんなことでも[Issue](https://github.com/hppRC/bert-classification-tutorial-2024/issues)にお書きください。


## 著者情報・引用

作者: [Hayato Tsukagoshi](https://hpprc.dev) \
email: [research.tsukagoshi.hayato@gmail.com](mailto:research.tsukagoshi.hayato@gmail.com)
関連学会記事: [BERTによるテキスト分類チュートリアル](https://www.jstage.jst.go.jp/article/jnlp/30/2/30_867/_article/-char/ja)

論文等で本実装を参照する場合は、以下をお使いください。


```bibtex
@article{
  hayato-tsukagoshi-2023-bert-classification-tutorial,,
  title={{BERT によるテキスト分類チュートリアル}},
  author={塚越 駿 and 平子 潤},
  journal={自然言語処理},
  volume={30},
  number={2},
  pages={867-873},
  year={2023},
  doi={10.5715/jnlp.30.867},
  url = {https://www.jstage.jst.go.jp/article/jnlp/30/2/30_867/_article/-char/ja},
}
```
