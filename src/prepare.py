import pdb
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from transformers import HfArgumentParser

import datasets as ds
from src import utils


@dataclass
class Args:
    output_dir: Path = "./datasets/livedoor"
    seed: int = 42


def process_title(title: str) -> str:
    title = unicodedata.normalize("NFKC", title)
    title = title.strip("　").strip()
    return title


# 記事本文の前処理
# 重複した改行の削除、文頭の全角スペースの削除、NFKC正規化を実施
def process_body(body: list[str]) -> str:
    body = [unicodedata.normalize("NFKC", line) for line in body]
    body = [line.strip("　").strip() for line in body]
    body = [line for line in body if line]
    body = "\n".join(body)
    return body


DATASET_URL = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"


def main(args: Args):
    # datasetsのダウンロード用クラスを利用することでbashコマンドを使わずにダウンロードできる
    # tar.gzの解凍も自動で行ってくれて便利
    dl_manager: ds.DownloadManager = ds.DownloadManager(
        download_config=ds.DownloadConfig(num_proc=16),
    )
    data_dir: str = dl_manager.download_and_extract(DATASET_URL)

    # ライブドアニュースコーパスの実データが保存されいているディレクトリへのパス
    input_dir = Path(data_dir, "text")

    # `.from_generator`の`gen_kwargs`にgenerator関数に渡す引数を指定
    # リストが渡されていると`num_proc`の数にリストを分割して分配・処理する
    # 例: リストの長さが4, num_proc=2の場合、2つのプロセスでそれぞれ2つの要素を処理する
    def generator(paths: list[Path]):
        for path in paths:
            category = path.parent.name

            # データフォーマット
            # １行目：記事のURL
            # ２行目：記事の日付
            # ３行目：記事のタイトル
            # ４行目以降：記事の本文
            lines: list[str] = path.read_text().splitlines()
            url, date, title, *body = lines

            yield {
                "category": category,
                "category-id": path.stem,
                "url": url.strip(),
                "date": date.strip(),
                "title": process_title(title.strip()),
                "body": process_body(body),
            }

    # ライセンスファイル以外のテキストデータへのパスを取得
    paths = [path for path in input_dir.glob("*/*.txt") if path.name != "LICENSE.txt"]

    # generator関数から直接データセットを作成
    dataset = ds.Dataset.from_generator(
        generator,
        gen_kwargs={"paths": paths},  # リストを渡すとnum_procの数に分割して処理される
        num_proc=16,
    )

    dataset = dataset.shuffle(seed=args.seed)

    # ラベルを作っておく
    labels = set(dataset["category"])
    label2id = {label: i for i, label in enumerate(sorted(labels))}

    # ラベルを数値に変換
    # datasetsのmap関数はデータセットの各要素に対して関数を適用する(batch=Falseの時)
    # こんな感じでローカル関数を都度作るようにすると名前を考えなくていいので楽、あとdictの参照とか楽
    def process(x: dict):
        return {
            "label": label2id[x["category"]],
        }

    dataset = dataset.map(process, num_proc=16)

    # train, validation, testに一括で分割できないので段階的にやる(4:(1→1:1))
    datasets = dataset.train_test_split(test_size=0.2)
    train_dataset = datasets["train"]
    val_test_datasets = datasets["test"].train_test_split(test_size=0.5)

    datasets = ds.DatasetDict(
        {
            "train": train_dataset,
            "validation": val_test_datasets["train"],
            "test": val_test_datasets["test"],
        }
    )

    datasets.save_to_disk(str(args.output_dir))
    utils.save_json(label2id, args.output_dir / "label2id.json")

    pdb.set_trace()


if __name__ == "__main__":
    parser = HfArgumentParser((Args,))
    [args] = parser.parse_args_into_dataclasses()
    main(args)
