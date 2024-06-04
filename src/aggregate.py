from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from transformers import HfArgumentParser

from src import utils


@dataclass
class Args:
    output_dir: Path = "./outputs"
    result_dir: Path = "./results"


def main(args: Args):
    data = []
    for path in args.output_dir.glob("**/metrics.json"):
        metrics = utils.load_json(path)
        config = utils.load_json(path.parent / "config.json")

        data.append(
            {
                "model_name": config["_name_or_path"],
                "best-val-f1": metrics["best-val"]["f1"],
                "best-val-acc": metrics["best-val"]["f1"],
                "f1": metrics["test"]["f1"],
                "accuracy": metrics["test"]["accuracy"],
                "precision": metrics["test"]["precision"],
                "recall": metrics["test"]["recall"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data).sort_values("f1", ascending=False)
    df.to_csv(str(args.output_dir / "all.csv"), index=False)

    best_df = (
        df.groupby("model_name")
        .apply(lambda x: x.nlargest(1, "best-val-f1").reset_index(drop=True), include_groups=False)
        .reset_index(level=0)
    ).sort_values("f1", ascending=False)

    best_df.to_csv(str(args.output_dir / "best.csv"), index=False)

    print("|Model|Accuracy|Precision|Recall|F1|")
    print("|:-|:-:|:-:|:-:|:-:|")
    for row in best_df.to_dict("records"):
        print(
            f'|[{row["model_name"]}](https://huggingface.co/{row["model_name"]})|{row["accuracy"]*100:.2f}|{row["precision"]*100:.2f}|{row["recall"]*100:.2f}|{row["f1"]*100:.2f}|'
        )


if __name__ == "__main__":
    parser = HfArgumentParser((Args,))
    [args] = parser.parse_args_into_dataclasses()
    main(args)
