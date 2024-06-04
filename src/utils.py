import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def save_jsonl(data: Iterable | pd.DataFrame, path: Path | str) -> None:
    path = Path(path)

    if type(data) != pd.DataFrame:
        data = pd.DataFrame(data)

    data.to_json(
        path,
        orient="records",
        lines=True,
        force_ascii=False,
    )


def save_json(data: dict[Any, Any], path: Path | str) -> None:
    path = Path(path)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    df = pd.read_json(path, lines=True)
    return df.to_dict(orient="records")


def load_jsonl_df(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_json(path, lines=True)
    return df


def load_json(path: Path | str) -> dict:
    path = Path(path)
    with path.open() as f:
        data = json.load(f)
    return data


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(data: dict, path: Path | str) -> dict:
    path = Path(path)

    if path.exists():
        df: pd.DataFrame = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        df.to_csv(path, index=False)
    else:
        pd.DataFrame([data]).to_csv(path, index=False)


def get_current_timestamp() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d/%H:%M:%S")
