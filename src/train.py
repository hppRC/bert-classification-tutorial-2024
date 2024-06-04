from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
)
from transformers import Trainer as HFTrainer
from transformers.trainer_utils import PredictionOutput

import datasets as ds
from src import utils


@dataclass
class TrainingArgs(TrainingArguments):
    output_dir: str = None

    num_train_epochs: int = 20
    learning_rate: float = 3e-5
    per_device_train_batch_size: int = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    dataloader_num_workers: int = 4
    lr_scheduler_type: str = "cosine"

    # 使用するデータ型、BF16を利用することで高速かつ省メモリで学習可能
    # 一般にFP16よりBF16の方が学習が安定している
    bf16: bool = True

    # optimizerが持つ勾配情報を適宜再計算することで保持するメモリを削減するGradient Checkpointingの設定
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": True})

    # tensorboardで実験ログを残しておくとどんな感じで学習が進んでいるかわかって便利
    report_to: str = "tensorboard"
    logging_steps: int = 10
    logging_dir: str = None

    # 最良のモデルを選ぶ際に基準となる指標
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    # val accuracyを基準に選ぶ場合は以下のようにする
    # metric_for_best_model: str = "acc"
    # greater_is_better: bool = True

    eval_strategy: str = "epoch"
    per_device_eval_batch_size: int = 32

    save_strategy: str = "epoch"
    save_total_limit: int = 1

    ddp_find_unused_parameters: bool = False
    load_best_model_at_end: bool = False
    remove_unused_columns: bool = False


@dataclass
class ExperimentConfig:
    model_name: str = "cl-tohoku/bert-base-japanese-v3"
    dataset_dir: Path = "./datasets/livedoor"
    experiment_name: str = "default"
    max_seq_len: int = 512

    def __post_init__(self):
        self.label2id = utils.load_json(self.dataset_dir / "label2id.json")


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizer
    max_seq_len: int

    def __call__(self, data_list: list[dict[str, Any]]) -> BatchEncoding:
        title = [d["title"] for d in data_list]
        body = [d["body"] for d in data_list]
        inputs: BatchEncoding = self.tokenizer(
            title,
            body,
            padding=True,
            truncation="only_second",
            return_tensors="pt",
            max_length=self.max_seq_len,
        )
        inputs["labels"] = torch.LongTensor([d["label"] for d in data_list])
        return inputs


class ComputeMetrics:
    def __init__(self, labels: list[str]):
        self.labels = labels

    def __call__(self, eval_pred: EvalPrediction):
        pred_labels = torch.Tensor(eval_pred.predictions.argmax(axis=1).reshape(-1))
        gold_labels = torch.Tensor(eval_pred.label_ids.reshape(-1))

        accuracy: float = accuracy_score(gold_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold_labels,
            pred_labels,
            average="macro",
            zero_division=0,
            labels=self.labels,
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def main(training_args: TrainingArgs, config: ExperimentConfig):
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.add_eos_token = True

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(config.label2id),
        label2id=config.label2id,
        id2label={v: k for k, v in config.label2id.items()},
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )

    datasets: ds.DatasetDict = ds.load_from_disk(str(config.dataset_dir))

    data_collator = DataCollator(
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
    )

    compute_metrics = ComputeMetrics(labels=list(config.label2id.values()))

    trainer = HFTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer._load_best_model()

    trainer.save_model()
    trainer.save_state()
    trainer.tokenizer.save_pretrained(training_args.output_dir)

    # 最良のモデルを使ってval set, test setで評価
    val_prediction_output: PredictionOutput = trainer.predict(test_dataset=datasets["validation"])
    test_prediction_output: PredictionOutput = trainer.predict(test_dataset=datasets["test"])

    if training_args.process_index == 0:
        val_metrics: dict[str, float] = val_prediction_output.metrics
        val_metrics = {k.replace("test_", ""): v for k, v in val_metrics.items()}

        test_metrics: dict[str, float] = test_prediction_output.metrics
        test_metrics = {k.replace("test_", ""): v for k, v in test_metrics.items()}

        metrics = {
            "best-val": val_metrics,
            "test": test_metrics,
        }

        utils.save_json(metrics, Path(training_args.output_dir, "metrics.json"))

        with Path(training_args.output_dir, "training_args.json").open("w") as f:
            f.write(trainer.args.to_json_string())


def summarize_config(training_args: TrainingArgs, config: ExperimentConfig) -> str:
    accelerator = Accelerator()
    batch_size = training_args.per_device_train_batch_size * accelerator.num_processes
    config_summary = {
        "B": batch_size,
        "E": training_args.num_train_epochs,
        "LR": training_args.learning_rate,
        "L": config.max_seq_len,
    }
    config_summary = "".join(f"{k}{v}" for k, v in config_summary.items())
    return config_summary


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArgs, ExperimentConfig))
    training_args, config = parser.parse_args_into_dataclasses()
    config_summary = summarize_config(training_args, config)
    model_name = config.model_name.replace("/", "__")

    training_args.output_dir = f"outputs/{model_name}/{config_summary}/{config.experiment_name}"
    training_args.logging_dir = training_args.output_dir
    training_args.run_name = f"{config_summary}/{config.experiment_name}"

    main(training_args, config)
