import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import itertools


def get_clone_detection_dataloaders(
    tokenizer, fold_num=0, batch_size=16, t=lambda x: x
):
    print("loading PoolC/5-fold-clone-detection-600k-5fold")
    ds = load_dataset("PoolC/5-fold-clone-detection-600k-5fold")
    train_split = "train"
    val_split = "val"

    def tokenize_function(examples):
        return tokenizer(
            list(itertools.chain(examples["code1"], map(t, examples["code1"]))),
            list(itertools.chain(examples["code2"], map(t, examples["code2"]))),
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    def process_dataset(dataset):
        return dataset.map(
            tokenize_function, batched=True, remove_columns=["code1", "code2"]
        ).with_format("torch")

    print("Processing train dataset")
    train_ds = process_dataset(ds[train_split]).take(10)
    print("Processing val dataset")
    val_ds = process_dataset(ds[val_split]).take(10)

    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["similar"].float() for x in batch]),
        }

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, val_loader
