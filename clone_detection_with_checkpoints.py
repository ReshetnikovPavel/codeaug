import torch
import os
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, load_from_disk


def get_clone_detection_dataloaders(
    tokenizer,
    from_chunk: int,
    to_chunk: int,
    fold_num=0,
    batch_size=1024,
    transforms=None,
    base_checkpoint_dir="./clone_detection_checkpoints",
    chunk_size=500,
):
    """
    Creates dataloaders with checkpointing support for code clone detection.

    Args:
        tokenizer: The tokenizer to use for processing
        fold_num: Which fold to use (0-4)
        batch_size: Batch size for dataloaders
        transforms: List of Compose transformations for data augmentation
        base_checkpoint_dir: Base directory for storing checkpoints
        chunk_size: Number of examples per processing chunk
    """
    print(f"Initializing clone detection dataloaders for fold {fold_num}")
    ds = load_dataset("PoolC/5-fold-clone-detection-600k-5fold")
    train_split = "train"
    val_split = "val"

    # Define transformation functions
    def apply_transform(examples, transform):
        """Applies a Compose transformation to code pairs"""
        return {
            "code1": [transform(c) for c in examples["code1"]],
            "code2": [transform(c) for c in examples["code2"]],
        }

    def tokenize_batch(examples):
        """Tokenizes a batch of examples"""
        return tokenizer(
            examples["code1"],
            examples["code2"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    # Checkpoint processing functions
    def process_with_checkpoint(dataset, process_fn, checkpoint_dir, desc="Processing"):
        """
        Processes dataset with checkpointing support
        Returns concatenated dataset with all processed chunks
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        total_examples = len(dataset)
        num_chunks = (total_examples + chunk_size - 1) // chunk_size

        processed_chunks = []

        # Check existing chunks
        last_completed = from_chunk
        for i in range(last_completed, to_chunk):
            chunk_path = os.path.join(checkpoint_dir, f"chunk_{i}")
            if os.path.exists(chunk_path):
                last_completed = i
            else:
                break

        # Load completed chunks
        # for i in range(last_completed + 1):
        #     chunk_path = os.path.join(checkpoint_dir, f"chunk_{i}")
        #     processed_chunks.append(load_from_disk(chunk_path))
        #     print(f"Loaded existing {desc} chunk {i}")

        # Process new chunks
        for i in range(last_completed + 1, to_chunk):
            print(f"{desc} chunk {i + 1}/{to_chunk}")
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_examples)

            # Select and process chunk
            chunk = dataset.select(range(start_idx, end_idx))
            processed = chunk.map(
                process_fn,
                batched=True,
                batch_size=32,
                remove_columns=["code1", "code2"],
            )

            # Save checkpoint
            chunk_path = os.path.join(checkpoint_dir, f"chunk_{i}")
            processed.save_to_disk(chunk_path)
            processed_chunks.append(processed)

        return concatenate_datasets(processed_chunks).with_format("torch")

    # Setup checkpoint directories
    fold_dir = os.path.join(base_checkpoint_dir, f"fold_{fold_num}")
    train_base_dir = os.path.join(fold_dir, "train")
    val_dir = os.path.join(fold_dir, "val")

    # Process validation dataset first (no transforms)
    # print("Processing validation dataset...")
    # val_ds = process_with_checkpoint(
    #     ds[val_split], lambda x: tokenize_batch(x), val_dir, desc="Validating"
    # )

    # Process training datasets
    train_datasets = []

    # 1. Base training dataset (no transforms)
    # print("\nProcessing base training dataset...")
    # base_train_ds = process_with_checkpoint(
    #     ds[train_split],
    #     lambda x: tokenize_batch(x),
    #     os.path.join(train_base_dir, "base"),
    #     desc="Base training",
    # )
    # train_datasets.append(base_train_ds)

    # 2. Transformed datasets
    if transforms is not None:
        for idx, transform in enumerate(transforms):
            print(f"\nProcessing transformation {idx + 1}/{len(transforms)}")

            def transform_fn(examples):
                transformed = apply_transform(examples, transform)
                return tokenize_batch(transformed)

            transform_dir = os.path.join(train_base_dir, f"transform_{idx}")
            transformed_ds = process_with_checkpoint(
                ds[train_split], transform_fn, transform_dir, desc=f"Transform {idx}"
            )
            train_datasets.append(transformed_ds)

    # Combine all training datasets
    train_ds = concatenate_datasets(train_datasets)

    # Create dataloaders
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["similar"].float() for x in batch]),
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True
    )

    return train_loader, val_loader
