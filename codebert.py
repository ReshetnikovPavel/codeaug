import os
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch

# Initialize Accelerator
accelerator = Accelerator()

# Configuration
model_name = "microsoft/codebert-base"
num_epochs = 3
batch_size = 16
max_length = 1024
checkpoint_dir = "checkpoints"
tokenized_dir = "tokenized_data"
model_dir = "models"

# Create directories
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tokenized_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["code1"],
        examples["code2"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

# Process each fold

        # Load dataset
dataset = load_dataset("PoolC/5-fold-clone-detection-600k-5fold")
train_dataset = dataset["train"]
eval_dataset = dataset["val"]

        # Tokenize
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

        # Save tokenized datasets
train_dataset.save_to_disk(train_tokenized_path)
eval_dataset.save_to_disk(eval_tokenized_path)

    # Set format for PyTorch
columns = ["input_ids", "attention_mask", "label"]
train_dataset.set_format(type="torch", columns=columns)
eval_dataset.set_format(type="torch", columns=columns)

    # DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    # Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Optimizer and Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps    )

    # Prepare with Accelerato
model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    # Check for existing checkpoints
fold_checkpoint_dir = f"{checkpoint_dir}"
start_epoch = 0
if os.path.exists(fold_checkpoint_dir):
    epochs = [int(d.split("_")[1]) for d in os.listdir(fold_checkpoint_dir) if d.startswith("epoch")]
    if epochs:
        start_epoch = max(epochs) + 1

    # Training loop
for epoch in range(start_epoch, num_epochs):
    accelerator.print(f"Epoch {epoch}")
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Evaluation
    model.eval()
    total_eval_loss = 0
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        total_eval_loss += loss.item()
    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    accelerator.print(f"Validation Loss: {avg_eval_loss:.4f}")

        # Save checkpoint
    epoch_checkpoint_dir = f"{fold_checkpoint_dir}/epoch_{epoch}"
    accelerator.save_state(epoch_checkpoint_dir)

    # Save final model
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(f"{model_dir}")
    tokenizer.save_pretrained(f"{model_dir}")

accelerator.print("Training complete!")
