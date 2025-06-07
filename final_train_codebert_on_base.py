import os
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from load_filtered_base import filtered_hf_dataset

# Configuration
MODEL_NAME = "microsoft/codebert-base"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
MAX_LENGTH = 256
CHECKPOINT_DIR = "./codebert_clone_detection_base_checkpoints"
SAVE_STEPS = 500  # Save checkpoint every 500 steps
LOGGING_STEPS = 100  # Log progress every 100 steps

# Load datasets
train_dataset = filtered_hf_dataset
val_dataset = load_dataset("PoolC/5-fold-clone-detection-600k-5fold", split="val").take(16385)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["code1"],
        examples["code2"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Format dataset for PyTorch
train_dataset = train_dataset.rename_column("similar", "labels")
val_dataset = val_dataset.rename_column("similar", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# Metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Training arguments with checkpointing
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    evaluation_strategy="steps",  # Changed to steps
    eval_steps=SAVE_STEPS,        # Evaluate at same interval as saving
    save_strategy="steps",         # Step-based saving
    save_steps=SAVE_STEPS,         # Save checkpoint every N steps
    save_total_limit=3,            # Keep only last 3 checkpoints
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=LOGGING_STEPS,
    report_to="none",
    # Enable these for faster training if supported
    fp16=True,  # Mixed precision
    gradient_accumulation_steps=2,  # Accumulate gradients
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Detect existing checkpoints for resumption
last_checkpoint = None
if os.path.exists(CHECKPOINT_DIR):
    last_checkpoint = get_last_checkpoint(CHECKPOINT_DIR)
    if last_checkpoint:
        print(f"Resuming training from checkpoint: {last_checkpoint}")

# Start training with resumption support
print("Starting training...")
trainer.train(resume_from_checkpoint=last_checkpoint)

# Save final model
final_model_path = os.path.join(CHECKPOINT_DIR, "final_model")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Training complete. Model saved to {final_model_path}")

# Evaluate on validation set
results = trainer.evaluate()
print("\nValidation results:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")
