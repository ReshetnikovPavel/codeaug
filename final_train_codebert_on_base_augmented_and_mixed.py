import os
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from load_filtered_base_augmented_and_mixed import combined_dataset

# Configuration
MODEL_NAME = "microsoft/codebert-base"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
MAX_LENGTH = 256

train_dataset = combined_dataset
val_dataset = load_dataset("PoolC/5-fold-clone-detection-600k-5fold", split="val").take(16385)


# 3. Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 4. Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["code1"],
        examples["code2"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

# 5. Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 6. Format dataset for PyTorch
train_dataset = train_dataset.rename_column("similar", "labels")
val_dataset = val_dataset.rename_column("similar", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 7. Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# 8. Metrics function
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

# 9. Training arguments
training_args = TrainingArguments(
    output_dir="./codebert_clone_detection",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=100,
    report_to="none"  # Change to "wandb" if using Weights & Biases
)

# 10. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 11. Start training
print("Starting training...")
trainer.train()

# 12. Save final model
trainer.save_model("./codebert_clone_detection/final_model")
tokenizer.save_pretrained("./codebert_clone_detection/final_model")
print("Training complete. Model saved to ./codebert_clone_detection/final_model")

# 13. Evaluate on validation set
results = trainer.evaluate()
print("\nValidation results:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")
