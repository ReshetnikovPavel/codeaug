import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
import torch
from clone_detection_with_checkpoints import get_clone_detection_dataloaders

# Initialize Accelerator
accelerator = Accelerator()

# Configuration
model_name = "microsoft/codebert-base"
num_epochs = 3
checkpoint_dir = "checkpoints"
model_dir = "models"

# Create directories
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Tokenizer and DataLoaders
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_loader, val_loader = get_clone_detection_dataloaders(tokenizer)

# Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# Prepare with Accelerator
model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, scheduler
)

# Checkpoint handling
start_epoch = 0
if os.path.exists(checkpoint_dir):
    existing_epochs = [
        int(folder.split("_")[1]) 
        for folder in os.listdir(checkpoint_dir) 
        if folder.startswith("epoch_")
    ]
    if existing_epochs:
        start_epoch = max(existing_epochs) + 1

# Training loop
for epoch in range(start_epoch, num_epochs):
    accelerator.print(f"Epoch {epoch}")
    
    # Training
    model.train()
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    # Validation
    model.eval()
    total_loss = 0
    for batch in val_loader:
        with torch.no_grad():
            outputs = model(**batch)
        total_loss += outputs.loss.item()
    avg_val_loss = total_loss / len(val_loader)
    accelerator.print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Save checkpoint
    epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    accelerator.save_state(epoch_dir)

# Save final model
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

accelerator.print("Training complete!")
