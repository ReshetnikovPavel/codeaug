import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaModel
import datetime
from dataloaders import get_clone_detection_dataloaders


MODEL_NAME = "microsoft/codebert-base"
BATCH_SIZE = 1024
LEARNING_RATE = 2e-5
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CodeBERTCloneDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebert = RobertaModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(self.codebert.config.hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding).squeeze(-1)


def train_model(name_prefix:str, fold_num=0, t=lambda x: x):
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = CodeBERTCloneDetector().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    train_loader, val_loader = get_clone_detection_dataloaders(
        tokenizer, fold_num=fold_num, batch_size=BATCH_SIZE, t=t
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch} started")
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(DEVICE),
                "attention_mask": batch["attention_mask"].to(DEVICE),
            }
            labels = batch["labels"].to(DEVICE)

            outputs = model(**inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    "input_ids": batch["input_ids"].to(DEVICE),
                    "attention_mask": batch["attention_mask"].to(DEVICE),
                }
                labels = batch["labels"].to(DEVICE)

                outputs = model(**inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            torch.save(model.state_dict(), f"{name_prefix}_best_model_fold{fold_num}_{timestamp}.pt")

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print("-----------------------------------")
