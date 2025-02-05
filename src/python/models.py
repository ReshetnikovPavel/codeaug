import torch
from torch import nn
from transformers import RobertaModel


class CodeBertBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding from last hidden state
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits.squeeze(-1)  # (batch_size,)

    def train_epoch(self, optimizer, train_loader, loss_fn, device="cpu"):
        self.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)

            outputs = self(input_ids, attn_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, test_loader, loss_fn, device="cpu"):
        self.eval()
        total_loss = 0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].float().to(device)

                outputs = self(input_ids, attn_mask)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()

                # Calculate accuracy
                preds = (torch.sigmoid(outputs) >= 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total_samples += labels.size(0)

        return {
            "loss": total_loss / len(test_loader),
            "acc": correct / total_samples * 100,
        }
