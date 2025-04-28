import dataloaders
import models
import torch
from transformers import RobertaTokenizer


def clone_detection(transform, out: str):
    print("Getting tokenizer")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    print("Getting dataloaders")
    train_loader, val_loader = dataloaders.get_clone_detection_dataloaders(
        tokenizer, t=transform
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Getting CodeBert")
    model = models.CodeBertBinaryClassifier().to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    epochs = 1
    for epoch in range(epochs):
        print("training epoch", epoch)
        train_loss = model.train_epoch(optimizer, train_loader, loss_fn, device)
        metrics = model.evaluate(val_loader, loss_fn, device)
        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {
                metrics['loss']:.4f
            } | Val Acc: {metrics['acc']:.2f}%"
        )
    torch.save(model.state_dict(), out)
