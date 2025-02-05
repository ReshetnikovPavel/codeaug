import argparse

from datasets import load_dataset
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel


import models
import dataloaders


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model on a dataset")

    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=True,
        help="A task to train model on",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="A model name to train",
    )

    return parser.parse_args()


def clone_detection(args):
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    # print("got tokenizer")
    train_loader, val_loader = dataloaders.get_clone_detection_dataloaders(tokenizer)
    # print("got loaders")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("got device")
    model = models.CodeBertBinaryClassifier().to(device)
    # print("got model")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # print("got loss")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # print("got optimizer")

    epochs = 1
    for epoch in range(epochs):
        print("training epoch", epoch)
        train_loss = model.train_epoch(optimizer, train_loader, loss_fn, device)
        metrics = model.evaluate(val_loader, loss_fn, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {metrics['loss']:.4f} | Val Acc: {metrics['acc']:.2f}%")


if __name__ == "__main__":
    args = parse_arguments()

    match args.task:
        case "clone-detection":
            clone_detection(args)
