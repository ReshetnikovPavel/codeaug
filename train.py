import argparse

import dataloaders
import models
import torch
from transformers import RobertaTokenizer

import codeaug.transforms


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
    parser.add_argument(
        "transformations",
        nargs="*",
        type=str,
        help="A list of code augmentations techniques applied in given order."
        " Possible values are: remove-comments, invert-ifs",
    )

    return parser.parse_args()


def build_transformations(args):
    transforms = []
    for t in args.transformations:
        if t == "remove-comments":
            transforms.append(codeaug.transforms.RemoveComments)
        elif t == "invert-ifs":
            transforms.append(codeaug.transforms.InvertIfs)
        else:
            raise argparse.ArgumentError(f"unknown transformation option `{t}`")
    return codeaug.transforms.Compose(transforms)


def clone_detection(args):
    transform = build_transformations(args)
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    train_loader, val_loader = dataloaders.get_clone_detection_dataloaders(
        tokenizer, t=transform
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    torch.save(model.state_dict(), "out")


if __name__ == "__main__":
    args = parse_arguments()

    match args.task:
        case "clone-detection":
            clone_detection(args)
