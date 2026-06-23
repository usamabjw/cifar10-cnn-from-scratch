import torch
import os


def save_checkpoint(model,
                    optimizer,
                    epoch,
                    accuracy,
                    path):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "accuracy": accuracy
        },
        path
    )


def load_checkpoint(path,
                    model,
                    optimizer=None):

    checkpoint = torch.load(path)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint