import torch

from models.cnn import CIFAR10CNN
from data.dataloader import get_dataloaders
from utils.metrics import accuracy
from utils.checkpoint import load_checkpoint


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def evaluate():

    _, test_loader = get_dataloaders(
        batch_size=128
    )

    model = CIFAR10CNN().to(DEVICE)

    checkpoint = load_checkpoint(
        "checkpoints/best_model.pth",
        model
    )

    model.eval()

    total_acc = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            total_acc += accuracy(
                outputs,
                labels
            )

    final_acc = (
        total_acc / len(test_loader)
    )

    print(
        f"Test Accuracy: "
        f"{final_acc:.4f}"
    )

    print(
        f"Best Validation Accuracy: "
        f"{checkpoint['accuracy']:.4f}"
    )


if __name__ == "__main__":
    evaluate()