import logging

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from models.cnn import CIFAR10CNN
from data.dataloader import get_dataloaders
from utils.metrics import accuracy
from utils.checkpoint import save_checkpoint


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 128


def train():

    train_loader, test_loader = get_dataloaders(
        BATCH_SIZE
    )

    model = CIFAR10CNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR
    )

    logger.info(
        "Starting training on %s with %d epochs, batch size %d, learning rate %.1e",
        DEVICE,
        EPOCHS,
        BATCH_SIZE,
        LR,
    )

    best_acc = 0

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0

        loop = tqdm(train_loader)

        for images, labels in loop:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            loop.set_description(
                f"Epoch {epoch+1}/{EPOCHS}"
            )

            loop.set_postfix(
                loss=loss.item()
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

        test_acc = (
            total_acc / len(test_loader)
        )

        logger.info(
            "Epoch %d completed. Test Accuracy=%.4f",
            epoch + 1,
            test_acc,
        )

        if test_acc > best_acc:

            best_acc = test_acc

            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_acc,
                "checkpoints/best_model.pth"
            )

            logger.info(
                "Saved checkpoint with best accuracy=%.4f",
                best_acc,
            )


if __name__ == "__main__":
    train()