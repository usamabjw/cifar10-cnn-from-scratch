from pathlib import Path

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

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

NORMALIZE_MEAN = torch.tensor(
    [0.4914, 0.4822, 0.4465]
).view(3, 1, 1)

NORMALIZE_STD = torch.tensor(
    [0.2470, 0.2435, 0.2616]
).view(3, 1, 1)


def denormalize(image):

    image = (
        image.cpu() * NORMALIZE_STD
        + NORMALIZE_MEAN
    )

    image = image.clamp(0, 1)

    return image.permute(1, 2, 0).numpy()


def save_prediction_plot(
    model,
    test_loader,
    output_path="checkpoints/evaluation_predictions.png",
    num_images=8
):

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(
            "Skipping prediction preview: "
            "install matplotlib with `pip install matplotlib`."
        )
        return

    images, labels = next(iter(test_loader))

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    with torch.no_grad():

        outputs = model(images)
        probabilities = torch.softmax(
            outputs,
            dim=1
        )
        confidences, predictions = probabilities.max(
            dim=1
        )

    num_images = min(
        num_images,
        images.size(0)
    )

    columns = 4
    rows = (
        num_images + columns - 1
    ) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.4, rows * 2.8)
    )

    axes = axes.reshape(-1)

    for index in range(num_images):

        prediction = predictions[index].item()
        label = labels[index].item()
        confidence = confidences[index].item()
        title_color = (
            "green"
            if prediction == label
            else "red"
        )

        axes[index].imshow(
            denormalize(images[index])
        )
        axes[index].set_title(
            f"Pred: {CIFAR10_CLASSES[prediction]}\n"
            f"True: {CIFAR10_CLASSES[label]}\n"
            f"{confidence:.1%}",
            color=title_color,
            fontsize=9
        )
        axes[index].axis("off")

    for index in range(num_images, len(axes)):

        axes[index].axis("off")

    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(figure)

    print(
        f"Saved prediction preview: "
        f"{output_path}"
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

    save_prediction_plot(
        model,
        test_loader
    )


if __name__ == "__main__":
    evaluate()
