# cifar10-cnn-from-scratch
Training a CNN on cifar10 dataset

A PyTorch implementation of a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.

Inspired by:
https://cs231n.github.io/convolutional-networks/

## Dataset

CIFAR-10 contains 60,000 color images (32x32 pixels) belonging to 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Architecture

Input (32x32x3)

Conv(3→32, 3x3)
ReLU

Conv(32→32, 3x3)
ReLU

MaxPool(2x2)

Conv(32→64, 3x3)
ReLU

Conv(64→64, 3x3)
ReLU

MaxPool(2x2)

Flatten

FC(64×8×8 → 512)
ReLU
Dropout(0.5)

FC(512 → 10)

## Installation

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

## Evaluate

```bash
python evaluate.py
```

## Results

Expected Accuracy:

- Train Accuracy: ~90%
- Test Accuracy: ~75–80%

## References

- CS231n Convolutional Networks
- PyTorch Documentation