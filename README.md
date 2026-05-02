# Stroke Detection with Deep Learning

A PyTorch-based CNN project for detecting strokes from brain scan images using a custom convolutional neural network called **OzNet**.

## Overview

This project trains a binary/multi-class image classifier on brain scan images to distinguish between stroke and non-stroke cases. It uses grayscale MRI/CT images resized to 128×128 pixels and logs training metrics to TensorBoard.

## Project Structure

```
Stroke_Detection/
├── dataset/              # Dataset directory (not tracked by git)
│   ├── train/            # Training images (organized by class)
│   ├── val/              # Validation images (organized by class)
│   └── test/             # Test images (organized by class)
├── models/
│   └── cnn.py            # OzNet CNN architecture
├── notebooks/
│   └── splitting_data.ipynb  # Notebook for splitting raw data
├── scripts/
│   └── main.py           # Entry point for training
├── src/
│   ├── dataloaders.py    # Dataset loading and transforms
│   └── train.py          # Training and validation loops
└── checkpoints/          # Saved model checkpoints (not tracked by git)
```

## Model Architecture — OzNet

OzNet is a custom CNN with:
- **4 convolutional blocks** (64 → 128 → 256 → 512 filters), each followed by BatchNorm and MaxPooling
- **3 fully connected layers** (4096 → 1024 → num_classes) with BatchNorm and Dropout
- Input: single-channel (grayscale) 128×128 images

## Requirements

Install dependencies with:

```bash
pip install torch torchvision torchmetrics tensorboard tqdm
```

## Dataset Setup

Organize your dataset in the following structure, where each subfolder represents a class (e.g., `stroke`, `normal`):

```
dataset/
├── train/
│   ├── stroke/
│   └── normal/
├── val/
│   ├── stroke/
│   └── normal/
└── test/
    ├── stroke/
    └── normal/
```

You can use the `notebooks/splitting_data.ipynb` notebook to split raw data into train/val/test sets.

## Usage

Run training from the project root:

```bash
python scripts/main.py
```

Training configuration can be adjusted directly in `scripts/main.py`:

| Parameter     | Default | Description                        |
|---------------|---------|------------------------------------|
| `batch_size`  | 32      | Number of images per batch         |
| `num_epochs`  | 15      | Number of training epochs          |
| `lr`          | 0.001   | SGD learning rate                  |
| `momentum`    | 0.9     | SGD momentum                       |
| `dropout_rate`| 0.5     | Dropout probability in OzNet       |

## Monitoring

Training metrics (loss, accuracy, F1 score) are logged with TensorBoard. Launch it with:

```bash
tensorboard --logdir=runs
```

## Output

- **Model checkpoints** are saved to `checkpoints/saved_model.pth` after each epoch.
- **TensorBoard logs** are written to the `runs/` directory.

## Metrics

The following metrics are tracked for both training and validation:
- Loss (CrossEntropy)
- Accuracy
- F1 Score (macro for training, weighted for validation)
