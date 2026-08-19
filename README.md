# CodeAlpha Handwritten Character Recognition

A beginner-friendly deep-learning project for **CodeAlpha Machine Learning Task 3**. It recognises handwritten digits using a Convolutional Neural Network (CNN).

## About the dataset

CodeAlpha allows handwritten recognition with MNIST or EMNIST. This project uses **MNIST**, the simplest beginner-friendly choice. MNIST contains 70,000 grayscale images of handwritten digits from 0 to 9. Each image is 28 x 28 pixels.

Although this implementation recognises digits rather than letters, it demonstrates the same image-processing and CNN workflow required for handwritten character recognition. To recognise alphabet characters later, the same structure can be used with EMNIST.

## What the project does

1. Downloads the MNIST dataset automatically through TensorFlow.
2. Normalises image pixels from 0-255 to 0-1 and adds a grayscale channel.
3. Builds and trains a CNN with convolution, pooling, dropout, and dense layers.
4. Evaluates the trained model on unseen test images.
5. Saves a classification report, confusion matrix, training curves, a trained model, and sample prediction images.

## Project structure

```text
CodeAlpha_Handwritten_Character_Recognition/
├── train.py                  # Dataset loading, preprocessing, training, evaluation
├── requirements.txt          # Required Python packages
├── .gitignore                # Ignores environments and generated outputs
├── README.md                 # Project documentation
└── outputs/                  # Created after training
    ├── mnist_cnn.keras
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── sample_predictions.png
    └── classification_report.txt
```

## Setup and run

### 1. Create and activate a virtual environment (recommended)

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Train and evaluate the model

```bash
python train.py
```

The first run downloads MNIST automatically. The default is 5 epochs, which usually gives high accuracy while remaining practical on a normal laptop.

For a faster test run:

```bash
python train.py --epochs 1
```

For more training:

```bash
python train.py --epochs 10 --batch-size 128
```

## Results

After training, inspect the `outputs` folder. It contains:

- `mnist_cnn.keras` — the saved trained CNN.
- `training_curves.png` — training versus validation accuracy and loss.
- `confusion_matrix.png` — correct and incorrect predictions for each digit.
- `sample_predictions.png` — 16 handwritten test digits with predicted labels and confidence.
- `classification_report.txt` — precision, recall, and F1-score for every digit.

Exact results vary slightly by machine and TensorFlow version. With the default settings, the CNN should normally achieve about 98–99% test accuracy.

## CNN explanation

- **Convolution layers** learn visual patterns such as edges, curves, and digit shapes.
- **Max-pooling layers** shrink feature maps and retain the strongest features.
- **Dropout layers** reduce overfitting by temporarily switching off some connections during training.
- The final **softmax** layer returns one probability for each digit class (0–9).

## GitHub upload

Upload this folder to a new GitHub repository named `CodeAlpha_Handwritten_Character_Recognition`. Do not upload the generated `outputs` folder or virtual environment; they are already listed in `.gitignore`.

## Possible extension: EMNIST letters

To extend this project to alphabet recognition, replace MNIST with an EMNIST letters or balanced split, update the number of output classes, and update `CLASS_NAMES`. The preprocessing, CNN design, evaluation, and visualisation steps remain nearly identical.

## Requirements

- Python 3.10 or newer recommended
- Internet connection the first time MNIST is downloaded
