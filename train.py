"""Train a CNN to recognise handwritten MNIST digits (0-9)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


SEED = 42
CLASS_NAMES = [str(number) for number in range(10)]


def set_seed() -> None:
    """Make results more repeatable between runs."""
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def load_and_preprocess_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download MNIST if needed, normalise pixels, and add a channel dimension."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Pixel values are 0-255. Neural networks train better with values from 0-1.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # A CNN expects (height, width, channels); MNIST images are grayscale (one channel).
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    return x_train, y_train, x_test, y_test


def build_model() -> tf.keras.Model:
    """Create a small CNN that is easy to understand and runs on a normal computer."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="mnist_cnn",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_curves(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    """Save accuracy and loss plots from training."""
    history_data = history.history
    epochs = range(1, len(history_data["accuracy"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history_data["accuracy"], label="Training")
    axes[0].plot(epochs, history_data["val_accuracy"], label="Validation")
    axes[0].set(title="Model accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history_data["loss"], label="Training")
    axes[1].plot(epochs, history_data["val_loss"], label="Validation")
    axes[1].set(title="Model loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(figure)


def save_evaluation_plots(
    y_true: np.ndarray, probabilities: np.ndarray, output_dir: Path
) -> None:
    """Save a confusion matrix and a grid of example model predictions."""
    predictions = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(y_true, predictions)
    figure, axis = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=axis, colorbar=False, cmap="Blues"
    )
    axis.set_title("MNIST test-set confusion matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def save_sample_predictions(
    images: np.ndarray, labels: np.ndarray, probabilities: np.ndarray, output_dir: Path
) -> None:
    """Save 16 test images labelled with their actual and predicted digit."""
    predictions = np.argmax(probabilities, axis=1)
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))
    for index, axis in enumerate(axes.flat):
        predicted = predictions[index]
        actual = labels[index]
        colour = "green" if predicted == actual else "red"
        confidence = probabilities[index, predicted] * 100
        axis.imshow(images[index].squeeze(), cmap="gray")
        axis.set_title(
            f"Actual: {actual} | Pred: {predicted}\n{confidence:.1f}%", color=colour, fontsize=9
        )
        axis.axis("off")
    figure.suptitle("Sample MNIST predictions", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_dir / "sample_predictions.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN on handwritten MNIST digits.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5).")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size (default: 128).")
    args = parser.parse_args()

    set_seed()
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    print("Loading and preprocessing the MNIST dataset...")
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    print(f"Training images: {x_train.shape}; test images: {x_test.shape}")

    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]
    print("\nTraining the CNN...")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(y_test, predictions, target_names=CLASS_NAMES, digits=4)
    print("\nClassification report:\n")
    print(report)

    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_training_curves(history, output_dir)
    save_evaluation_plots(y_test, probabilities, output_dir)
    save_sample_predictions(x_test, y_test, probabilities, output_dir)
    model.save(output_dir / "mnist_cnn.keras")

    print("\nSaved model and visualisations to the outputs folder:")
    print("- mnist_cnn.keras")
    print("- training_curves.png")
    print("- confusion_matrix.png")
    print("- sample_predictions.png")
    print("- classification_report.txt")


if __name__ == "__main__":
    main()
