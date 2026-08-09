"""
Train the Brain Tumor MRI CNN classifier.

Usage:
    python train.py --data-dir ./brain-tumor-mri-dataset --epochs 40

Expects a directory with the Kaggle "Brain Tumor MRI Dataset" layout:
    <data-dir>/Training/{glioma,meningioma,notumor,pituitary}/*.jpg
    <data-dir>/Testing/{glioma,meningioma,notumor,pituitary}/*.jpg

Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Saves the trained model to model/tumor_classifier.h5 (or --out-path).
"""

import argparse
import os
import sys

from src.model_architecture import IMG_SIZE, CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Train brain tumor MRI CNN classifier")
    parser.add_argument("--data-dir", default="./brain-tumor-mri-dataset",
                         help="Path to dataset root containing Training/ and Testing/ folders")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-path", default="model/tumor_classifier.keras",
                         help="Where to save the trained model")
    return parser.parse_args()


def main():
    args = parse_args()

    train_dir = os.path.join(args.data_dir, "Training")
    test_dir = os.path.join(args.data_dir, "Testing")
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        print(f"ERROR: expected '{train_dir}' and '{test_dir}' to exist.")
        print("Download the dataset first: "
              "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        sys.exit(1)

    # Imports are deferred until after the arg/path checks above so that
    # `python train.py --help` and early error paths don't pay TensorFlow's
    # import cost.
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from src.model_architecture import build_model

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        zoom_range=0.05,
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_data = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale",
        batch_size=args.batch_size, class_mode="sparse", shuffle=True,
    )
    test_data = test_datagen.flow_from_directory(
        test_dir, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale",
        batch_size=args.batch_size, class_mode="sparse", shuffle=False,
    )

    # Sanity check: class_indices order must match CLASS_NAMES, or predicted
    # labels at inference time will silently be wrong.
    expected = {name: i for i, name in enumerate(CLASS_NAMES)}
    if train_data.class_indices != expected:
        print("WARNING: dataset class order does not match expected "
              f"{expected}, got {train_data.class_indices}. "
              "Update src/model_architecture.CLASS_NAMES to match, "
              "or predictions will be mislabeled.")

    model = build_model()
    model.summary()

    lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                                    min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor="val_accuracy", patience=12,
                                restore_best_weights=True, verbose=1, mode="max")

    history = model.fit(
        train_data, epochs=args.epochs, validation_data=test_data,
        callbacks=[lr_reducer, early_stop],
    )

    test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    model.save(args.out_path)
    print(f"\nModel saved to {args.out_path}")


if __name__ == "__main__":
    main()
