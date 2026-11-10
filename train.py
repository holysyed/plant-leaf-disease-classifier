"""Train a plant leaf disease classifier from class-named image folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/PlantVillage"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use ImageNet pretrained weights (downloaded on first run) or train from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(
            f"Dataset folder not found: {args.data_dir}\n"
            "Create one subfolder per class, with that class's images inside."
        )
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive integers.")

    image_size = (args.image_size, args.image_size)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="int",
    )
    class_names = train_ds.class_names
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="int",
    )
    if len(class_names) < 2:
        raise SystemExit("The dataset needs at least two class subfolders.")

    autotune = tf.data.AUTOTUNE

    def prepare(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
        return images, labels

    train_ds = train_ds.map(prepare, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(prepare, num_parallel_calls=autotune).prefetch(autotune)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights=None if args.weights == "none" else "imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*image_size, 3), name="leaf_image")
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="plant_disease_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.output_dir / "plant_disease_model.keras")
    (args.output_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )

    training_history = {
        metric: [float(value) for value in values]
        for metric, values in history.history.items()
    }
    (args.output_dir / "training_history.json").write_text(
        json.dumps(training_history, indent=2), encoding="utf-8"
    )

    validation_metrics = {
        metric: float(value)
        for metric, value in model.evaluate(val_ds, return_dict=True, verbose=0).items()
    }
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2), encoding="utf-8"
    )

    print(f"Saved model and reports in {args.output_dir.resolve()}")
    print(
        "Final validation metrics: "
        + ", ".join(f"{name}={value:.4f}" for name, value in validation_metrics.items())
    )


if __name__ == "__main__":
    main()
