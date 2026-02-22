import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))
    return loss


def ensure_dirs():
    Path("models").mkdir(exist_ok=True)
    Path("assets").mkdir(exist_ok=True)


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Greens"
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    cfg = load_config()
    ensure_dirs()

    data_dir = cfg["dataset"]["clean_dir"]
    img_size = cfg["train"]["img_size"]
    batch_size = cfg["train"]["batch_size"]
    val_split = cfg["train"]["val_split"]
    seed = cfg["train"]["seed"]

    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Data generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        horizontal_flip=True,
        validation_split=val_split
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=val_split
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=seed
    )

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    # Class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Model
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(train_gen.num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
        ModelCheckpoint("models/best_model.keras", monitor="val_accuracy", save_best_only=True)
    ]

    # Phase 1: train head
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=focal_loss(),
        metrics=["accuracy"]
    )

    print("\n=== Phase 1: Training classifier head ===")
    model.fit(
        train_gen,
        epochs=12,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    # Phase 2: fine-tune last layers
    base_model.trainable = True
    for layer in base_model.layers[:-80]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=focal_loss(),
        metrics=["accuracy"]
    )

    print("\n=== Phase 2: Fine-tuning last EfficientNet layers ===")
    model.fit(
        train_gen,
        epochs=25,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    # Evaluation
    print("\n🏆 FINAL EVALUATION")
    val_gen.reset()
    pred = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(pred, axis=1)
    y_true = val_gen.classes

    class_names = list(train_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    with open("assets/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, "assets/confusion_matrix.png")

    print("✅ Saved model to: models/best_model.keras")
    print("✅ Saved report to: assets/classification_report.txt")
    print("✅ Saved confusion matrix to: assets/confusion_matrix.png")


if __name__ == "__main__":
    main()