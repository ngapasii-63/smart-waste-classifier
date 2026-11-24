import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data/processed"
MODELS_DIR = "models"
DOCS_DIR = "docs"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names

    # Optimasi performa pembacaan data di CPU
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_baseline_model(num_classes):
    # Base model MobileNetV2 (pretrained ImageNet), dibekukan (frozen)
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    # Preprocessing bawaan MobileNetV2: mengubah rentang piksel [0, 255] ke [-1, 1]
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="baseline_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_curves(history):
    os.makedirs(DOCS_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Grafik Akurasi
    ax1.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy", marker="o")
    ax1.set_title("Baseline Model - Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Grafik Loss
    ax2.plot(history.history["loss"], label="Train Loss", marker="o")
    ax2.plot(history.history["val_loss"], label="Val Loss", marker="o")
    ax2.set_title("Baseline Model - Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(DOCS_DIR, "baseline_training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Grafik kurva training disimpan ke: {out_path}")


def evaluate_and_save_cm(model, test_ds, class_names):
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n=== CLASSIFICATION REPORT (BASELINE - TEST SET) ===")
    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report_str)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(os.path.join(DOCS_DIR, "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - Baseline Model (Test Set)")
    plt.xlabel("Prediksi Model")
    plt.ylabel("Label Sebenarnya")
    plt.tight_layout()
    cm_path = os.path.join(DOCS_DIR, "baseline_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix disimpan ke: {cm_path}")


if __name__ == "__main__":
    tf.random.set_seed(SEED)
    train_ds, val_ds, test_ds, class_names = load_datasets()
    print("Urutan kelas:", class_names)

    model = build_baseline_model(len(class_names))
    model.summary()

    print("\nMemulai training Model Baseline (10 Epochs)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "baseline_model.keras")
    model.save(model_path)
    print(f"\nModel baseline disimpan ke: {model_path}")

    save_training_curves(history)
    evaluate_and_save_cm(model, test_ds, class_names)