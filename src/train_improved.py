import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers, models
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data/processed"
MODELS_DIR = "models"
DOCS_DIR = "docs"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def load_datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names

    # Hitung class_weight untuk mengatasi ketidakseimbangan kelas (terutama 'trash')
    y_train = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight_dict = dict(enumerate(weights))
    print("\n=== BOBOT KELAS (CLASS WEIGHTS) ===")
    for idx, cls in enumerate(class_names):
        print(f"- {cls:<10}: {class_weight_dict[idx]:.4f}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names, class_weight_dict


def build_improved_model(num_classes):
    # Augmentasi data langsung di dalam arsitektur (aktif hanya saat training)
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ],
        name="data_augmentation",
    )

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="improved_mobilenetv2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def save_combined_curves(hist1, hist2):
    os.makedirs(DOCS_DIR, exist_ok=True)
    acc = hist1.history["accuracy"] + hist2.history["accuracy"]
    val_acc = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]
    loss = hist1.history["loss"] + hist2.history["loss"]
    val_loss = hist1.history["val_loss"] + hist2.history["val_loss"]
    warmup_end = len(hist1.history["accuracy"]) - 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(acc, label="Train Accuracy", marker="o")
    ax1.plot(val_acc, label="Val Accuracy", marker="o")
    ax1.axvline(x=warmup_end, color="gray", linestyle="--", label="Mulai Fine-Tuning")
    ax1.set_title("Improved Model - Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(loss, label="Train Loss", marker="o")
    ax2.plot(val_loss, label="Val Loss", marker="o")
    ax2.axvline(x=warmup_end, color="gray", linestyle="--", label="Mulai Fine-Tuning")
    ax2.set_title("Improved Model - Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(DOCS_DIR, "improved_training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Grafik kurva training improved disimpan ke: {out_path}")


def evaluate_and_compare(model, test_ds, class_names):
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n=== CLASSIFICATION REPORT (IMPROVED - TEST SET) ===")
    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report_str)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(os.path.join(DOCS_DIR, "improved_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - Improved Model (Test Set)")
    plt.xlabel("Prediksi Model")
    plt.ylabel("Label Sebenarnya")
    plt.tight_layout()
    cm_path = os.path.join(DOCS_DIR, "improved_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix improved disimpan ke: {cm_path}")

    # Perbandingan langsung dengan Baseline jika file baseline_metrics.json ada
    baseline_json = os.path.join(DOCS_DIR, "baseline_metrics.json")
    if os.path.exists(baseline_json):
        with open(baseline_json, "r", encoding="utf-8") as f:
            base = json.load(f)
        print("\n=== PERBANDINGAN BASELINE VS IMPROVED (TEST SET) ===")
        print(f"Accuracy     : {base['accuracy']:.4f}  -->  {report_dict['accuracy']:.4f}")
        print(f"Macro F1     : {base['macro avg']['f1-score']:.4f}  -->  {report_dict['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1  : {base['weighted avg']['f1-score']:.4f}  -->  {report_dict['weighted avg']['f1-score']:.4f}")
        print(f"F1 'trash'   : {base['trash']['f1-score']:.4f}  -->  {report_dict['trash']['f1-score']:.4f}")


if __name__ == "__main__":
    tf.random.set_seed(SEED)
    train_ds, val_ds, test_ds, class_names, class_weights = load_datasets()

    model, base_model = build_improved_model(len(class_names))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    print("\n--- TAHAP 1: Warm-up Classification Head (8 Epochs) ---")
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print("\n--- TAHAP 2: Fine-Tuning 30 Layer Teratas MobileNetV2 (7 Epochs) ---")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=7,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "improved_model.keras")
    model.save(model_path)
    print(f"\nModel improved disimpan ke: {model_path}")

    save_combined_curves(hist1, hist2)
    evaluate_and_compare(model, test_ds, class_names)