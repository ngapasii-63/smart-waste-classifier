import os
import shutil
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
DOCS_DIR = "docs"
IMG_SIZE = (224, 224)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SEED = 42


def check_and_collect_images():
    valid_data = {cls: [] for cls in CLASSES}
    corrupt_count = 0

    for cls in CLASSES:
        cls_path = os.path.join(RAW_DIR, cls)
        if not os.path.exists(cls_path):
            raise FileNotFoundError(
                f"Folder kelas '{cls_path}' tidak ditemukan! Pastikan 6 folder kelas ada di dalam data/raw/."
            )

        for fname in os.listdir(cls_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(cls_path, fname)
            img = cv2.imread(fpath)
            if img is None:
                corrupt_count += 1
                continue
            valid_data[cls].append(fpath)

    print("=== RINGKASAN DATASET MENTAH (EDA) ===")
    total = 0
    for cls, files in valid_data.items():
        print(f"- {cls:<10}: {len(files)} gambar")
        total += len(files)
    print(f"Total valid  : {total} gambar")
    print(f"File rusak   : {corrupt_count} file\n")
    return valid_data


def plot_distribution(valid_data):
    counts = [len(valid_data[c]) for c in CLASSES]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(CLASSES, counts, color=["#8D6E63", "#4FC3F7", "#90A4AE", "#FFF176", "#81C784", "#E57373"])
    plt.title("Distribusi Jumlah Citra per Kelas (Dataset TrashNet)")
    plt.xlabel("Kategori Sampah")
    plt.ylabel("Jumlah Gambar")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, count + 10, str(count), ha="center", fontweight="bold")

    os.makedirs(DOCS_DIR, exist_ok=True)
    out_chart = os.path.join(DOCS_DIR, "eda_class_distribution.png")
    plt.tight_layout()
    plt.savefig(out_chart, dpi=150)
    plt.close()
    print(f"Grafik distribusi disimpan ke: {out_chart}")


def split_and_resize(valid_data):
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)

    splits = ["train", "val", "test"]
    for sp in splits:
        for cls in CLASSES:
            os.makedirs(os.path.join(PROCESSED_DIR, sp, cls), exist_ok=True)

    summary = {sp: 0 for sp in splits}

    for cls, files in valid_data.items():
        # Split 70% train, 30% temp (val + test)
        train_files, temp_files = train_test_split(files, test_size=0.30, random_state=SEED)
        # Split 30% temp menjadi 15% val dan 15% test
        val_files, test_files = train_test_split(temp_files, test_size=0.50, random_state=SEED)

        split_map = {"train": train_files, "val": val_files, "test": test_files}

        for sp, sp_files in split_map.items():
            for fpath in sp_files:
                img = cv2.imread(fpath)
                img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
                out_name = os.path.basename(fpath)
                out_path = os.path.join(PROCESSED_DIR, sp, cls, out_name)
                cv2.imwrite(out_path, img_resized)
                summary[sp] += 1

    print("\n=== HASIL PEMBAGIAN & PREPROCESSING (224x224) ===")
    print(f"- Train (70%) : {summary['train']} gambar")
    print(f"- Val   (15%) : {summary['val']} gambar")
    print(f"- Test  (15%) : {summary['test']} gambar")
    print(f"Data siap pakai tersimpan di: {PROCESSED_DIR}/")


if __name__ == "__main__":
    data = check_and_collect_images()
    plot_distribution(data)
    split_and_resize(data)