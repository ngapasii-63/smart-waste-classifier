import sys

print("Python      :", sys.version.split()[0])

import numpy as np
print("NumPy       :", np.__version__)

import cv2
print("OpenCV      :", cv2.__version__)

import tensorflow as tf
print("TensorFlow  :", tf.__version__)

import sklearn, matplotlib, streamlit
print("scikit-learn:", sklearn.__version__)
print("matplotlib  :", matplotlib.__version__)
print("Streamlit   :", streamlit.__version__)

# --- Uji OpenCV: buat gambar acak, ubah ke grayscale & resize ---
gambar = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
kecil = cv2.resize(gambar, (224, 224))
print("\nOpenCV OK -> grayscale:", abu.shape, "| resize:", kecil.shape)

# --- Uji TensorFlow: model mini, 1 langkah training pada data acak ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(6, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
x = np.random.rand(8, 224, 224, 3).astype("float32")
y = np.random.randint(0, 6, size=(8,))
model.fit(x, y, epochs=1, verbose=0)
print("TensorFlow OK -> model mini berhasil dilatih 1 epoch")

print("\nGPU terdeteksi:", tf.config.list_physical_devices("GPU") or "tidak ada (normal, kita pakai CPU/Colab)")
print("\nSEMUA OK, siap lanjut!")