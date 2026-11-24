import os
import tensorflow as tf
import keras
from keras import layers, models
from keras.applications import MobileNetV2
import tf2onnx

OLD_MODEL_PATH = "models/baseline_model.keras"
ONNX_OUT_PATH = "web-vercel/public/model/waste_model.onnx"

if __name__ == "__main__":
    print("Memuat model baseline terlatih...")
    old_model = keras.models.load_model(OLD_MODEL_PATH)

    # Bangun ulang arsitektur bersih yang menerima input tensor [-1, 1]
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,
    )
    inputs = layers.Input(shape=(224, 224, 3), name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(6, activation="softmax", name="predictions")(x)

    clean_model = models.Model(inputs, outputs)
    clean_model.set_weights(old_model.get_weights())

    os.makedirs(os.path.dirname(ONNX_OUT_PATH), exist_ok=True)
    spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input_image"),)

    model_proto, _ = tf2onnx.convert.from_keras(
        clean_model,
        input_signature=spec,
        opset=13,
        output_path=ONNX_OUT_PATH,
    )
    print(f"Berhasil! Model ONNX disimpan di: {ONNX_OUT_PATH}")