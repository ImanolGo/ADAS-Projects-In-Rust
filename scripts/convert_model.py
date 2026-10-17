"""Convert the Keras traffic-sign model (.h5) to ONNX for the Rust activities.

The original script used keras2onnx, which is unmaintained and needs
TensorFlow 2.3 / Python <= 3.8, and pointed at "traffic_classifier.h5", a name
that no longer exists in data/Models. tf2onnx is the maintained converter.

    pip install -r scripts/requirements.txt
    python scripts/convert_model.py
"""

from pathlib import Path

import tensorflow as tf
import tf2onnx

MODELS = Path(__file__).resolve().parent.parent / "data" / "Models"
NAME = "traffic-sign-recognition-model"

model = tf.keras.models.load_model(MODELS / f"{NAME}.h5")
# The network takes NHWC 30x30 RGB crops; keep the input name the Rust code has always seen.
spec = (tf.TensorSpec((None, 30, 30, 3), tf.float32, name="conv2d_input"),)
tf2onnx.convert.from_keras(model, input_signature=spec, opset=15, output_path=str(MODELS / f"{NAME}.onnx"))
print(f"Wrote {MODELS / f'{NAME}.onnx'}")
