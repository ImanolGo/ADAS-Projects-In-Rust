import tf2onnx

# Convert the model
model_proto, _ = tf2onnx.convert.from_keras(model_path="../Models/traffic-sign-recognition-model.h5")

# Save the ONNX model
with open("traffic-sign-recognition-model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())
