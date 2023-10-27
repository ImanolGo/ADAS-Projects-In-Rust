import tf2onnx

# Convert the model
model_proto, _ = tf2onnx.convert.from_keras(model_path="my_model2.h5")

# Save the ONNX model
with open("model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())
