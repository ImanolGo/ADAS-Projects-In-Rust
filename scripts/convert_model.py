from tensorflow import keras
import keras2onnx

# Load your Keras model
model = keras.models.load_model('../data/Models/traffic-sign-recognition-model.h5')

# Convert the model to ONNX format
onnx_model = keras2onnx.convert_keras(model, model.name)

# Save the ONNX model to disk
keras2onnx.save_model(onnx_model, '../data/Models/traffic-sign-recognition-model.onnx')
