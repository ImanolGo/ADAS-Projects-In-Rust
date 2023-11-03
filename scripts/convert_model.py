from tensorflow import keras
import keras2onnx

model_name = 'traffic_classifier'
# Load your Keras model
model = keras.models.load_model(f'../data/Models/{model_name}.h5')

# Convert the model to ONNX format
onnx_model = keras2onnx.convert_keras(model, model.name)

# Save the ONNX model to disk
keras2onnx.save_model(onnx_model, f'../data/Models/{model_name}.onnx')
