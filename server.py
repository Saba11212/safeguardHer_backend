from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa
from io import BytesIO
import joblib

app = Flask(__name__)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Function to extract features from audio files using librosa
def extract_features(byte_data, mfcc=True, chroma=True, mel=True):
    y, sr = librosa.load(BytesIO(byte_data), mono=True)
    features = []
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        features.extend(mfccs)
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        features.extend(chroma)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
        features.extend(mel)
    return np.array(features).reshape(1, -1).astype('float32')  # Ensure matching the model's expected input shape and type

# Function to predict whether an audio file contains a scream or not using a TFLite model
def predict_audio_tflite(byte_data):
    feature = extract_features(byte_data)

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], feature)

    # Run the inference
    interpreter.invoke()

    # Extract the output data from the interpreter
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = "Scream" if predicted_index > 0.5 else "Not a Scream"  # Adjust the condition as per your model's output logic
    return predicted_label

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return "Audio file is missing", 400

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()

    prediction = predict_audio_tflite(audio_bytes)
    scream_detected = prediction == "Scream"
    
    return jsonify({"scream_detected": scream_detected})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
