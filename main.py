import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import requests
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa
from io import BytesIO
import joblib

 
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

API_URL = os.getenv("HUGGING_API_URL")
token = "Bearer " + os.getenv("HUGGINGFACEHUB_API_TOKEN")
headers = {"Authorization": token}

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


def generate_transcript(audio_byte):
    url = os.getenv("DEEPGRAM_API_URL")
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

    # file = open(filepath, "rb")
    payload = audio_byte
    headers = {
        "Content-Type": "audio/*",
        "Accept": "application/json",
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
    }
    response = requests.post(url, data=payload, headers=headers)
    
    # file.close()
    # os.remove(filepath)

    text = response.json()["results"]["channels"][0]['alternatives'][0]["transcript"]
    print("Transcript:", text)

    return text



def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()


@app.route('/process', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the audio file for speech recognition
        speech = generate_transcript(file_path)
        if speech:
            # Detect hate speech
            llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.0)
            hate_speech_output = llm.invoke(f'Given a Hindi text, check for any hate speech or abus related words in '
                                            f'the text and return True if detected, otherwise return False. Input '
                                            f'text: {speech}')
            return jsonify({'hate_speech': hate_speech_output.content})
    else:
        return jsonify({'error': 'Invalid file type'})

def identify_hate_speech(audio_bytes):
    speech = generate_transcript(audio_bytes)
    if speech:
        # Detect hate speech
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.0)
        hate_speech_output = llm.invoke(f'Given a Hindi text, check for any hate speech or abus related words in '
                                        f'the text and return True if detected, otherwise return False. Input '
                                        f'text: {speech}')
        return hate_speech_output.content


def identify_screem(audio_bytes):
    prediction = predict_audio_tflite(audio_bytes)
    return prediction == "Scream"


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return "Audio file is missing", 400

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()

    prediction = predict_audio_tflite(audio_bytes)
    scream_detected = prediction == "Scream"
    
    return jsonify({"scream_detected": scream_detected})


def check_for_keywords(transcript, keywords=["safeguard help", "saveguard help", "safeguard health", "safe cuard health", "Safe"]):
    # Convert the transcript to lower case for case-insensitive matching
    transcript_lower = transcript.lower()
   
    # Check each keyword against the transcript
    for keyword in keywords:
        if keyword.lower() in transcript_lower:
            print("Keyword Detected")
            return True  # Return True as soon as any keyword matches
    print("Keyword NOT Detected")
    return False  # Return False if no keywords match

@app.route('/threat', methods=['POST'])
def threat():
    if 'audio' not in request.files:
        return "Audio file is missing", 400

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    screem = identify_screem(audio_bytes)
    hate_speech = identify_hate_speech(audio_bytes)

    transcript = generate_transcript(audio_bytes)
    keywords_detected = check_for_keywords(transcript)
    print("Scream:",screem, "Hate Speech:", hate_speech)

    
    return jsonify({"scream_detected": screem,
                    "hate_speech": hate_speech,
                    "keywords_detected":keywords_detected,
                    "threat": screem or hate_speech or keywords_detected})

    
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
