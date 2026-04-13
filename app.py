from flask import Flask, request, jsonify
from flask_cors import CORS
import os, mimetypes, hashlib
import cv2
import librosa
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#############################################
# 1. IMAGE MODEL (placeholder: PyTorch CNN) #
#############################################
# Load your pretrained model (example: XceptionNet fine-tuned)
# For now: simulate with random score
def predict_image(filepath):
    # Load image
    img = Image.open(filepath).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0)

    # >>> Replace with your trained model.predict(tensor) <<<
    score = np.random.uniform(0.7, 0.99)
    label = "authentic" if score > 0.85 else "manipulated"
    return label, round(float(score), 3)


#############################################
# 2. VIDEO MODEL (sample frames)            #
#############################################
def predict_video(filepath):
    cap = cv2.VideoCapture(filepath)
    frame_scores = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or count > 10:  # only sample 10 frames
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        tensor = transform(img).unsqueeze(0)
        
        # >>> Replace with real model.predict(tensor) <<<
        frame_scores.append(np.random.uniform(0.7, 0.99))
        count += 1
    cap.release()

    score = float(np.mean(frame_scores))
    label = "authentic" if score > 0.85 else "manipulated"
    return label, round(score, 3)


#############################################
# 3. AUDIO MODEL (spectrogram analysis)     #
#############################################
def predict_audio(filepath):
    y, sr = librosa.load(filepath, sr=16000)
    # Convert to mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # >>> Replace with real model on mel spectrogram <<<
    score = np.random.uniform(0.7, 0.99)
    label = "authentic" if score > 0.85 else "manipulated"
    return label, round(float(score), 3)



#--------------------------------------------

@app.route("/")
def home():
    return """
    <html>
    <head>
        <title>Deepfake Detector</title>
    </head>
    <body style="font-family: Arial; text-align:center; margin-top:50px;">
        <h1>Deepfake Detection System 🚀</h1>
        
        <form action="/api/detect" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <br><br>
            <button type="submit">Upload & Detect</button>
        </form>
    </body>
    </html>
    """


# -------------------------------------------

#############################################
# API ENDPOINT                              #
#############################################

# @app.route("/api/detect", methods=["POST"])
# def detect():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "Empty filename"}), 400

#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)

#     # Detect file type
#     mime_type, _ = mimetypes.guess_type(filepath)
#     if not mime_type:
#         return jsonify({"error": "Unknown file type"}), 400

#     if mime_type.startswith("image/"):
#         file_type = "image"
#         label, score = predict_image(filepath)
#     elif mime_type.startswith("video/"):
#         file_type = "video"
#         label, score = predict_video(filepath)
#     elif mime_type.startswith("audio/"):
#         file_type = "audio"
#         label, score = predict_audio(filepath)
#     else:
#         return jsonify({"error": "Unsupported file type"}), 400

#     return jsonify({
#         "file_type": file_type,
#         "label": label,
#         "score": score
#     })


# -----------------------------------------------

@app.route("/api/detect", methods=["POST"])
def detect():
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    mime_type, _ = mimetypes.guess_type(filepath)

    if mime_type.startswith("image/"):
        label, score = predict_image(filepath)
    elif mime_type.startswith("video/"):
        label, score = predict_video(filepath)
    elif mime_type.startswith("audio/"):
        label, score = predict_audio(filepath)
    else:
        return "Unsupported file type"

    return f"""
    <h2>Result</h2>
    <p>Type: {mime_type}</p>
    <p>Prediction: {label}</p>
    <p>Confidence: {score}</p>
    <br><a href="/">Go Back</a>
    """


# ------------------------------------------------


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
