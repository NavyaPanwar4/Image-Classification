"""
Flask web server for the Image Classifier
------------------------------------------
Serves the frontend and exposes a /classify endpoint.

Usage:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import os
import io
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory

from classifier import load_model, extract_hog, CLASSES, MODEL_PATH, cmd_train

app = Flask(__name__, static_folder=".")

# ── Load or auto-train model on startup ──────────────────────────────────────

def get_model():
    if not os.path.exists(MODEL_PATH):
        print("[!] No model.pkl found — training on synthetic data first …")
        cmd_train()
    return load_model()

pipeline = get_model()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/classify", methods=["POST"])
def classify():
    """
    Accepts a JSON body: { "image": "<base64 data URL>" }
    Returns: { "label": str, "confidence": float, "scores": {cls: float} }
    """
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Strip data URL prefix  e.g. "data:image/jpeg;base64,..."
        b64 = data["image"].split(",", 1)[-1]
        img_bytes = base64.b64decode(b64)

        # Decode to OpenCV BGR array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"error": "Could not decode image"}), 400

        feat = extract_hog(img_bgr).reshape(1, -1)
        label_idx = pipeline.predict(feat)[0]
        proba     = pipeline.predict_proba(feat)[0]

        label      = CLASSES[label_idx]
        confidence = float(round(proba[label_idx] * 100, 1))
        scores     = {CLASSES[i]: float(round(proba[i] * 100, 1)) for i in range(len(CLASSES))}

        return jsonify({"label": label, "confidence": confidence, "scores": scores})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n  Image Classifier running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
