"""
Flask web server — EfficientNet-B3 Image Classifier
----------------------------------------------------
Serves the frontend and exposes /classify endpoint.

Usage:
    python app.py
    Open http://localhost:5000
"""

import os
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from classifier import load_model, load_labels, predict

app = Flask(__name__, static_folder=".")

# ── Load model on startup ─────────────────────────────────────────────────────

print("[*] Loading model (downloading on first run) ...")
LABELS               = load_labels()
MODEL, TRANSFORM, MODEL_NAME = load_model()
print(f"[✓] {MODEL_NAME} ready — {len(LABELS)} classes")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/classify", methods=["POST"])
def classify():
    """
    POST { "image": "<base64 data URL>", "topk": 5 }
    Returns {
        "label": str,
        "confidence": float,
        "group": str | null,
        "model": str,
        "results": [{label, confidence, group}, ...]
    }
    """
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        b64     = data["image"].split(",", 1)[-1]
        nparr   = np.frombuffer(base64.b64decode(b64), np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"error": "Could not decode image"}), 400

        topk    = int(data.get("topk", 5))
        results = predict(MODEL, TRANSFORM, LABELS, img_bgr, topk=topk)

        return jsonify({
            "label":      results[0]["label"],
            "confidence": results[0]["confidence"],
            "group":      results[0]["group"],
            "model":      MODEL_NAME,
            "results":    results,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n  Open http://localhost:5000\n")
    app.run(debug=False, port=5000)