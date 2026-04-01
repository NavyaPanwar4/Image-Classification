"""
Image Classifier using HOG Features + SVM (scikit-learn + OpenCV)
------------------------------------------------------------------
Uses a Support Vector Machine trained on HOG (Histogram of Oriented
Gradients) features extracted from images. Supports 4 categories:
cats, dogs, cars, and planes — using bundled synthetic demo weights.

Usage:
    python classifier.py --image path/to/image.jpg
    python classifier.py --demo          # runs on all sample images
    python classifier.py --train         # retrains the model
"""

import os
import sys
import argparse
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ── Config ────────────────────────────────────────────────────────────────────

IMG_SIZE      = (128, 128)   # resize all images to this before feature extraction
MODEL_PATH    = "model.pkl"  # saved pipeline (scaler + SVM)
CLASSES       = ["cat", "dog", "car", "airplane"]

HOG_PARAMS = dict(
    winSize        = (128, 128),
    blockSize      = (16, 16),
    blockStride    = (8, 8),
    cellSize       = (8, 8),
    nbins          = 9,
)

# ── Feature Extraction ────────────────────────────────────────────────────────

def extract_hog(image_bgr: np.ndarray) -> np.ndarray:
    """
    Resize image and extract HOG (Histogram of Oriented Gradients) features.
    HOG captures edge/texture patterns that are robust to lighting changes.
    Returns a 1-D float32 feature vector.
    """
    img = cv2.resize(image_bgr, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog = cv2.HOGDescriptor(
        HOG_PARAMS["winSize"],
        HOG_PARAMS["blockSize"],
        HOG_PARAMS["blockStride"],
        HOG_PARAMS["cellSize"],
        HOG_PARAMS["nbins"],
    )
    features = hog.compute(gray)          # shape: (N, 1)
    return features.flatten().astype(np.float32)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk; raise FileNotFoundError if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"OpenCV could not read: {path}  (unsupported format?)")
    return img

# ── Model: build / train / save / load ───────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    sklearn Pipeline:
      1. StandardScaler  – zero-mean, unit-variance normalisation
      2. SVC             – RBF-kernel Support Vector Classifier
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, random_state=42)),
    ])


def train(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Fit the pipeline on feature matrix X and integer labels y."""
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    return pipeline


def save_model(pipeline: Pipeline, path: str = MODEL_PATH) -> None:
    joblib.dump(pipeline, path)
    print(f"[✓] Model saved → {path}")


def load_model(path: str = MODEL_PATH) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found at '{path}'.\n"
            "Run:  python classifier.py --train"
        )
    return joblib.load(path)

# ── Synthetic demo dataset ────────────────────────────────────────────────────

def _synthetic_features_for_class(label_idx: int, n: int, rng) -> np.ndarray:
    """
    Generate plausible HOG-like feature vectors per class by adding
    class-specific biases to random noise.  Used only when no real
    image dataset is present so the demo can run out of the box.
    """
    hog_dim = 3780          # HOG descriptor length for 128x128 / 8x8 cells
    bias = np.zeros(hog_dim, dtype=np.float32)

    # Rough class fingerprints: different frequency bands dominate
    step = hog_dim // len(CLASSES)
    bias[label_idx * step : (label_idx + 1) * step] = 1.5

    noise = rng.standard_normal((n, hog_dim)).astype(np.float32) * 0.4
    return noise + bias


def make_synthetic_dataset(n_per_class: int = 80):
    """Return (X, y) arrays built from synthetic feature vectors."""
    rng = np.random.default_rng(0)
    X_parts, y_parts = [], []
    for i in range(len(CLASSES)):
        feats = _synthetic_features_for_class(i, n_per_class, rng)
        X_parts.append(feats)
        y_parts.append(np.full(n_per_class, i, dtype=np.int32))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    # shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

# ── Prediction ────────────────────────────────────────────────────────────────

def predict(pipeline: Pipeline, image_bgr: np.ndarray):
    """
    Classify a single BGR image.
    Returns (label_str, confidence_dict).
    """
    feat = extract_hog(image_bgr).reshape(1, -1)
    label_idx   = pipeline.predict(feat)[0]
    proba       = pipeline.predict_proba(feat)[0]
    label       = CLASSES[label_idx]
    confidence  = {CLASSES[i]: float(round(proba[i], 4)) for i in range(len(CLASSES))}
    return label, confidence


def print_result(path: str, label: str, confidence: dict) -> None:
    bar_len = 30
    print(f"\n{'─'*50}")
    print(f"  Image : {path}")
    print(f"  Result: {label.upper()}  ({confidence[label]*100:.1f}% confidence)")
    print(f"{'─'*50}")
    print("  Class probabilities:")
    for cls, prob in sorted(confidence.items(), key=lambda x: -x[1]):
        filled = int(prob * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        marker = " ◀" if cls == label else ""
        print(f"    {cls:>8s}  {bar}  {prob*100:5.1f}%{marker}")
    print()

# ── CLI ───────────────────────────────────────────────────────────────────────

def cmd_train():
    """Train on synthetic data and save the model."""
    print("[*] Generating synthetic training dataset …")
    X, y = make_synthetic_dataset(n_per_class=100)

    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"[*] Training SVM on {len(X_train)} samples …")
    pipeline = train(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[✓] Test accuracy: {acc*100:.1f}%\n")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    save_model(pipeline)


def cmd_classify(image_path: str):
    """Load model and classify a single image."""
    print(f"[*] Loading model from '{MODEL_PATH}' …")
    pipeline = load_model()

    print(f"[*] Reading image: {image_path}")
    img = load_image(image_path)

    label, confidence = predict(pipeline, img)
    print_result(image_path, label, confidence)


def cmd_demo():
    """
    Run classification on synthetic test vectors (no real images needed).
    Generates one test sample per class and prints predicted vs actual.
    """
    print("[*] Loading model …")
    try:
        pipeline = load_model()
    except FileNotFoundError:
        print("[!] No model found — training first …\n")
        cmd_train()
        pipeline = load_model()

    print("\n[*] Running demo on synthetic test samples …\n")
    rng = np.random.default_rng(99)
    correct = 0
    for i, cls in enumerate(CLASSES):
        feat = _synthetic_features_for_class(i, 1, rng)
        label_idx  = pipeline.predict(feat)[0]
        proba      = pipeline.predict_proba(feat)[0]
        predicted  = CLASSES[label_idx]
        conf       = proba[label_idx] * 100
        ok = "✓" if predicted == cls else "✗"
        print(f"  [{ok}]  Actual: {cls:>8s}  →  Predicted: {predicted:>8s}  ({conf:.1f}%)")
        if predicted == cls:
            correct += 1

    print(f"\n  Demo accuracy: {correct}/{len(CLASSES)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Image Classifier — HOG + SVM (scikit-learn + OpenCV)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  metavar="PATH", help="Classify a single image file")
    group.add_argument("--train",  action="store_true", help="Train and save the model")
    group.add_argument("--demo",   action="store_true", help="Run demo on synthetic samples")
    args = parser.parse_args()

    if args.train:
        cmd_train()
    elif args.demo:
        cmd_demo()
    elif args.image:
        cmd_classify(args.image)


if __name__ == "__main__":
    main()
