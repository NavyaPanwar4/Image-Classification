import os
import sys
import argparse
import numpy as np
import cv2

LABELS_PATH = "imagenet_labels.txt"
LABELS_URL  = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)

# ── Label grouping ────────────────────────────────────────────────────────────

LABEL_GROUPS = {
    "vehicle": [
        "car", "truck", "bus", "van", "jeep", "cab", "taxi", "ambulance",
        "fire engine", "minivan", "sports car", "convertible", "racer",
        "limousine", "pickup", "tow truck", "garbage truck", "snowplow",
        "motorcycle", "moped", "bicycle", "tricycle", "scooter", "go-kart",
    ],
    "animal": [
        "dog", "cat", "bird", "fish", "snake", "lizard", "frog", "turtle",
        "bear", "elephant", "lion", "tiger", "cheetah", "leopard", "horse",
        "zebra", "giraffe", "camel", "deer", "rabbit", "hamster", "squirrel",
        "fox", "wolf", "raccoon", "panda", "koala", "kangaroo", "monkey",
        "gorilla", "parrot", "flamingo", "penguin", "eagle", "owl", "duck",
        "goldfish", "shark", "whale", "dolphin", "lobster", "crab",
        "butterfly", "bee", "spider", "ant",
    ],
    "food": [
        "pizza", "burger", "sandwich", "hot dog", "taco", "sushi", "noodle",
        "soup", "salad", "steak", "bread", "bagel", "pretzel", "croissant",
        "waffle", "pancake", "ice cream", "cake", "cupcake", "cookie",
        "donut", "apple", "banana", "orange", "strawberry", "grape",
        "watermelon", "pineapple", "mango", "broccoli", "carrot", "mushroom",
        "potato", "egg", "cheese", "coffee", "wine", "beer",
    ],
    "flower": [
        "rose", "daisy", "tulip", "sunflower", "orchid", "lily", "lotus",
        "poppy", "lavender", "hibiscus", "daffodil", "flower", "blossom",
        "bouquet", "petal",
    ],
    "plant": [
        "tree", "bush", "cactus", "fern", "grass", "moss", "vine",
        "bamboo", "palm", "pine", "oak", "maple", "plant", "leaf",
        "garden", "forest",
    ],
    "furniture": [
        "chair", "sofa", "couch", "table", "desk", "bed", "wardrobe",
        "cabinet", "shelf", "lamp", "mirror", "clock", "refrigerator",
        "microwave", "oven", "toaster", "sink", "bathtub", "toilet",
    ],
    "electronics": [
        "laptop", "computer", "keyboard", "mouse", "monitor", "television",
        "phone", "mobile", "camera", "headphones", "speaker", "remote",
        "printer", "projector", "tablet",
    ],
    "clothing": [
        "shirt", "jacket", "coat", "dress", "jeans", "shoe", "boot",
        "sneaker", "hat", "cap", "helmet", "glove", "scarf", "tie",
    ],
    "nature": [
        "mountain", "valley", "beach", "lake", "river", "waterfall",
        "desert", "field", "rock", "cloud", "sky", "sunset", "snow", "ice",
    ],
    "sports": [
        "basketball", "soccer", "football", "tennis", "golf", "baseball",
        "volleyball", "skateboard", "surfboard", "snowboard", "ski",
        "racket", "bat", "bicycle",
    ],
}

_KW_TO_GROUP = {}
for _g, _kws in LABEL_GROUPS.items():
    for _kw in _kws:
        _KW_TO_GROUP[_kw.lower()] = _g


def get_group(label: str):
    ll = label.lower()
    if ll in _KW_TO_GROUP:
        return _KW_TO_GROUP[ll]
    for kw, grp in _KW_TO_GROUP.items():
        if kw in ll:
            return grp
    return None


# ── Labels ────────────────────────────────────────────────────────────────────

def _download(url, dest, label):
    import urllib.request
    print(f"[↓] Downloading {label} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"done ({os.path.getsize(dest)/1024:.0f} KB)")
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"Download failed: {e}")


def load_labels():
    if not os.path.exists(LABELS_PATH):
        _download(LABELS_URL, LABELS_PATH, "ImageNet class labels")
    with open(LABELS_PATH) as f:
        return [line.strip() for line in f]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    """
    Load EfficientNet-B3 with pretrained ImageNet weights via torchvision.
    Falls back to MobileNetV2 if EfficientNet is unavailable.
    Weights are cached by torchvision in ~/.cache/torch/hub/checkpoints/
    and downloaded automatically on first run.
    """
    try:
        import torch
        import torchvision.models as models
        from torchvision.models import EfficientNet_B3_Weights

        print("[*] Loading EfficientNet-B3 (downloading weights on first run ~48 MB) ...")
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        model   = models.efficientnet_b3(weights=weights)
        model.eval()
        transform = weights.transforms()
        print("[✓] EfficientNet-B3 ready")
        return model, transform, "EfficientNet-B3"

    except Exception as e:
        print(f"[!] EfficientNet-B3 unavailable ({e}), falling back to MobileNetV2 ...")
        import torch
        import torchvision.models as models
        from torchvision.models import MobileNet_V2_Weights

        weights   = MobileNet_V2_Weights.IMAGENET1K_V1
        model     = models.mobilenet_v2(weights=weights)
        model.eval()
        transform = weights.transforms()
        print("[✓] MobileNetV2 ready")
        return model, transform, "MobileNetV2"


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(image_bgr, transform):
    """
    Convert OpenCV BGR image to the tensor format expected by the model.
    torchvision's weights.transforms() handles resize, crop, normalisation.
    """
    from PIL import Image
    import torch

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return transform(pil).unsqueeze(0)   # (1, 3, H, W)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model, transform, labels, image_bgr, topk=5):
    """
    Classify a BGR image.
    Returns list of dicts: {label, confidence, group}
    """
    import torch

    tensor = preprocess(image_bgr, transform)
    with torch.no_grad():
        logits = model(tensor)[0]
        probs  = torch.softmax(logits, dim=0)

    top_probs, top_idx = probs.topk(topk)

    results = []
    for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
        lbl = labels[idx]
        results.append({
            "label":      lbl,
            "confidence": round(prob * 100, 2),
            "group":      get_group(lbl),
        })
    return results


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"OpenCV could not read: {path}")
    return img


# ── CLI display ───────────────────────────────────────────────────────────────

def print_results(path, results, model_name=""):
    top = results[0]
    bar_len = 26
    grp_tag = f"  [{top['group']}]" if top['group'] else ""
    print(f"\n{'─'*56}")
    print(f"  Model  : {model_name}")
    print(f"  Image  : {path}")
    print(f"  Result : {top['label']}{grp_tag}")
    print(f"  Conf.  : {top['confidence']:.1f}%")
    print(f"{'─'*56}")
    top_prob = results[0]["confidence"]
    for r in results:
        filled = int((r["confidence"] / max(top_prob, 1)) * bar_len)
        bar    = "█" * filled + "░" * (bar_len - filled)
        grp    = f" [{r['group']}]" if r["group"] else ""
        mark   = " ◀" if r == top else ""
        lbl    = r["label"][:28]
        print(f"  {lbl:28s}  {bar}  {r['confidence']:5.1f}%{grp}{mark}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Image Classifier — EfficientNet-B3 (torchvision)"
    )
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--topk",  type=int, default=5,
                        help="Top-k predictions to show (default: 5)")
    args = parser.parse_args()

    labels              = load_labels()
    model, transform, name = load_model()
    img                 = load_image(args.image)
    results             = predict(model, transform, labels, img, topk=args.topk)
    print_results(args.image, results, model_name=name)


if __name__ == "__main__":
    main()