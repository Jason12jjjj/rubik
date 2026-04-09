"""Core validation, solving and CV-method helpers for Rubik's Cube."""

from collections import Counter
import numpy as np
import cv2
import kociemba

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLORS = ("White", "Red", "Green", "Yellow", "Orange", "Blue")
FACES  = ("Up", "Left", "Front", "Right", "Back", "Down")

COLOR_TO_FACE = {
    "White":  "U",
    "Red":    "R",
    "Green":  "F",
    "Yellow": "D",
    "Orange": "L",
    "Blue":   "B",
}

# Default HSV reference values for each color  (H, S, V)
DEFAULT_HSV = {
    "White":  (  0,  30, 220),
    "Yellow": ( 30, 160, 200),
    "Orange": ( 12, 200, 240),
    "Red":    (  0, 210, 180),
    "Green":  ( 60, 180, 150),
    "Blue":   (110, 180, 160),
}

# HSV threshold ranges for the simple HSV-Thresholding classifier
#   Each entry: (lower_H, upper_H, lower_S, lower_V)
HSV_RANGES = {
    "Red":    [(  0,  10, 100, 80), (165, 180, 100, 80)],   # red wraps around 180
    "Orange": [( 11,  25, 100, 80)],
    "Yellow": [( 26,  34,  80, 80)],
    "Green":  [( 35,  85,  60, 40)],
    "Blue":   [( 86, 130,  60, 40)],
    "White":  [(  0, 180,   0, 180)],                        # low saturation, high value
}


# ---------------------------------------------------------------------------
# Cube validation
# ---------------------------------------------------------------------------

def validate_cube_state(faces_data):
    """Return (is_valid, message) after strict physical consistency checks."""
    for face in FACES:
        if face not in faces_data:
            return False, f"Missing face: {face}"
        if len(faces_data[face]) != 9:
            return False, f"{face} face has {len(faces_data[face])} stickers, expected 9."

    counts    = Counter()
    locations = {c: set() for c in COLORS}
    for face, stickers in faces_data.items():
        for sticker in stickers:
            if sticker not in COLOR_TO_FACE:
                return False, f"Invalid color '{sticker}' detected on {face} face."
            counts[sticker] += 1
            locations[sticker].add(face)

    issues = []
    for color in COLORS:
        if counts[color] != 9:
            issues.append(f"{color}={counts[color]} (expected 9)")

    if issues:
        suspect_faces = sorted({f for c in COLORS if counts[c] != 9 for f in locations[c]})
        suffix = f" Check faces: {', '.join(suspect_faces)}." if suspect_faces else ""
        return False, "Color count mismatch: " + "; ".join(issues) + "." + suffix

    return True, "Cube state passed validation."


# ---------------------------------------------------------------------------
# Kociemba solver
# ---------------------------------------------------------------------------

def to_kociemba_string(faces_data):
    """Convert face-color dictionary into URFDLB 54-char string."""
    face_order = ("Up", "Right", "Front", "Down", "Left", "Back")
    chars = []
    for face in face_order:
        for color in faces_data[face]:
            chars.append(COLOR_TO_FACE[color])
    return "".join(chars)


def solve_cube(faces_data):
    """Run Kociemba and return the solution string, or '!IMPOSSIBLE_STATE!'."""
    try:
        cube_string = to_kociemba_string(faces_data)
        solution    = kociemba.solve(cube_string)
        return solution
    except ValueError:
        return "!IMPOSSIBLE_STATE!"
    except Exception as exc:
        return f"!ERROR: {exc}!"


# ---------------------------------------------------------------------------
# Color Classification Methods
# ---------------------------------------------------------------------------

def _bgr_to_lab(bgr_pixel):
    """Convert a single BGR pixel (np.uint8 array len 3) to CIE-LAB."""
    px  = np.uint8([[bgr_pixel]])
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0][0]
    return lab.astype(float)


def _hsv_ref_to_lab(hsv_tuple):
    """Convert an HSV reference (H,S,V) → BGR → LAB."""
    px  = np.uint8([[[hsv_tuple[0], hsv_tuple[1], hsv_tuple[2]]]])
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0]
    lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]
    return lab.astype(float)


# ---- Method 1 : CIE-LAB weighted distance (your original method) ----------

def classify_color_lab(bgr_pixel, hsv_refs=None):
    """
    Classify a BGR pixel to a Rubik's color using weighted CIE-LAB distance.

    Weights: L*0.1, a*2.4, b*2.4  — emphasises hue/chroma over lightness,
    making it robust to brightness variation from different cameras / lighting.
    """
    if hsv_refs is None:
        hsv_refs = DEFAULT_HSV
    lab = _bgr_to_lab(bgr_pixel)
    best_color, best_dist = "White", float("inf")
    for name, hsv in hsv_refs.items():
        ref = _hsv_ref_to_lab(hsv)
        dist = np.sqrt(
            0.1 * (lab[0] - ref[0]) ** 2
            + 2.4 * (lab[1] - ref[1]) ** 2
            + 2.4 * (lab[2] - ref[2]) ** 2
        )
        if dist < best_dist:
            best_dist, best_color = dist, name
    return best_color


# ---- Method 2 : HSV Range Thresholding (classical CV) --------------------

def classify_color_hsv(bgr_pixel):
    """
    Classify a BGR pixel using fixed HSV range rules.

    Classic computer vision baseline — works well under controlled lighting,
    but degrades with shadows or non-standard white balance.
    """
    px  = np.uint8([[bgr_pixel]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # White: low saturation, high value
    if s < 60 and v > 160:
        return "White"

    for color, ranges in HSV_RANGES.items():
        if color == "White":
            continue
        for rng in ranges:
            lo_h, hi_h, lo_s, lo_v = rng
            if lo_h <= h <= hi_h and s >= lo_s and v >= lo_v:
                return color

    # Fallback to closest HSV euclidean distance
    best_color, best_dist = "White", float("inf")
    for color, ref_hsv in DEFAULT_HSV.items():
        dH = min(abs(h - ref_hsv[0]), 180 - abs(h - ref_hsv[0]))
        dist = dH * 2 + abs(s - ref_hsv[1]) + abs(v - ref_hsv[2])
        if dist < best_dist:
            best_dist, best_color = dist, color
    return best_color


# ---- Method 3 : K-Nearest Neighbours (Machine Learning) ------------------

def _build_knn_classifier():
    """
    Build a KNN classifier trained on synthetically generated HSV color data.

    We generate ~150 samples per class by adding realistic noise to the
    reference HSV values and converting to LAB feature space, so no external
    dataset is needed.
    """
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(42)
    X, y = [], []

    noise_cfg = {
        "White":  (10, 20, 20),
        "Yellow": ( 8, 25, 25),
        "Orange": ( 8, 25, 25),
        "Red":    ( 8, 25, 25),
        "Green":  ( 8, 25, 25),
        "Blue":   ( 8, 25, 25),
    }

    for color, (h, s, v) in DEFAULT_HSV.items():
        hn, sn, vn = noise_cfg[color]
        for _ in range(200):
            hs = int(np.clip(h + rng.integers(-hn, hn + 1), 0, 179))
            ss = int(np.clip(s + rng.integers(-sn, sn + 1), 0,  255))
            vs = int(np.clip(v + rng.integers(-vn, vn + 1), 0,  255))
            bgr = cv2.cvtColor(np.uint8([[[hs, ss, vs]]]), cv2.COLOR_HSV2BGR)[0][0]
            lab = _bgr_to_lab(bgr)
            X.append(lab.tolist())
            y.append(color)

    clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    clf.fit(X, y)
    return clf


_knn_clf = None  # lazy-loaded singleton


def classify_color_knn(bgr_pixel):
    """
    Classify a BGR pixel using a KNN model trained on synthetic HSV samples.

    This is the Machine Learning method: converts pixel to CIE-LAB feature
    space, then asks the KNN model (k=5) for the nearest-neighbour majority vote.
    """
    global _knn_clf
    if _knn_clf is None:
        _knn_clf = _build_knn_classifier()
    lab = _bgr_to_lab(bgr_pixel)
    return _knn_clf.predict([lab.tolist()])[0]


# ---- Method 4 : Multi-Layer Perceptron (Deep-ish Neural Network) -----------

def _build_mlp_classifier():
    """
    Build an MLP neural-network classifier (2 hidden layers, 128-64 neurons).

    Trained on the same 1 200-sample synthetic LAB dataset as the KNN model,
    but uses back-propagation gradient descent instead of distance search.
    Architecture: LAB(3) → Dense(128,ReLU) → Dense(64,ReLU) → Softmax(6)
    """
    from sklearn.neural_network import MLPClassifier

    rng = np.random.default_rng(7)
    X, y = [], []

    noise_cfg = {
        "White":  (10, 20, 20),
        "Yellow": ( 8, 25, 25),
        "Orange": ( 8, 25, 25),
        "Red":    ( 8, 25, 25),
        "Green":  ( 8, 25, 25),
        "Blue":   ( 8, 25, 25),
    }

    for color, (h, s, v) in DEFAULT_HSV.items():
        hn, sn, vn = noise_cfg[color]
        for _ in range(200):
            hs = int(np.clip(h + rng.integers(-hn, hn + 1), 0, 179))
            ss = int(np.clip(s + rng.integers(-sn, sn + 1), 0, 255))
            vs = int(np.clip(v + rng.integers(-vn, vn + 1), 0, 255))
            bgr = cv2.cvtColor(np.uint8([[[hs, ss, vs]]]), cv2.COLOR_HSV2BGR)[0][0]
            lab = _bgr_to_lab(bgr)
            X.append(lab.tolist())
            y.append(color)

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=800,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    clf.fit(X, y)
    return clf


_mlp_clf = None  # lazy-loaded singleton


def classify_color_mlp(bgr_pixel):
    """
    Classify a BGR pixel using a 2-hidden-layer MLP neural network.

    Architecture: 3 LAB features → Dense(128) → Dense(64) → 6-class output.
    Trained on 1 200 synthetic samples with early stopping.
    """
    global _mlp_clf
    if _mlp_clf is None:
        _mlp_clf = _build_mlp_classifier()
    lab = _bgr_to_lab(bgr_pixel)
    return _mlp_clf.predict([lab.tolist()])[0]


# ---------------------------------------------------------------------------
# Academic comparison utility
# ---------------------------------------------------------------------------

def extract_center_bgr(image_bytes_data):
    """
    Extract the median BGR pixel from a 60x60 centre crop of an image.
    `image_bytes_data` must be raw bytes (not a file-like object).
    """
    arr = np.frombuffer(image_bytes_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    roi = img[max(0, cy - 30):cy + 30, max(0, cx - 30):cx + 30]
    return np.median(roi, axis=(0, 1)).astype(np.uint8)


def compare_methods(samples):
    """
    Run all three classifiers on a list of (bgr_pixel, true_label) tuples.

    Returns a dict with per-method metrics:
        {
          "CIE-LAB Distance": {"accuracy":..., "per_class": {color: {P,R,F1}}},
          "HSV Thresholding": {...},
          "KNN Classifier":   {...},
        }
    """
    from sklearn.metrics import classification_report, accuracy_score

    methods = {
        "CIE-LAB Distance": classify_color_lab,
        "HSV Thresholding": classify_color_hsv,
        "KNN Classifier":   classify_color_knn,
    }

    results = {}
    true_labels = [s[1] for s in samples]

    for method_name, fn in methods.items():
        preds = [fn(s[0]) for s in samples]
        acc   = accuracy_score(true_labels, preds)
        report = classification_report(
            true_labels, preds,
            labels=list(COLORS),
            output_dict=True,
            zero_division=0,
        )
        results[method_name] = {
            "accuracy":  round(acc * 100, 1),
            "per_class": {
                c: {
                    "Precision": round(report.get(c, {}).get("precision", 0) * 100, 1),
                    "Recall":    round(report.get(c, {}).get("recall",    0) * 100, 1),
                    "F1":        round(report.get(c, {}).get("f1-score",  0) * 100, 1),
                }
                for c in COLORS
            },
            "macro_precision": round(report["macro avg"]["precision"] * 100, 1),
            "macro_recall":    round(report["macro avg"]["recall"]    * 100, 1),
            "macro_f1":        round(report["macro avg"]["f1-score"]  * 100, 1),
        }

    return results
