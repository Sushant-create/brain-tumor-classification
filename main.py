"""
Brain Tumor MRI Classifier - Flask app.

Serves the dashboard (index.html) and a /predict endpoint that accepts an
uploaded MRI image and returns the predicted tumor class with confidence
scores for all 4 classes.

Run directly:
    python main.py
Or via the cross-platform launcher:
    python launch.py
"""

import os
import numpy as np
from flask import Flask, request, jsonify, render_template

from src.model_architecture import DISPLAY_NAMES
from src.preprocess import preprocess_image_bytes, allowed_file, InvalidImageError

MODEL_PATH = os.environ.get("MODEL_PATH", "model/tumor_classifier.keras")
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload cap

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_model = None
_model_load_error = None


def get_model():
    """Lazily load the Keras model on first request instead of at import
    time, so the web server can start and show a clear error page even if
    the model file hasn't been trained/placed yet."""
    global _model, _model_load_error
    if _model is not None or _model_load_error is not None:
        return _model
    try:
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user via API
        _model_load_error = str(exc)
    return _model


@app.route("/")
def index():
    return render_template("index.html", model_ready=os.path.exists(MODEL_PATH))


@app.route("/health")
def health():
    return jsonify(status="ok", model_present=os.path.exists(MODEL_PATH))


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify(error="No file part in request. Expected form field 'file'."), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify(error="No file selected."), 400

    if not allowed_file(file.filename):
        return jsonify(error="Unsupported file type. Use PNG, JPG, JPEG, BMP or TIFF."), 400

    model = get_model()
    if model is None:
        return jsonify(error=f"Model not available: {_model_load_error or 'not trained yet'}. "
                              f"Run train.py first to produce {MODEL_PATH}."), 503

    try:
        image_array = preprocess_image_bytes(file.read())
    except InvalidImageError as exc:
        return jsonify(error=str(exc)), 400

    predictions = model.predict(image_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))

    return jsonify(
        predicted_class=DISPLAY_NAMES[predicted_idx],
        confidence=float(predictions[predicted_idx]),
        all_scores={name: float(score) for name, score in zip(DISPLAY_NAMES, predictions)},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
