from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Class labels
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Preprocess image
def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file).convert("RGB")

    processed = preprocess_image(image)

    prediction = model.predict(processed)
    predicted_index = np.argmax(prediction)

    result = classes[predicted_index]

    # Confidence
    confidence = float(np.max(prediction)) * 100

    # Tumor detection message
    if result == "notumor":
        status = "No Tumor Detected"
    else:
        status = "Tumor Detected"

    return jsonify({
        "prediction": result,
        "confidence": f"{confidence:.2f}%",
        "status": status
    })

if __name__ == "__main__":
    app.run(debug=True)