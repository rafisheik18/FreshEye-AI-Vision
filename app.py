from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model('healthy_vs_rotten.h5')

# ✅ Class Labels (kept EXACTLY as yours)
labels = [
    "Apple_Healthy", "Apple_Rotten",
    "Banana_Healthy", "Banana_Rotten",
    "Bellpepper_Healthy", "Bellpepper_Rotten",
    "Carrot_Healthy", "Carrot_Rotten",
    "Cucumber_Healthy", "Cucumber_Rotten",
    "Grape_Healthy", "Grape_Rotten",
    "Guava_Healthy", "Guava_Rotten",
    "Jujube_Healthy", "Jujube_Rotten",
    "Mango_Healthy", "Mango_Rotten",
    "Orange_Healthy", "Orange_Rotten",
    "Pomegranate_Healthy", "Pomegranate_Rotten",
    "Potato_Healthy", "Potato_Rotten",
    "Strawberry_Healthy", "Strawberry_Rotten",
    "Tomato_Healthy", "Tomato_Rotten"
]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- HOME ----------------
@app.route('/')
def index():
    return render_template("index.html")

# ---------------- CAMERA PAGE ----------------
@app.route('/camera')
def camera():
    return render_template("camera.html")

# ---------------- PREVIEW PAGE ----------------
@app.route('/preview', methods=['POST'])
def preview():

    file = request.files.get('file')

    if file and file.filename != "":
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        return render_template("preview.html", image_path=filepath)

    return render_template("index.html")

# ---------------- PREDICTION ----------------
@app.route('/predict', methods=['POST'])
def predict():

    image_path = request.form.get("image_path")

    if image_path:
        filepath = image_path

    else:
        file = request.files.get('file')

        if not file or file.filename == "":
            return render_template(
                "output.html",
                prediction="No file uploaded",
                confidence=0,
                image_path=None
            )

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

    try:
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        # ✅ Prevent label mismatch crash
        if class_index < len(labels):
            predicted_class = labels[class_index]
        else:
            predicted_class = f"Class_{class_index}"

        return render_template(
            "output.html",
            prediction=predicted_class,
            confidence=round(confidence, 2),
            image_path=filepath
        )

    except Exception as e:
        print("Prediction Error:", e)

        return render_template(
            "output.html",
            prediction="Error processing image",
            confidence=0,
            image_path=None
        )

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)