from flask import Flask, request, render_template, jsonify, url_for
import tensorflow as tf
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Initialize Flask app
app = Flask(__name__)


labels = ["Fungal Infection", "Eczema", "Acne", "Maligant"]
# Predefined medication suggestions
medications = {
    0: "Apply topical antifungal cream. Keep the area clean and dry.",
    1: "Use hydrocortisone cream to reduce inflammation. Avoid allergens.",
    2: "Consult a dermatologist. Oral antibiotics might be required.",
    3: "Moisturize regularly. Use over-the-counter emollients."
}
# Paths to saved models
CNN_MODEL_PATH = "models/hybrid_model_vgg16_new128_afterFineTune.keras"
RF_MODEL_PATH = "models/hybrid_rf_vgg16_model_128.pkl"

# Load models
print("Loading CNN model...")
cnn_model = load_model(CNN_MODEL_PATH)
print("Loading Random Forest model...")
with open(RF_MODEL_PATH, 'rb') as file:
    rf_model = pickle.load(file)

# Function to preprocess the image
def preprocess_image(image_path, target_size):
    print("Preprocessing image...")
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # Normalize pixel values
    return image

# Function to extract features from CNN
def extract_features(image, cnn_model):
    print("Extracting features from CNN...")
    features = cnn_model.predict(image, verbose=0)
    return features.reshape(features.shape[0], -1)

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['file']
        if image_file:
            # Save uploaded file
            if not os.path.exists("static/uploads"):
                os.makedirs("static/uploads")
            filename = secure_filename(image_file.filename)
            filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
            image_location = os.path.join("static/uploads", filename)
            image_file.save(image_location)

            # Preprocess image
            img_array = preprocess_image(image_location, target_size=(224, 224))

            # CNN Prediction (Hybrid Model)
            cnn_predictions = cnn_model.predict(img_array)
            cnn_pred_class = np.argmax(cnn_predictions, axis=1)[0]
            hybrid_confidence = float(np.max(cnn_predictions)) * 100

            # # Traditional ML Prediction
            # features = extract_features(img_array, cnn_model)
            # rf_pred_class = rf_model.predict([features])[0]

            # Labels
            cnn_label = labels[cnn_pred_class]
            # rf_label = labels[rf_pred_class]

            # Medication
            medication = medications.get(cnn_pred_class, "Consult a dermatologist.")

            return render_template(
                'result.html',
                cnn_label=cnn_label,
                # rf_label=rf_label,
                medication=medication,
                model_confidences=hybrid_confidence,
                image_url=url_for('static', filename=f'uploads/{filename}')
            )
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    # Start the Flask application
    app.run(debug=True, host='0.0.0.0', port=5001)
