import os
import numpy as np
import joblib
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA
from werkzeug.utils import secure_filename
import re

# Flask application
app = Flask(__name__, template_folder="templates")

# Load models and PCA
cnn_model_path = "models/improved_hybrid_model.keras"
knn_model_path = "models/improved_knn_model.pkl"
rf_model_path = "models/improved_random_forest_model.pkl"
pca_model_path = "models/pca_model.pkl"

cnn_model = load_model(cnn_model_path)
knn_model = joblib.load(knn_model_path)
rf_model = joblib.load(rf_model_path)
pca = joblib.load(pca_model_path)

# Predefined medication suggestions
medications = {
    0: "Apply topical antifungal cream. Keep the area clean and dry.",
    1: "Use hydrocortisone cream to reduce inflammation. Avoid allergens.",
    2: "Consult a dermatologist. Oral antibiotics might be required.",
    3: "Moisturize regularly. Use over-the-counter emollients."
}

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess the input image for the CNN model."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def extract_features_with_cnn(cnn_model, img_array):
    """Extract features from the CNN's intermediate layer."""
    feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
    features = feature_extractor.predict(img_array)
    return features.flatten()

def predict_with_ml(features, ml_model, pca):
    """Predict using traditional ML models (KNN, RF) with PCA."""
    # Apply PCA to match dimensions
    features_pca = pca.transform(features.reshape(1, -1))
    return ml_model.predict(features_pca)[0]

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
            img_array = preprocess_image(image_location)

            # CNN Prediction
            cnn_predictions = cnn_model.predict(img_array)
            cnn_pred_class = np.argmax(cnn_predictions, axis=1)[0]
            model_confidences = {
                "CNN": float(np.max(cnn_predictions)) * 100  # Confidence percentage
            }

            # Extract features for traditional ML
            feature_vector = extract_features_with_cnn(cnn_model, img_array)
            knn_pred = predict_with_ml(feature_vector, knn_model, pca)
            rf_pred = predict_with_ml(feature_vector, rf_model, pca)

            # Get results
            labels = ["Fungal Infection", "Allergic Reaction", "Bacterial Infection", "Dry Skin"]
            cnn_label = labels[cnn_pred_class]
            knn_label = labels[knn_pred]
            rf_label = labels[rf_pred]

            # Medication
            medication = medications.get(cnn_pred_class, "Consult a dermatologist.")

            return render_template(
                'result.html',
                cnn_label=cnn_label,
                knn_label=knn_label,
                rf_label=rf_label,
                medication=medication,
                model_confidences=model_confidences,
                image_url=url_for('static', filename=f'uploads/{filename}')
            )
    return render_template('index.html')


if __name__ == '__main__':
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    # Create templates for the UI
    with open(os.path.join("templates", 'index.html'), 'w') as f:
        f.write('''
        <!doctype html>
        <html>
        <head><title>Skin Disease Classifier</title></head>
        <body>
        <h1>Upload an image of a skin condition</h1>
        <form method="post" enctype="multipart/form-data">
          <input type="file" name="file" required>
          <button type="submit">Upload</button>
        </form>
        </body>
        </html>
        ''')

    with open(os.path.join("templates", 'result.html'), 'w') as f:
        f.write('''
        <!doctype html>
        <html>
        <head><title>Prediction Result</title></head>
        <body>
        <h1>Prediction Results</h1>
        <h2>CNN Prediction: {{ cnn_label }}</h2>
        <h2>KNN Prediction: {{ knn_label }}</h2>
        <h2>RF Prediction: {{ rf_label }}</h2>
        <h2>Recommended Medication:</h2>
        <p>{{ medication }}</p>
        <h2>Uploaded Image:</h2>
        <img src="{{ image_url }}" alt="Uploaded Image" width="300">
        <br><a href="/">Go Back</a>
        </body>
        </html>
        ''')

    # Start the Flask application
    app.run(debug=True, host='0.0.0.0', port=5001)
