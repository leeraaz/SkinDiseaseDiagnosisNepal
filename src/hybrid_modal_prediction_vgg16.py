import os
import re
import numpy as np
import joblib

from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50

from sklearn.decomposition import PCA

# Flask application setup
app = Flask(__name__, template_folder="templates", static_folder="src/static")

# Model paths
MODEL_PATHS = {
    "vgg16": "models/hybrid_model_vgg16_new128_afterFineTune.keras",
    "rf": "models/hybrid_rf_vgg16_model_128.pkl",
    "knn": "models/hybrid_knn_vgg16_model_128.pkl",
    "pca": "models/vgg7030_pca_model.pkl",
}

# Load models and utilities
vgg16_model = load_model(MODEL_PATHS["vgg16"])
rf_model = joblib.load(MODEL_PATHS["rf"])
knn_model = joblib.load(MODEL_PATHS["knn"])
pca_model = joblib.load(MODEL_PATHS["pca"])

# Labels and medication suggestions
LABELS = ["Acne", "Eczema", "Fungal Infection", "Malignant"]
MEDICATIONS = {
    0: "Consult a dermatologist. Oral antibiotics might be required.",
    1: "Use hydrocortisone cream to reduce inflammation. Avoid allergens.",
    2: "Apply topical antifungal cream. Keep the area clean and dry.",
    3: "Moisturize regularly. Use over-the-counter emollients."
}

############################
#    UTILITY FUNCTIONS    #
############################

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the input image to match the model's required input size.
    Normalizes pixel values to [0,1].
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def extract_features_vgg16(cnn_model, image_path, target_size=(224, 224)):
    """
    Extract features from VGG16’s second-last layer (the fully-connected layer).
    """
    img_array = preprocess_image(image_path, target_size)
    # Create a feature-extractor model from VGG16, ignoring the last (classification) layer
    feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
    features = feature_extractor.predict(img_array)
    return features.reshape(1, -1)

def predict_with_vgg16(cnn_model, image_path, target_size=(224, 224)):
    """
    Predict using the final classification layer of the VGG16 model.
    Returns the predicted class index and confidence score.
    """
    img_array = preprocess_image(image_path, target_size)
    predictions = cnn_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions)) * 100
    return predicted_class, confidence

def predict_with_ml_model(image_path, vgg16_model, ml_model, pca_model, 
                          target_size=(224, 224), expected_dim=41472):
    """
    Predict using an ML model (e.g., RF or KNN) after extracting features from VGG16
    and reducing them with PCA.
    """
    # Extract features from VGG16
    features = extract_features_vgg16(vgg16_model, image_path, target_size)
    
    # Ensure features match expected dimension for PCA
    # (if your pretrained PCA expects a fixed dim, e.g., 41472)
    current_dim = features.shape[1]
    if current_dim < expected_dim:
        # Zero-pad if too few dimensions
        padding = np.zeros((1, expected_dim - current_dim))
        features = np.hstack((features, padding))
    else:
        # Truncate if dimension is larger
        features = features[:, :expected_dim]

    # Apply PCA
    features_pca = pca_model.transform(features)

    # Predict using the ML model (RF or KNN)
    prediction = ml_model.predict(features_pca)
    return prediction[0]

############################
#       FLASK ROUTES      #
############################

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        # Retrieve file from the form
        image_file = request.files['file']
        if image_file:
            # Create upload directory if not exists
            upload_dir = "src/static/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            
            # Clean filename and save
            filename = secure_filename(image_file.filename)
            filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)  # Remove unwanted chars
            image_location = os.path.join(upload_dir, filename)
            image_file.save(image_location)

            # -------------------
            # Make predictions
            # -------------------

            # 1) VGG16 (end-to-end CNN)
            vgg_pred_idx, vgg_confidence = predict_with_vgg16(
                vgg16_model, image_location, target_size=(224, 224)
            )
            vgg_label = LABELS[vgg_pred_idx]
            vgg_medication = MEDICATIONS.get(vgg_pred_idx, "Consult a dermatologist.")

            # 2) Random Forest on VGG16 features
            rf_pred_idx = predict_with_ml_model(
                image_location, vgg16_model, rf_model, pca_model, 
                target_size=(224, 224), expected_dim=41472
            )
            rf_label = LABELS[rf_pred_idx]
            rf_medication = MEDICATIONS.get(rf_pred_idx, "Consult a dermatologist.")

            # 3) KNN on VGG16 features
            knn_pred_idx = predict_with_ml_model(
                image_location, vgg16_model, knn_model, pca_model, 
                target_size=(224, 224), expected_dim=41472
            )
            knn_label = LABELS[knn_pred_idx]
            knn_medication = MEDICATIONS.get(knn_pred_idx, "Consult a dermatologist.")

            # -------------------
            # Choose how you want to display:
            # Either pick the best or show all.
            # Below code shows all in the template.
            # -------------------
            
            # Render result with all predictions
            return render_template(
                'result.html',
                vgg16_label=vgg_label,
                vgg16_confidence=f"{vgg_confidence:.2f}",
                vgg16_medication=vgg_medication,

                rf_label=rf_label,
                rf_medication=rf_medication,

                knn_label=knn_label,
                knn_medication=knn_medication,

                # You can also pass the raw predictions or additional data
                image_url=url_for('static', filename=f'uploads/{filename}')
            )
    
    # On GET request, show the upload page
    return render_template('index.html')


if __name__ == '__main__':
    # Ensure directories exist for templates & static files
    os.makedirs("src/static/uploads", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5010)
