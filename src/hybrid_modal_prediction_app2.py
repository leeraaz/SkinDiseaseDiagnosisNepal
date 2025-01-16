import os
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
import joblib
import re
from tensorflow.keras.applications import ResNet50

# Flask application setup
app = Flask(__name__, template_folder="templates", static_folder="src/static")

# Model paths
MODEL_PATHS = {
    # "cnn": "models/improved_hybrid_model.keras",
    "vgg16": "models/hybrid_model_vgg16_new128_afterFineTune.keras",
    "resnet50": "models/hybrid_model_train_resnet.keras",
    # "inceptionv3": "models/Improved_inceptionV3_model.keras",
    "rf": "models/hybrid_rf_vgg16_model_128.pkl",
    "knn": "models/hybrid_knn_vgg16_model_128.pkl",
    "res_rf": "models/hybrid_rf_res_model_7030.pkl",
    "pca": "models/vgg7030_pca_model.pkl",
    # "scaler": "models/traditional_scaler.joblib",
}

# Load models and utilities
# cnn_model = load_model(MODEL_PATHS["cnn"])
vgg16_model = load_model(MODEL_PATHS["vgg16"])
resnet50_model = load_model(MODEL_PATHS["resnet50"])
# inceptionv3_model = load_model(MODEL_PATHS["inceptionv3"])
rf_model = joblib.load(MODEL_PATHS["rf"])
res_rf_model = joblib.load(MODEL_PATHS["res_rf"])
pca = joblib.load(MODEL_PATHS["pca"])
# scaler = joblib.load(MODEL_PATHS["scaler"])
# resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Labels and medication suggestions
LABELS = ["Acne", "Eczema", "Fungal Infection", "Malignant"]
MEDICATIONS = {
    0: "Consult a dermatologist. Oral antibiotics might be required.",
    1: "Use hydrocortisone cream to reduce inflammation. Avoid allergens.",
    2: "Apply topical antifungal cream. Keep the area clean and dry.",
    3: "Moisturize regularly. Use over-the-counter emollients."
}

# Utility functions
def preprocess_image(image_path, target_size):
    """Preprocess the input image to match the model's required input size."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_with_cnn(model, image_path, target_size):
    """Predict using a CNN model after resizing the image to the required size."""
    img_array = preprocess_image(image_path, target_size)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions)) * 100
    return predicted_class, confidence

def extract_features_with_cnn(cnn_model, image_path, target_size):
    """Extract features from the CNN's intermediate layer."""
    img_array = preprocess_image(image_path, target_size)
    feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
    features = feature_extractor.predict(img_array)
    features_flat = features.reshape(1, -1)
    return features_flat.flatten()

def adjust_feature_dimensions(features, expected_dim):
    """Adjust feature dimensions to match the expected size for PCA."""
    current_dim = features.shape[0]
    if current_dim < expected_dim:
        padding = np.zeros(expected_dim - current_dim)
        return np.hstack((features, padding))
    return features[:expected_dim]

def apply_pca(features, pca_model, expected_dim=41472):
    """Apply PCA transformation to reduce feature dimensions."""
    adjusted_features = adjust_feature_dimensions(features, expected_dim)
    return pca_model.transform(adjusted_features.reshape(1, -1))

def predict_with_rf(image_path, cnn_model, rf_model, pca_model, target_size, expected_dim=41472):
    """Predict using RandomForest with features extracted from a CNN model."""
    features = extract_features_with_cnn(cnn_model, image_path, target_size)
    features = apply_pca(features, pca_model, expected_dim)
    return rf_model.predict(features)[0]

resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
def predict_disease_by_tdm(image_path, model, scaler_path=None, class_labels=None):
    """
    Predict the disease from an image using a trained model (KNN or Random Forest).
    
    Args:
        image_path (str): Path to the image.
        model: Trained classification model (KNN or Random Forest).
        scaler: Scaler used to normalize features (optional, for KNN).
        class_labels (dict): Mapping of class indices to class names.
        
    Returns:
        prediction (str): Predicted class label (name).
    """
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Extract features using ResNet50
    resnet_features = resnet.predict(img_array)
    resnet_features_flat = resnet_features.reshape(1, -1)
    
    # # Load scaler if path is provided and scale features
    # if scaler_path:
    #     scaler = load(scaler_path)
    #     resnet_features_flat = scaler.transform(resnet_features_flat)
    
    # Make a prediction
    prediction = model.predict(resnet_features_flat)
    predicted_class_index = prediction[0]  # Extract the predicted class index
    
    # Map class index to class name
    if class_labels:
        predicted_class = class_labels.get(predicted_class_index, "Unknown Class")
    else:
        predicted_class = str(predicted_class_index)  # Fallback to index if no mapping provided
    
    return predicted_class

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['file']
        if image_file:
            upload_dir = "src/static/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            filename = secure_filename(image_file.filename)
            filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
            image_location = os.path.join(upload_dir, filename)
            image_file.save(image_location)

            # Predictions
            vgg_pred, vgg_conf = predict_with_cnn(vgg16_model, image_location, target_size=(299, 299))
            res_pred, res_conf = predict_with_cnn(resnet50_model, image_location, target_size=(224, 224))
            # rf_pred = predict_with_rf(image_location, vgg16_model, rf_model, pca, target_size=(299, 299), expected_dim=41472)
            # res_rf_pred = predict_with_rf(image_location, resnet50_model, res_rf_model, pca, target_size=(224, 224), expected_dim=41472)

            # rf_trandition =  predict_disease_by_tdm(image_location,res_rf_model)

            # Find the Best Model
            model_confidences = {
                # "Hybrid Model": hybrid_confidence,
                "VGG16": vgg_conf,
                "ResNet50": res_conf,
                # "InceptionV3": inceptionv3_confidence
            }
            best_model = max(model_confidences, key=model_confidences.get)

            # Render result
            return render_template(
                'result.html',
                vgg16_label=LABELS[vgg_pred],
                resnet50_label=LABELS[res_pred],
                # rf_label=LABELS[rf_trandition],
                rf_pred=res_rf_model,
                model_confidences=model_confidences,
                medication=MEDICATIONS.get(vgg_pred, "Consult a dermatologist."),
                image_url=url_for('static', filename=f'uploads/{filename}')
            )
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs("src/static/uploads", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5010)
