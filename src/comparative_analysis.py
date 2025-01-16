import os
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import joblib
from werkzeug.utils import secure_filename
import re

# Flask app
app = Flask(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "models"
UPLOADS_DIR = "static/uploads"
TEMPLATES_DIR = "templates"

# Load models
hybrid_model_path = "models/larger_dataset_hybrid_modal.keras"
knn_model_path = "models/knn_model.pkl"
rf_model_path = "models/random_forest_model.pkl"

hybrid_model = load_model(hybrid_model_path)
knn_model = joblib.load(knn_model_path)
rf_model = joblib.load(rf_model_path)

# Predefined labels and medications
labels = ["Acne", "Eczema", "Fungal Infections", "Malignant"]
medications = {
    "Acne": "Use salicylic acid or benzoyl peroxide. Maintain a healthy skincare routine.",
    "Eczema": "Apply hydrocortisone cream and keep the skin moisturized.",
    "Fungal Infections": "Use topical antifungal creams. Keep the area dry and clean.",
    "Malignant": "Consult a dermatologist immediately for a biopsy and treatment plan."
}

# Helper function: Image preprocessing
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Helper function: Predict using hybrid CNN model
def predict_with_hybrid_model(image_path, cnn_model):
    img_array = preprocess_image(image_path)
    predictions = cnn_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return labels[predicted_class], predictions

# Helper function: Feature extraction
def extract_features(image_path, cnn_model):
    img_array = preprocess_image(image_path)

    resnet_features = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('global_average_pooling2d_3').output).predict(img_array)
    vgg_features = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('global_average_pooling2d_4').output).predict(img_array)
    inception_features = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('global_average_pooling2d_5').output).predict(img_array)

    concatenated_features = np.concatenate([resnet_features.flatten(), vgg_features.flatten(), inception_features.flatten()])
    return concatenated_features

# Prediction with traditional ML models
def predict_with_ml_models(image_path, knn_model, rf_model, cnn_model):
    features = extract_features(image_path, cnn_model)

    if features.shape[0] != knn_model.n_features_in_:
        raise ValueError(f"KNN expects {knn_model.n_features_in_} features, but got {features.shape[0]}")
    if features.shape[0] != rf_model.n_features_in_:
        raise ValueError(f"RF expe scts {rf_model.n_features_in_} features, but got {features.shape[0]}")

    knn_pred = knn_model.predict([features])[0]
    rf_pred = rf_model.predict([features])[0]

    return knn_pred, rf_pred

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['file']
        if image_file:
            if not os.path.exists(UPLOADS_DIR):
                os.makedirs(UPLOADS_DIR)
            filename = secure_filename(image_file.filename)
            filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
            image_path = os.path.join(UPLOADS_DIR, filename)
            image_file.save(image_path)

            # CNN Prediction
            cnn_label, cnn_predictions = predict_with_hybrid_model(image_path, hybrid_model)

            # KNN and RF Predictions
            knn_label_index, rf_label_index = predict_with_ml_models(image_path, knn_model, rf_model, hybrid_model)
            knn_label = labels[knn_label_index]
            rf_label = labels[rf_label_index]

            # Ensemble: Final Prediction
            all_predictions = [cnn_label, knn_label, rf_label]
            final_label = max(set(all_predictions), key=all_predictions.count)

            # Medication suggestion
            medication = medications.get(final_label, "Consult a dermatologist for further assistance.")

            image_url = url_for('static', filename=f'uploads/{filename}')
            return render_template('result.html', prediction=final_label, medication=medication,
                                   cnn_label=cnn_label, knn_label=knn_label,
                                   rf_label=rf_label, image_url=image_url)
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5001)