import os
import numpy as np
import joblib
from joblib import load
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA
from werkzeug.utils import secure_filename
import re
from tensorflow.keras.applications import ResNet50
from sklearn.preprocessing import StandardScaler

# Flask application
app = Flask(__name__, template_folder="templates")

# Load models and PCA
cnn_model_path = "models/improved_hybrid_model.keras"
knn_model_path = "models/improved_knn_model.pkl"
rf_model_path = "models/improved_random_forest_model.pkl"
pca_model_path = "models/pca_model.pkl"

# Load the saved models
tr_knn = joblib.load('models/improved_knn_model.pkl')
tr_rf = joblib.load('models/improved_random_forest_model.pkl')
vgg16_model_path = "models/Improved_vgg16_model.keras"
resnet50_model_path = "models/Improved_resnet50_model.keras"
inceptionv3_model_path = "models/Improved_inceptionV3_model.keras"

# Load models
cnn_model = load_model(cnn_model_path)
vgg16_model = load_model(vgg16_model_path)
resnet50_model = load_model(resnet50_model_path)
inceptionv3_model = load_model(inceptionv3_model_path)

knn_model = joblib.load(knn_model_path)
rf_model = joblib.load(rf_model_path)
pca = joblib.load(pca_model_path)

# Scale features for KNN
scaler = StandardScaler()

# Predefined medication suggestions
medications = {
    0: "Apply topical antifungal cream. Keep the area clean and dry.",
    1: "Use hydrocortisone cream to reduce inflammation. Avoid allergens.",
    2: "Consult a dermatologist. Oral antibiotics might be required.",
    3: "Moisturize regularly. Use over-the-counter emollients."
}

labels = ["Fungal Infection", "Eczema", "Acne", "Maligant"]

def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_with_cnn(model, img_array, target_size=(224, 224)):
    # Ensure the image is preprocessed to the correct size for the model
    img_array_resized = preprocess_image(image_path=img_array, target_size=target_size)
    predictions = model.predict(img_array_resized)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions)) * 100
    return predicted_class, confidence

def extract_features_with_cnn(cnn_model, img_array):
    feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
    features = feature_extractor.predict(img_array)
    return features.flatten()

def predict_with_ml(features, ml_model, pca):
    # Apply PCA to match dimensions
    features_pca = pca.transform(features.reshape(1, -1))
    return ml_model.predict(features_pca)[0]

resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
def predict_disease_by_tdm(image_path, model, scaler_path=None, class_labels=None):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Extract features using ResNet50
    resnet_features = resnet.predict(img_array)
    resnet_features_flat = resnet_features.reshape(1, -1)
    
    # Load scaler if path is provided and scale features
    if scaler_path:
        scaler = load(scaler_path)
        resnet_features_flat = scaler.transform(resnet_features_flat)
    
    # Make a prediction
    prediction = model.predict(resnet_features_flat)
    predicted_class_index = prediction[0]  # Extract the predicted class index
    
    # Map class index to class name
    if class_labels:
        predicted_class = class_labels.get(predicted_class_index, "Unknown Class")
    else:
        predicted_class = str(predicted_class_index)  # Fallback to index if no mapping provided
    
    return predicted_class


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['file']
        if image_file:
            # Save uploaded file
            if not os.path.exists("src/static/uploads"):
                os.makedirs("src/static/uploads")
            filename = secure_filename(image_file.filename)
            filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
            image_location = os.path.join("src/static/uploads", filename)
            image_file.save(image_location)

            # Preprocess image
            img_array = preprocess_image(image_location, target_size=(224, 224))

            # CNN Prediction (Hybrid Model)
            cnn_predictions = cnn_model.predict(img_array)
            cnn_pred_class = np.argmax(cnn_predictions, axis=1)[0]
            hybrid_confidence = float(np.max(cnn_predictions)) * 100

            # Standalone CNN Models
            vgg16_pred_class, vgg16_confidence = predict_with_cnn(vgg16_model, image_location, target_size=(224, 224))
            resnet50_pred_class, resnet50_confidence = predict_with_cnn(resnet50_model, image_location, target_size=(224, 224))
            inceptionv3_pred_class, inceptionv3_confidence = predict_with_cnn(inceptionv3_model, image_location, target_size=(299, 299))

            # # Traditional ML Models
            # feature_vector = extract_features_with_cnn(cnn_model, img_array)
            # knn_pred = predict_with_ml(feature_vector, knn_model, pca)
            # rf_pred = predict_with_ml(feature_vector, rf_model, pca)
            knn_trandition =  predict_disease_by_tdm(image_location,tr_knn,"models/traditional_scaler.joblib")
            rf_trandition =  predict_disease_by_tdm(image_location,tr_rf)

            # Find the Best Model
            model_confidences = {
                "Hybrid Model": hybrid_confidence,
                "VGG16": vgg16_confidence,
                "ResNet50": resnet50_confidence,
                "InceptionV3": inceptionv3_confidence
            }
            best_model = max(model_confidences, key=model_confidences.get)

            # Get results
            cnn_label = labels[cnn_pred_class]
            vgg16_label = labels[vgg16_pred_class]
            resnet50_label = labels[resnet50_pred_class]
            inceptionv3_label = labels[inceptionv3_pred_class]
            
            print("THis is what I cogt"+ knn_trandition)
            knn_label = labels[int(knn_trandition)]
            rf_label = labels[int(rf_trandition)]

            # Medication
            medication = medications.get(cnn_pred_class, "Consult a dermatologist.")

            return render_template(
                'resultWithGraph.html',
                cnn_label=cnn_label,
                vgg16_label=vgg16_label,
                resnet50_label=resnet50_label,
                inceptionv3_label=inceptionv3_label,
                knn_label=knn_label,
                rf_label=rf_label,
                medication=medication,
                model_confidences=model_confidences,
                best_model=best_model,
                image_url=url_for('static', filename=f'uploads/{filename}')
            )
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs("src/static/uploads", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    # Start the Flask application
    app.run(debug=True, host='0.0.0.0', port=5100)
