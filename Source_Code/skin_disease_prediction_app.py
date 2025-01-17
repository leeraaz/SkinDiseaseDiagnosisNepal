# type: ignore
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

# Load the saved models
tr_knn = joblib.load('models/hybrid_knn_resnet50_model_128.pkl')
tr_rf = joblib.load('models/hybrid_rf_feature_vgg16_model_128.pkl')
ensemble_model_path = "models/improved_hybrid_model_batch128.keras"
vgg16_model_path = "models/hybrid_model_vgg16_new128_afterFineTune.keras"
resnet50_model_path = "models/hybrid_model_resnet50_new128_afterFineTune.keras"
inceptionv3_model_path = "models/hybrid_model_inceptionV3_new128_afterFineTune.keras"

# Load models
ensemble_model = load_model(ensemble_model_path)
vgg16_model = load_model(vgg16_model_path)
resnet50_model = load_model(resnet50_model_path)
inceptionv3_model = load_model(inceptionv3_model_path)

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

resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
def predict_disease_by_knn(image_path, model, scaler_path=None, class_labels=None):
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

def predict_disease_by_rf(img_array, cnn_model, traditional_model, class_labels=None):
    # Expected feature count for the RF model (from training data)
    expected_features = 128

    # Debug: Print model layers
    print("Available layers:", [layer.name for layer in cnn_model.layers])

    # Extract features using the correct layer of the CNN
    feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('dense').output)
    features = feature_extractor.predict(img_array, verbose=0)

    # Debug: Print extracted feature shape
    print("Extracted feature shape before flattening:", features.shape)

    # Flatten the features
    features = features.reshape(features.shape[0], -1)

    # Debug: Print flattened feature shape
    print("Flattened feature shape:", features.shape)

    # Debug: Print final feature shape after scaling
    print("Final feature shape after scaling:", features.shape)

    # Slice features if dimensions are larger than expected
    if features.shape[1] > expected_features:
        print(f"Slicing feature dimensions from {features.shape[1]} to {expected_features}.")
        features = features[:, :expected_features]

    # Debug: Print final adjusted feature shape
    print("Final feature shape after adjustment:", features.shape)

    # Make predictions
    print("Making predictions with Random Forest...")
    predictions = traditional_model.predict(features)
    
    return predictions



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


            # Standalone CNN Models
            ess_pred_class, ess_confidence = predict_with_cnn(vgg16_model, image_location, target_size=(224, 224))
            vgg16_pred_class, vgg16_confidence = predict_with_cnn(vgg16_model, image_location, target_size=(224, 224))
            resnet50_pred_class, resnet50_confidence = predict_with_cnn(resnet50_model, image_location, target_size=(224, 224))
            inceptionv3_pred_class, inceptionv3_confidence = predict_with_cnn(inceptionv3_model, image_location, target_size=(224, 224))

            # # Traditional ML Models
            knn_trandition =  predict_disease_by_knn(image_location,tr_knn,"models/traditional_scaler.joblib")
            rf_trandition =  predict_disease_by_rf(img_array, vgg16_model, tr_rf)

            # Find the Best Model
            model_confidences = {
                "Hybrid Mode": ess_confidence,
                "VGG16": vgg16_confidence,
                "ResNet50": resnet50_confidence,
                "InceptionV3": inceptionv3_confidence
            }
            best_model = max(model_confidences, key=model_confidences.get)

            # Get results
            cnn_label = labels[ess_pred_class]
            vgg16_label = labels[vgg16_pred_class]
            resnet50_label = labels[resnet50_pred_class]
            inceptionv3_label = labels[inceptionv3_pred_class]
            
            print("THis is what I cogt"+ knn_trandition)
            knn_label = labels[int(knn_trandition)]
            rf_label = labels[int(rf_trandition)]

            # Medication
            medication = medications.get(inceptionv3_pred_class, "Consult a dermatologist.")

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
    app.run(debug=True, host='0.0.0.0', port=5001)

