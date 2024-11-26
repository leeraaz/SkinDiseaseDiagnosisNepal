import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Flask application for user interface
app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Load trained hybrid model
model_save_path_hybrid =  "models/skin_disease_hybrid_model.h5"
if os.path.exists(model_save_path_hybrid):
    hybrid_model = load_model(model_save_path_hybrid)
    print(f"Loaded Hybrid Model from {model_save_path_hybrid}")
else:
    raise FileNotFoundError(f"Model file not found at {model_save_path_hybrid}. Please ensure the model is trained and saved correctly.")

# Predefined medication suggestions
medications = {
    0: "Apply topical antifungal cream. Keep the area clean and dry.",
    1: "Use hydrocortisone cream to reduce inflammation. Avoid allergens.",
    2: "Consult a dermatologist. Oral antibiotics might be required.",
    3: "Moisturize regularly. Use over-the-counter emollients."
}

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['file']
        if image_file:
            # Save uploaded file
            if not os.path.exists(UPLOADS_DIR):
                os.makedirs(UPLOADS_DIR)
            image_location = os.path.join(UPLOADS_DIR, secure_filename(image_file.filename))
            image_file.save(image_location)
            
            # Predict using the hybrid model
            prediction, medication = predict_and_get_medication(hybrid_model, image_location)
            
            # Render result
            return render_template('result.html', prediction=prediction, medication=medication)
    return render_template('index.html')

def predict_and_get_medication(model, image_path, target_size=(224, 224)):
    # Preprocess the image
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Get prediction label and medication suggestion
    labels = ["Fungal Infection", "Allergic Reaction", "Bacterial Infection", "Dry Skin"]
    predicted_label = labels[predicted_class]
    medication = medications.get(predicted_class, "Please consult a dermatologist for further assistance.")

    return predicted_label, medication

if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    if not os.path.exists(UPLOADS_DIR):
        os.makedirs(UPLOADS_DIR)

    # Ensure the 'templates' directory exists
    if not os.path.exists(TEMPLATES_DIR):
        os.makedirs(TEMPLATES_DIR)

    # Create simple HTML templates for the UI
    with open(os.path.join(TEMPLATES_DIR, 'index.html'), 'w') as f:
        f.write('''
        <!doctype html>
        <title>Upload an Image</title>
        <h1>Upload an image of a skin condition</h1>
        <form method="post" enctype="multipart/form-data">
          <input type="file" name="file">
          <input type="submit" value="Upload">
        </form>
        ''')

    with open(os.path.join(TEMPLATES_DIR, 'result.html'), 'w') as f:
        f.write('''
        <!doctype html>
        <title>Prediction Result</title>
        <h1>Prediction: {{ prediction }}</h1>
        <h2>Recommended Medication:</h2>
        <p>{{ medication }}</p>
        <a href="/">Go Back</a>
        ''')

    # Run the Flask app
    app.run(debug=False)