import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, concatenate, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

def create_data_generators(train_dir, validation_dir, target_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical')
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical')

    return train_generator, validation_generator

def build_hybrid_model(resnet_base, vgg_base, num_classes):
    resnet_base.trainable = False
    vgg_base.trainable = False

    # Create input layer
    input_tensor = Input(shape=(224, 224, 3))

    # ResNet branch
    resnet_output = resnet_base(input_tensor, training=False)
    resnet_output = GlobalAveragePooling2D()(resnet_output)

    # VGG16 branch
    vgg_output = vgg_base(input_tensor, training=False)
    vgg_output = GlobalAveragePooling2D()(vgg_output)

    # Concatenate both branches
    merged = concatenate([resnet_output, vgg_output])

    # Add fully connected layers
    x = Dense(512, activation='relu')(merged)
    output_tensor = Dense(num_classes, activation='softmax')(x)

    # Create hybrid model
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

def plot_history(history, metric='accuracy', title='Training and Validation Metrics Over Epochs'):
    if metric in history.history:
        plt.plot(history.history[metric], label=f'Training {metric.upper()}')
    if f'val_{metric}' in history.history:
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.upper()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.upper())
    plt.legend()
    plt.title(title)
    plt.show()

def train_model(model, train_generator, validation_generator, epochs, learning_rate=1e-3):
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])
    history = model.fit(train_generator, validation_data=validation_generator, epochs=epochs)
    return history

def predict_image(model, image_path, target_size=(224, 224)):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return predictions

# Flask application for user interface
app = Flask(__name__)

# Paths for the dataset and model
train_dir = "data2/train"
validation_dir = "data2b/validation"
model_save_path_hybrid = "models/skin_disease_hybrid_model.h5"

# Check if the model exists, if not, train and save it
if not os.path.exists(model_save_path_hybrid):
    # Data Generators
    train_generator, validation_generator = create_data_generators(train_dir, validation_dir)
    num_classes = train_generator.num_classes

    # Load base models
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Build Hybrid model
    hybrid_model = build_hybrid_model(resnet_base, vgg_base, num_classes)

    # Initial Training
    history_hybrid = train_model(hybrid_model, train_generator, validation_generator, epochs=10)
    plot_history(history_hybrid)

    # Fine-Tuning Hybrid Model
    for layer in resnet_base.layers[-10:]:
        layer.trainable = True
    for layer in vgg_base.layers[-4:]:
        layer.trainable = True

    history_hybrid_fine_tune = train_model(hybrid_model, train_generator, validation_generator, epochs=10, learning_rate=1e-5)
    plot_history(history_hybrid_fine_tune, title='Fine-Tuning Training and Validation Metrics Over Epochs (Hybrid Model)')

    # Save the trained Hybrid model
    hybrid_model.save(model_save_path_hybrid)
    print(f"Hybrid Model saved at {model_save_path_hybrid}")
else:
    # Load trained model
    hybrid_model = load_model(model_save_path_hybrid)
    print(f"Loaded Hybrid Model from {model_save_path_hybrid}")

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
            image_location = os.path.join('uploads', secure_filename(image_file.filename))
            image_file.save(image_location)
            predictions = predict_image(hybrid_model, image_location)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = train_generator.class_indices.keys()
            predicted_label = list(predicted_label)[predicted_class]
            medication = medications.get(predicted_class, "Please consult a dermatologist for further assistance.")
            return render_template('result.html', prediction=predicted_label, medication=medication)
    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Create simple HTML templates for the UI
    with open('templates/index.html', 'w') as f:
        f.write('''
        <!doctype html>
        <title>Upload an Image</title>
        <h1>Upload an image of a skin condition</h1>
        <form method="post" enctype="multipart/form-data">
          <input type="file" name="file">
          <input type="submit" value="Upload">
        </form>
        ''')

    with open('templates/result.html', 'w') as f:
        f.write('''
        <!doctype html>
        <title>Prediction Result</title>
        <h1>Prediction: {{ prediction }}</h1>
        <h2>Recommended Medication:</h2>
        <p>{{ medication }}</p>
        <a href="/">Go Back</a>
        ''')

    app.run(debug=True)
