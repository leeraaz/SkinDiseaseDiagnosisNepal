import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Paths for the dataset
train_dir = "data/train"
validation_dir = "data/validation"
model_save_path = "models/skin_disease_resnet50_model.h5"

# Preprocess the images using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Load the ResNet50 model with pretrained weights, excluding the top classification layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of ResNet50
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dense(4, activation='softmax')  # Adjust to the number of classes
])

# Compile the model with AUC as a metric
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])

# Train the model and capture history
history = model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Plot validation AUC
plt.plot(history.history['auc'], label='Training AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.title('Training and Validation AUC Over Epochs')
plt.show()

# Optionally, unfreeze some layers of ResNet50 for fine-tuning
for layer in base_model.layers[-10:]:  # Unfreeze last 10 layers for fine-tuning
    layer.trainable = True

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])

# Continue training with fine-tuning
history_fine_tune = model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Plot fine-tuned validation AUC
plt.plot(history_fine_tune.history['auc'], label='Training AUC (Fine-Tune)')
plt.plot(history_fine_tune.history['val_auc'], label='Validation AUC (Fine-Tune)')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.title('Fine-Tuning Training and Validation AUC Over Epochs')
plt.show()

# Save the trained model
model.save(model_save_path)
print(f"Model saved at {model_save_path}")
