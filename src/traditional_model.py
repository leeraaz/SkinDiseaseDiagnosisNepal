import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# Function to load and preprocess data for traditional ML
def load_data_for_ml(data_dir):
    X = []
    y = []
    class_labels = sorted(os.listdir(data_dir))  # Ensure consistent label mapping
    label_map = {label: idx for idx, label in enumerate(class_labels)}

    for label in class_labels:
        folder_path = os.path.join(data_dir, label)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (128, 128))  # Resize for consistency
            X.append(img.flatten())  # Flatten for traditional ML
            y.append(label_map[label])
    
    return np.array(X), np.array(y), label_map

# Define paths to your data folders
train_dir = "data/train"
test_dir = "data/test"

# Load train and test data
X_train, y_train, label_map = load_data_for_ml(train_dir)
X_test, y_test, _ = load_data_for_ml(test_dir)

# Split train data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate on validation data
y_val_pred = rf_model.predict(X_val)
print("Validation Report (Random Forest):")
print(classification_report(y_val, y_val_pred))

# Test on test data
y_test_pred = rf_model.predict(X_test)
print("Test Report (Random Forest):")
print(classification_report(y_test, y_test_pred))

# Save the Random Forest model
joblib.dump(rf_model, 'random_forest_skin_disease.pkl')
print("Random Forest Model saved!")

# Train Support Vector Machine (SVM) model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Evaluate SVM on validation data
y_val_pred_svm = svm_model.predict(X_val)
print("Validation Report (SVM):")
print(classification_report(y_val, y_val_pred_svm))

# Test SVM on test data
y_test_pred_svm = svm_model.predict(X_test)
print("Test Report (SVM):")
print(classification_report(y_test, y_test_pred_svm))

# Save the SVM model
joblib.dump(svm_model, 'svm_skin_disease.pkl')
print("SVM Model saved!")
