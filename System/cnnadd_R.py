import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

test_datagen = ImageDataGenerator(rescale=1./255)

# Path to the test directory
test_dir = r'C:\Users\NST37\AI Project\Kaggle\test'

# List of class names
class_names = ["Big Truck", "City Car", "Multi Purpose Vehicle", "Sedan", "Sport Utility Vehicle", "Truck", "van"]

# Input shape of the model
input_shape = (224, 224, 3)

# Create a test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    classes=class_names
)

# Load the saved model
model = keras.models.load_model(r'C:\Users\NST37\AI Project\Kaggle\cnnadd_vehicle_classifier.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

# Make predictions on the test set
y_pred = model.predict(test_generator)
y_true = test_generator.classes

# Convert predictions from one-hot encoding to labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Calculate precision, recall, and F1-score
precision = precision_score(y_true, y_pred_labels, average='weighted')
recall = recall_score(y_true, y_pred_labels, average='weighted')
f1score = f1_score(y_true, y_pred_labels, average='weighted')

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1score:.4f}")

# Generate classification report
class_report = classification_report(y_true, y_pred_labels, target_names=class_names)
print(class_report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_labels)
print(conf_matrix)
