import cv2
from tqdm import tqdm
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

input_shape = (224, 224, 3)
class_names = ['Big Truck', 'City Car', 'Multi Purpose Vehicle', 'Sedan', 'Sport Utility Vehicle', 'Truck', 'van']

def predict_and_display(model_path, img_path):
    model = tf.keras.models.load_model(model_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape[:2])
    img = img / 255.0
    label = model.predict(np.expand_dims(img, axis=0))
    label = class_names[np.argmax(label)]
    progress_bar = tqdm(total=100, desc=f'Processing image {img_path}', position=0, leave=True)
    for i in range(101):
        progress_bar.update(i - progress_bar.n)
        progress_bar.set_postfix({'label': label})
        time.sleep(0.01)
    progress_bar.close()
    plt.imshow(img)
    plt.title(f"Predicted Label: {label}")
    plt.show()
predict_and_display(r'C:\Users\NST37\AI Project\Kaggle\cnnadd_vehicle_classifier.h5', r'C:\Users\NST37\AI Project\Kaggle\sample\sample4.jpg')
