from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained models
model1 = load_model(r'C:\Users\NST37\AI Project\Kaggle\cnnadd_vehicle_classifier.h5')
model2 = load_model(r'C:\Users\NST37\AI Project\Kaggle\cnnact_vehicle_classifier.h5')

# Define the image size the models were trained on
img_size = (224, 224)

# Create an instance of the Flask class
app = Flask(__name__, static_folder='C:/Users/NST37/AI Project/Kaggle/static')


@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded file and algorithm from the request
    file = request.files['file']
    algorithm = request.form['algorithm']
    
    # Load and preprocess the image
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.static_folder, filename)
    file.save(file_path)
    img = image.load_img(file_path, target_size=img_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    
    # Show the uploaded image in the response
    image_src = f'static/{filename}'
    spinner = 'show'
    # Use the appropriate model to make a prediction
    if algorithm == 'model1':
        pred = model1.predict(img)
    elif algorithm == 'model2':
        pred = model2.predict(img)
    else:
        return jsonify({'error': 'Invalid algorithm selection'})
    
    # Get the predicted class label
    class_idx = np.argmax(pred)
    if algorithm == 'model1':
        if class_idx == 0:
            class_label = 'Big Truck'
        elif class_idx == 1:
            class_label = 'City Car'
        elif class_idx == 2:
            class_label = 'Multi Purpose Vehicle'
        elif class_idx == 3:
            class_label = 'Sedan'
        elif class_idx == 4:
            class_label = 'Sport Utility Vehicle'
        elif class_idx == 5:
            class_label = 'Truck'
        else:
            class_label = 'van'
    elif algorithm == 'model2':
        if class_idx == 0:
            class_label = 'Big Truck'
        elif class_idx == 1:
            class_label = 'City Car'
        elif class_idx == 2:
            class_label = 'Multi Purpose Vehicle'
        elif class_idx == 3:
            class_label = 'Sedan'
        elif class_idx == 4:
            class_label = 'Sport Utility Vehicle'
        elif class_idx == 5:
            class_label = 'Truck'
        else:
            class_label = 'van'
    
    spinner = 'hide'
    return jsonify({'result': class_label, 'image_src': image_src, 'spinner': spinner})
