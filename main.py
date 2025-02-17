import os
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.densenet import preprocess_input


app = Flask(__name__)

model = tf.keras.models.load_model('heart_attack_prediction_model.h5')

def get_className(prediction):
    predicted_class = np.argmax(prediction)
    if predicted_class == 0:
        return "No"  # Update class names accordingly

    elif predicted_class == 1:
        return "Mild"
    elif predicted_class == 2:
        return "Moderate"
    elif predicted_class == 3:
        return "Severe"
    else:
        return "Proliferate"

INPUT_SIZE = 224  # Update the input size to match the model

def preprocess_single_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image = preprocess_input(image)

    # Add an extra dimension to match the model input shape
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    return prediction


@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/chart')
def chart():
    return render_template('chart.html')


@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        # Save the uploaded file
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = preprocess_single_image(file_path)
        result = get_className(value)
        return render_template('result.html', result=result)  # Render result template

    return render_template('upload.html')  # Render the upload form



if __name__ == '__main__':
    app.run(debug=True)