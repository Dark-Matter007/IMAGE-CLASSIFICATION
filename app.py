import os
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set the path for saving the image
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Check if a file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')  # Render the upload page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400  # Handle case where file is not included in the request
    
    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return "No selected file", 400
    
    # If the file is valid, save it and proceed
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Ensure the directory exists before saving the file
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # Save the file to the static/images folder
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)
        
        # Load the model
        model = load_model('image_classifier.model')

        # Read and process the image for prediction
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (32, 32))  # Resize image to match model input size
        img = np.array([img]) / 255.0  # Normalize the image

        # Make the prediction
        prediction = model.predict(img)
        class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]

        # Render the result page with the classification result
        return render_template('recommendation.html', class_name=class_name)

    return "Invalid file format", 400

if __name__ == '__main__':
    app.run(debug=True)
