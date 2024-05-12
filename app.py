from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load the OCR model
model_path = 'dzongkha_trail2.h5'
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")

# Define Dzongkha words mapping
dzongkha_words = {
    0: 'ཀ', 1: 'ཁ', 2: 'ག', 3: 'ང', 4: 'ཅ', 5: 'ཆ', 6: 'ཇ', 7: 'ཉ', 8: 'ཏ', 9: 'ཐ',
    10: 'ད', 11: 'ན', 12: 'པ', 13: 'ཕ', 14: 'བ', 15: 'མ', 16: 'ཙ', 17: 'ཚ', 18: 'ཛ',
    19: 'ཝ', 20: 'ཞ', 21: 'ཟ', 22: 'འ', 23: 'ཡ', 24: 'ར', 25: 'ལ', 26: 'ཤ', 27: 'ས',
    28: 'ཧ', 29: 'ཨ'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform OCR using the model
        try:
            extracted_text = perform_ocr(filepath)
            # Empty the upload folder after OCR
            empty_upload_folder()
            return render_template('result.html', result=extracted_text)
        except Exception as e:
            # Empty the upload folder in case of error
            empty_upload_folder()
            return f'Error performing OCR: {str(e)}'

    return 'Invalid file format'

def preprocess_image(image_path):
    """Preprocesses an image for text extraction."""
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_image = cv2.bitwise_not(binary_image)
    img = Image.fromarray(binary_image)
    resized_image = img.resize((64, 64))
    resized_image_rgb = resized_image.convert('RGB')
    image_array = np.array(resized_image_rgb)
    image_array = image_array / 255.0
    return image_array

def perform_ocr(image_path):
    preprocessed_image_rgb = preprocess_image(image_path)
    prediction = model.predict(np.expand_dims(preprocessed_image_rgb, axis=0))
    top_indices = np.argsort(prediction[0])[-1:][::-1]  # Extract the top prediction
    top_word_index = top_indices[0]
    extracted_text = dzongkha_words[top_word_index]
    return extracted_text

def empty_upload_folder():
    # Get the list of files in the upload folder
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    # Iterate over each file and delete it
    for file in files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        os.remove(file_path)

# Error handling
@app.errorhandler(500)
def internal_server_error(error):
    return 'Internal Server Error', 500

if __name__ == '__main__':
    app.run(debug=True)
