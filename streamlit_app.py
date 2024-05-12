import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Load the OCR model
model_path = 'dzongkha_trail2.h5'
model = load_model(model_path)

# Define Dzongkha words mapping
dzongkha_words = {
    0: 'ཀ', 1: 'ཁ', 2: 'ག', 3: 'ང', 4: 'ཅ', 5: 'ཆ', 6: 'ཇ', 7: 'ཉ',
    8: 'ཏ', 9: 'ཐ', 10: 'ད', 11: 'ན', 12: 'པ', 13: 'ཕ', 14: 'བ', 15: 'མ',
    16: 'ཙ', 17: 'ཚ', 18: 'ཛ', 19: 'ཝ', 20: 'ཞ', 21: 'ཟ', 22: 'འ', 23: 'ཡ',
    24: 'ར', 25: 'ལ', 26: 'ཤ', 27: 'ས', 28: 'ཧ', 29: 'ཨ'
}

def preprocess_image(image):
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

def predict(image):
    preprocessed_image_rgb = preprocess_image(image)
    prediction = model.predict(np.expand_dims(preprocessed_image_rgb, axis=0))
    top_indices = np.argsort(prediction[0])[-1:][::-1]
    top_word_index = top_indices[0]
    extracted_text = dzongkha_words[top_word_index]
    return extracted_text

def main():
    st.title("Dzongkha OCR")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        print(f"Received file: {uploaded_file.name}")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is not None:
            predicted_text = predict(image)
            response = Response(predicted_text)
            response.headers.add('Access-Control-Allow-Headers', 'Origin')  # Add this line
            st.write(f"Predicted Text: {predicted_text}")
        else:
            st.write("Invalid image file.")
            
if __name__ == "__main__":
    main()
