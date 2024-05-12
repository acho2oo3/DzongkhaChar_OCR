import streamlit as st
from PIL import Image
import cv2  # Assuming you use OpenCV for preprocessing
import requests  # For receiving data from Flask
from tensorflow.keras.models import load_model
import numpy as np

# Replace with your model path (assuming it's accessible within Streamlit)
model_path = 'dzongkha_trail2.h5'

# Load the model outside the main function for efficiency
model = st.cache(load_model)(model_path)  # Use st.cache for efficiency

# Define Dzongkha words mapping (unchanged)
dzongkha_words = {
    # Define your Dzongkha words mapping here...
    0: 'ཀ', 1: 'ཁ', 2: 'ག', 3: 'ང', 4: 'ཅ', 5: 'ཆ', 6: 'ཇ', 7: 'ཉ', 8: 'ཏ', 9: 'ཐ',
    10: 'ད', 11: 'ན', 12: 'པ', 13: 'ཕ', 14: 'བ', 15: 'མ', 16: 'ཙ', 17: 'ཚ', 18: 'ཛ',
    19: 'ཝ', 20: 'ཞ', 21: 'ཟ', 22: 'འ', 23: 'ཡ', 24: 'ར', 25: 'ལ', 26: 'ཤ', 27: 'ས',
    28: 'ཧ', 29: 'ཨ'
}


def preprocess_image(image):
    # Preprocess the image (unchanged logic)
    image = np.array(image)
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

    # Delete the temporary file (commented out)
    # os.unlink(image)

    return image_array


def perform_ocr(image_path):
    preprocessed_image_rgb = preprocess_image(image_path)
    prediction = model.predict(np.expand_dims(preprocessed_image_rgb, axis=0))
    top_indices = np.argsort(prediction[0])[-1:][::-1]  # Extract the top prediction
    top_word_index = top_indices[0]
    extracted_text = dzongkha_words[top_word_index]

    # Send result back to Flask using a POST request (replace with your Flask URL)
    url = 'http://127.0.0.1:5000/result'  # Replace with your actual URL
    response = requests.post(url, data={'result': extracted_text})
    print(response)

    if response.status_code == 200:
        st.success('Result sent to Flask successfully!')
    else:
        st.error('Error sending result to Flask')

    return extracted_text


def main():
    st.title('OCR Service')

    if 'filepath' in st.session_state:  # Access data from session state
        image_path = st.session_state['filepath']
        print(image_path)

        # Open the image from the received path
        image = Image.open(image_path)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform OCR
        extracted_text = perform_ocr(image_path)

        # Display result on Streamlit for user feedback (optional)
        st.write('Extracted Text (for reference only):')
        st.write(extracted_text)

    else:
        uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

        if uploaded_file is not None:
            image_path = Image.open(uploaded_file)
            st.image(image_path, caption='Uploaded Image', use_column_width=True)

            # Perform OCR
            extracted_text = perform_ocr(image_path)

            # Display result on Streamlit for user feedback (optional)
            st.write('Extracted Text (for reference only):')
            st.write(extracted_text)


if __name__ == '__main__':
    main()
