# streamlit_app.py

import streamlit as st
import requests

# Function to upload image to Flask backend
def upload_image_to_flask(image_file):
    url = 'http://localhost:5000/upload'
    files = {'file': image_file}
    response = requests.post(url, files=files)
    return response.text

# Streamlit UI
def main():
    st.title("Dzongkha OCR")
    st.write("Upload an image to extract Dzongkha text.")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        if st.button('Extract Text'):
            result = upload_image_to_flask(uploaded_file)
            st.write("Extracted Text:", result)

if __name__ == '__main__':
    main()
