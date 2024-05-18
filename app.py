import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Path to the saved model
path = '/best.keras'

# Load the pre-trained model
best_model = load_model(path)

# Define the class names (ensure the order matches the model's output layer)
class_names = ['begnin', 'malignant', 'non-cancer']  # Replace with your actual class names

# Function to load and preprocess the image
def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to predict the class of an image
def predict_class(model, img_path):
    target_size = (256, 256)
    img_array = load_and_preprocess_image(img_path, target_size)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_idx]
    return predicted_class

# Streamlit app
st.title("Skin Lesion Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Make predictions on the uploaded image
    predicted_class = predict_class(best_model, uploaded_file)

    # Display the predicted class
    st.write("Predicted Class:", predicted_class)
