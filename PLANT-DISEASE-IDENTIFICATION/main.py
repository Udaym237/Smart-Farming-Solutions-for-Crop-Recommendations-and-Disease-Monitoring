import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def model_prediction(test_image):
    # Load the trained model
    model = tf.keras.models.load_model("PLANT-DISEASE-IDENTIFICATION/trained_plant_disease_model.keras")
    
    # Preprocess the input image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    
    # Get predictions
    predictions = model.predict(input_arr)
    confidence = np.max(predictions)  # Confidence score (highest probability)
    predicted_index = np.argmax(predictions)  # Index of the predicted class
    
    return predicted_index, confidence  # Return both the index and confidence score

# Sidebar
st.sidebar.title("AgriWay")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display the image at the top of the page
img = Image.open("PLANT-DISEASE-IDENTIFICATION/Diseases.png")
st.image(img)

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image and st.button("Show Image"):
        st.image(test_image, use_column_width=True)
    
    # Predict button
    if test_image and st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        
        # Get prediction and confidence score
        result_index, confidence = model_prediction(test_image)
        
        # Reading labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        
        # Check confidence and display appropriate message
        if confidence < 0.75:
            st.warning("⚠️ The model is not confident in its prediction. This image may not belong to a plant disease, or the disease may not be in our trained dataset.")
        else:
            st.success(f"Model Prediction: {class_name[result_index]}")
            st.info(f"Prediction Confidence: {confidence * 100:.2f}%")
