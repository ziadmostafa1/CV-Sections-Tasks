import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

# Dictionary of class names
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

@st.cache_resource
def load_model():
    """Load the plant disease classification model"""
    model = tf.keras.models.load_model('DenseNet121.keras', compile=False)
    return model

def preprocess_image(img):
    """Preprocess image for model prediction using the specified preprocessing steps"""
    # Convert PIL Image to array
    img_array = image.img_to_array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize
    img_array = img_array / 255.
    
    return img_array

def predict_disease(image, model):
    """Make prediction on the input image"""
    # Convert PIL image to size 224x224
    img = image.resize((224, 224))
    
    # Preprocess image
    processed_image = preprocess_image(img)
    
    # Get prediction
    predictions = model.predict(processed_image, verbose=0)
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        (CLASS_NAMES[idx], float(predictions[0][idx])) 
        for idx in top_3_idx
    ]
    
    return top_3_predictions

# Main app
def main():
    st.title("🌿 Plant Disease Classifier")
    st.write("Upload an image of a plant leaf to detect diseases")
    
    # Load model
    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Two columns layout
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            col1.subheader("Uploaded Image")
            col1.image(image, use_column_width=True)
            
            # Make prediction
            with st.spinner('Analyzing image...'):
                predictions = predict_disease(image, model)
            
            # Display results
            col2.subheader("Top 3 Predictions")
            
            # Create a better visualization for predictions
            for i, (class_name, confidence) in enumerate(predictions, 1):
                formatted_name = class_name.replace('___', ' - ')
                confidence_percentage = confidence * 100
                
                # Display prediction with progress bar
                col2.write(f"**{i}. {formatted_name}**")
                col2.progress(confidence)
                col2.write(f"Confidence: {confidence_percentage:.2f}%")
                col2.markdown("---")
            
            # Display raw predictions for debugging
            if st.checkbox("Show raw prediction values"):
                st.write("Raw prediction values:", predictions)
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Add information about the model
    st.markdown("---")
    st.markdown("""
    ### Tips for Better Results:
    1. Use clear, well-lit images
    2. Focus on the affected area of the leaf
    3. Avoid blurry or dark images
    4. Make sure the image is of a plant leaf
    
    ### Note
    This is a diagnostic tool and should not be used as the sole basis for treatment decisions. 
    Always consult with agricultural experts for confirmed diagnoses and treatment plans.
    """)

if __name__ == "__main__":
    main()