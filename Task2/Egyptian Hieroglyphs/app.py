import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import cv2

# Set page config
st.set_page_config(
    page_title="Egyptian Hieroglyph Classifier",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-box {
        border: 2px dashed #aaa;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-box {
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and label encoder"""
    try:
        model = tf.keras.models.load_model('DenseNet121_hieroglyphs.keras', compile=False)
        label_encoder = joblib.load('label_encoder.joblib')
        return model, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(img):
    """Preprocess image for model prediction using the specified preprocessing steps"""
    # Convert PIL Image to array
    img_array = np.array(img)
    # Resize image to target size
    img_array = cv2.resize(img_array, (224, 224))
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize pixel values to be between 0 and 1
    img_array = img_array / 255.0
    
    return img_array

def get_prediction(model, img, label_encoder):
    """Make prediction on the input image"""
    # Preprocess image
    processed_image = preprocess_image(img)
    
    # Get prediction
    predictions = model.predict(processed_image, verbose=0)
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        (label_encoder.inverse_transform([idx])[0], float(predictions[0][idx])) 
        for idx in top_3_idx
    ]
    
    return top_3_predictions

def main():
    # Load model and label encoder
    model, label_encoder = load_model()
    
    if model is None or label_encoder is None:
        st.error("Failed to load model or label encoder. Please check the files exist.")
        return

    # Title and description
    st.title("Egyptian Hieroglyph Classifier 🏺")
    st.markdown("""
        Upload an image of an Egyptian hieroglyph and the model will classify it.
        The image should contain a single hieroglyph symbol.
    """)

    # Create two columns for upload options
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a single hieroglyph"
        )

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            img = Image.open(uploaded_file)
            col1.image(img, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            with st.spinner('Analyzing image...'):
                predictions = get_prediction(model, img, label_encoder)
            
            # Display results
            col2.subheader("Top 3 Predictions")
            
            # Create a better visualization for predictions
            for i, (class_name, confidence) in enumerate(predictions, 1):
                confidence_percentage = confidence * 100
                
                # Display prediction with progress bar
                col2.write(f"**{i}. {class_name}**")
                col2.progress(confidence)
                col2.write(f"Confidence: {confidence_percentage:.2f}%")
                col2.markdown("---")
            
            # Display raw predictions for debugging
            if st.checkbox("Show raw prediction values"):
                st.write("Raw prediction values:", predictions)
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Add information about the model
    with st.expander("About the Model"):
        st.markdown("""
            This classifier uses a MobileNetV2 architecture fine-tuned on Egyptian hieroglyphs.
            The model can recognize various hieroglyphic symbols commonly found in ancient Egyptian writing.
            
            **Tips for best results:**
            - Ensure good lighting
            - Center the hieroglyph in the image
            - Use a solid background if possible
            - Avoid multiple hieroglyphs in one image
        """)

if __name__ == "__main__":
    main()