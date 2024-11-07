# app.py
import streamlit as st
import tensorflow as tf
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import io

# Setup MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Constants
CLASS_NAMES = ['Prone', 'Supine', 'To-Left', 'To-Right']
IMAGE_SIZE = (224, 224)

# Page config
st.set_page_config(
    page_title="Sleep Posture Classifier",
    page_icon="🛏️",
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('sleep_posture_model.h5')

def process_image(image):
    # Preserve original resolution for display
    original_size = image.size
    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Use higher resolution for visualization
    display_size = (640, 640)  # Larger size for better visualization
    display_image = cv2.resize(image_array, display_size)
    
    # Process smaller image for model
    model_image = cv2.resize(image_array, IMAGE_SIZE)
    
    results = pose.process(cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        # Draw on higher resolution image
        mp_drawing.draw_landmarks(
            display_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        return landmarks.flatten(), cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
    return None, None

# Sidebar
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

# Main UI
st.title("Sleep Posture Classification")
st.write("Upload an image to classify sleep posture")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        # Load model
        model = load_model()
        
        # Load and display image
        image = Image.open(uploaded_file)
        
        # Process image and get landmarks
        with st.spinner("Processing image..."):
            features, annotated_image = process_image(image)
        

        st.subheader("Detected Pose")
        if annotated_image is not None:
            st.image(annotated_image, use_column_width=True)
        
        # Make prediction
        if features is not None:
            prediction = model.predict(np.expand_dims(features, axis=0))
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Only show prediction if confidence is above threshold
            if confidence/100 >= confidence_threshold:
                st.success(f"Predicted Posture: {predicted_class}")
                st.info(f"Confidence: {confidence:.2f}%")
                
                # Show probability distribution
                st.subheader("Class Probabilities")
                prob_dict = dict(zip(CLASS_NAMES, prediction[0]))
                st.bar_chart(prob_dict)
            else:
                st.warning("Prediction confidence below threshold")
        else:
            st.error("No pose detected in the image. Please try another image.")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        
# Footer
st.markdown("---")
st.markdown("Built with Streamlit, MediaPipe, and TensorFlow")