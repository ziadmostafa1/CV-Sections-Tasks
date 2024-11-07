import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image

# Initialize session state
if 'stop_button' not in st.session_state:
    st.session_state['stop_button'] = False
if 'camera_running' not in st.session_state:
    st.session_state['camera_running'] = True

# Cache model loading
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.h5")

model = load_model()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Define class names
sign_language_classes = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "ExcuseMe",
    "F", "Food", "G", "H", "Hello", "Help", "House", "I", "I Love You", "J", "K", "L",
    "M", "N", "No", "O", "P", "Please", "Q", "R", "S", "T", "ThankYou", "U", "V", "W",
    "X", "Y", "Yes", "Z"
]

def process_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def pad_landmarks():
    return [0.0] * 63

def classify_gesture(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    
    if result.multi_hand_landmarks:
        combined_landmarks = []
        
        # Process first hand
        combined_landmarks.extend(process_landmarks(result.multi_hand_landmarks[0]))
        
        # Process second hand or pad
        if len(result.multi_hand_landmarks) > 1:
            combined_landmarks.extend(process_landmarks(result.multi_hand_landmarks[1]))
        else:
            combined_landmarks.extend(pad_landmarks())
            
        # Make prediction
        landmarks_array = np.array(combined_landmarks).reshape(1, -1)
        prediction = model.predict(landmarks_array, verbose=0)
        class_id = np.argmax(prediction[0])
        confidence = prediction[0][class_id]
        
        return sign_language_classes[class_id], result.multi_hand_landmarks, confidence
    
    return None, None, None

def process_uploaded_image(image_bytes):
    # Convert uploaded image to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

def main():
    st.title("Hand Signs Recognition")
    
    # Input source selection
    input_source = st.radio("Select Input Source:", ["Webcam", "Upload Image"])
    
    if input_source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Process the uploaded image
            image_bytes = uploaded_file.read()
            frame = process_uploaded_image(image_bytes)
            
            # Make prediction
            gesture, hand_landmarks, confidence = classify_gesture(frame)
            
            # Draw landmarks and display results
            if hand_landmarks:
                for landmarks in hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            if gesture:
                cv2.putText(frame, f"Prediction: {gesture}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                st.write(f"Detected Sign: {gesture}")
                if confidence:
                    st.write(f"Confidence: {confidence:.2%}")
            else:
                st.write("No sign detected")
            
            # Display the processed image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Processed Image", use_column_width=True)
    
    else:  # Webcam
        video_placeholder = st.empty()
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        stop_button = st.button("Stop")
        
        cap = cv2.VideoCapture(0)
        
        try:
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam")
                    break

                gesture, hand_landmarks, confidence = classify_gesture(frame)

                if hand_landmarks:
                    for landmarks in hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, landmarks, mp_hands.HAND_CONNECTIONS)

                if gesture:
                    cv2.putText(frame, f"Prediction: {gesture}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    prediction_placeholder.text(f"Detected Sign: {gesture}")
                    if confidence:
                        confidence_placeholder.progress(float(confidence), f"Confidence: {confidence:.2%}")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        finally:
            cap.release()
            st.session_state['camera_running'] = False
            
# Add a new section to take and predict an image
    st.title("Predict Sign from Image")
    take_photo_button = st.button("Take Photo")
    
    if take_photo_button:
        # Take a photo using the webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        # Make prediction
        gesture, hand_landmarks, confidence = classify_gesture(frame)
        
        # Draw landmarks and display results
        if hand_landmarks:
            for landmarks in hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, landmarks, mp_hands.HAND_CONNECTIONS)
        
        if gesture:
            cv2.putText(frame, f"Prediction: {gesture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            st.write(f"Detected Sign: {gesture}")
            if confidence:
                st.write(f"Confidence: {confidence:.2%}")
        else:
            st.write("No sign detected")
        
        # Display the processed image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()