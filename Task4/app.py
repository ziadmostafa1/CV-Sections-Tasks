import tempfile
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Add this near the top of app.py after imports
model_options = {
    "50 Epochs": "fire-yolo-seg-50epoch.pt",
    "100 Epochs": "fire-yolo-seg-100epoch.pt"
}

# Add this before the input type selection
model_choice = st.selectbox("Select Model Version", list(model_options.keys()))
model = YOLO(model_options[model_choice])
class_names = model.names

def process_frame(im0):
    annotator = Annotator(im0, line_width=2)
    results = model.track(im0, iou=0.5, show=False, persist=True, tracker="bytetrack.yaml")

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for mask, track_id, class_id in zip(masks, track_ids, class_ids):
            class_name = class_names[class_id]
            # Get color based on class_id
            color = colors(class_id, True)  # RGB color
            color = (int(color[2]), int(color[1]), int(color[0]))  # Convert to BGR
            
            # Draw polygon with class-specific color
            cv2.polylines(im0, [np.int32([mask])], isClosed=True, color=color, thickness=2)
            
            # Add text with smaller size and higher resolution
            text = f'{track_id} {class_name}'
            font_scale = 0.6  # Reduced from 1.2
            thickness = 2    # Reduced from 3
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Improve text position and add background for better visibility
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = int(mask[0][0])
            text_y = int(mask[0][1] - 5)
            
            # Draw semi-transparent background for text
            cv2.rectangle(im0, 
                        (text_x - 2, text_y - text_size[1] - 4),
                        (text_x + text_size[0] + 2, text_y + 2),
                        color, -1)
            
            # Draw text in white for better contrast
            cv2.putText(im0, text, (text_x, text_y), 
                       font, font_scale, (255,255,255), thickness)
    
    return im0

st.title("YOLO Segmentation and Tracking")

option = st.selectbox("Choose input type", ("Image", "Video"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        processed_image = process_frame(image)
        st.image(processed_image, channels="BGR", width=800)  # Added width parameter

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame)
            stframe.image(processed_frame, channels="BGR", width=800)  # Added width parameter
        cap.release()


st.write("YOLO Segmentation and Tracking App")