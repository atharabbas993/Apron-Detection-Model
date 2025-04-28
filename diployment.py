import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

# Initialize the models (cached to load only once)
@st.cache_resource
def load_models(person_model_path, apron_model_path):
    return YOLO(person_model_path), YOLO(apron_model_path)

class ApronDetectionPipeline:
    def __init__(self, person_model, apron_model):
        self.person_model = person_model
        self.apron_model = apron_model
        self.person_conf_threshold = 0.5  # 50% confidence threshold for person detection
        self.apron_conf_threshold = 0.5   # 50% confidence threshold for apron classification
    
    def process_frame(self, frame):
        """Process a frame and return annotated image and results"""
        # Convert color if needed (Streamlit uses RGB, OpenCV uses BGR)
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Person detection with confidence threshold
        person_results = self.person_model(frame)
        person_boxes = []
        
        for result in person_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if cls_id == 0 and conf >= self.person_conf_threshold:  # Person class with confidence threshold
                    person_boxes.append({
                        'box': box,
                        'confidence': conf
                    })
        
        final_results = []
        processed_frame = frame.copy()
        
        # Apron classification for each person with confidence threshold
        for i, person in enumerate(person_boxes):
            x1, y1, x2, y2 = map(int, person['box'])
            person_img = frame[y1:y2, x1:x2]
            
            if person_img.size == 0:
                continue
                
            apron_results = self.apron_model(person_img)
            apron_cls = None
            apron_conf = 0
            
            for result in apron_results:
                if len(result.boxes) > 0 and result.boxes.conf[0] >= self.apron_conf_threshold:
                    apron_cls = int(result.boxes.cls[0].item())
                    apron_conf = result.boxes.conf[0].item()
            
            if apron_cls is None:  # Skip if no detection meets confidence threshold
                continue
                
            apron_status = "apron" if apron_cls == 0 else "no_apron"
            
            final_results.append({
                'person_id': i+1,
                'box': [x1, y1, x2, y2],
                'apron_status': apron_status,
                'apron_confidence': apron_conf,
                'person_confidence': person['confidence']
            })
            
            # Draw bounding box and label
            color = (0, 255, 0) if apron_status == "apron" else (0, 0, 255)
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            label = f"Person {i+1}: {apron_status} ({apron_conf:.2f})"
            cv2.putText(processed_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert back to RGB for Streamlit
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        return processed_frame, final_results

def main():
    st.title("Apron Detection App")
    st.write("Detect whether people are wearing aprons using YOLOv11 models (50% confidence threshold)")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    person_model_path = st.sidebar.text_input(
        "Person Model Path", 
        value=r"D:\worksapce\Apron_Pipeline\Person_detector.pt",
        help="Path to YOLOv8 person detection model"
    )
    apron_model_path = st.sidebar.text_input(
        "Apron Model Path", 
        value=r"D:\worksapce\Apron_Pipeline\Apron_detector_Model.pt",
        help="Path to YOLOv8 apron classification model"
    )
    
    # Load models
    try:
        person_model, apron_model = load_models(person_model_path, apron_model_path)
        pipeline = ApronDetectionPipeline(person_model, apron_model)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()
    
    # Input selection
    input_option = st.radio(
        "Select Input Type:",
        ("Webcam", "Image Upload"),
        horizontal=True
    )
    
    if input_option == "Webcam":
        st.header("Webcam Apron Detection")
        run_webcam = st.checkbox("Start Webcam")
        
        if run_webcam:
            st.warning("Webcam feature might not work on all servers. For remote servers, consider image upload.")
            FRAME_WINDOW = st.image([])
            cap = cv2.VideoCapture(0)
            
            stop_button = st.button("Stop Webcam")
            
            while run_webcam and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from webcam")
                    break
                
                # Process frame
                processed_frame, results = pipeline.process_frame(frame)
                FRAME_WINDOW.image(processed_frame)
                
                # Display results
                with st.expander("Detection Results"):
                    if not results:
                        st.warning("No persons detected with confidence ≥50% or no aprons classified with confidence ≥50%")
                    for result in results:
                        st.write(f"Person {result['person_id']}: {result['apron_status']} "
                                f"(Apron conf: {result['apron_confidence']:.2f}, "
                                f"Person conf: {result['person_confidence']:.2f})")
            
            cap.release()
            if stop_button:
                st.success("Webcam stopped")
    
    else:  # Image Upload
        st.header("Image Apron Detection")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Read the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to numpy array
            frame = np.array(image)
            
            # Process the image
            with st.spinner("Processing image..."):
                processed_frame, results = pipeline.process_frame(frame)
            
            st.image(processed_frame, caption="Processed Image", use_column_width=True)
            
            # Display results
            st.subheader("Detection Results")
            if not results:
                st.warning("No persons detected with confidence ≥50% or no aprons classified with confidence ≥50%")
            else:
                for result in results:
                    st.success(f"Person {result['person_id']}: {result['apron_status']} "
                             f"(Apron confidence: {result['apron_confidence']:.2f}, "
                             f"Person confidence: {result['person_confidence']:.2f})")
                    
                    # Show cropped person
                    x1, y1, x2, y2 = result['box']
                    person_crop = frame[y1:y2, x1:x2]
                    st.image(person_crop, caption=f"Person {result['person_id']} Crop")
            
            # Create download button
            processed_image = Image.fromarray(processed_frame)
            img_byte_arr = io.BytesIO()
            processed_image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            st.download_button(
                label="Download Processed Image",
                data=img_byte_arr,
                file_name=f"processed_{uploaded_file.name}",
                mime="image/jpeg"
            )

if __name__ == "__main__":
    main()