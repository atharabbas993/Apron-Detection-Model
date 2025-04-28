import cv2
import numpy as np
from ultralytics import YOLO

class ApronDetectionPipeline:
    def __init__(self, person_model_path, apron_model_path):
        """
        Initialize the pipeline with YOLOv8 models.
        
        Args:
            person_model_path: Path to the trained YOLOv8 person detection model
            apron_model_path: Path to the trained YOLOv8 apron classification model
        """
        self.person_model = YOLO(person_model_path)
        self.apron_model = YOLO(apron_model_path)
        
    def process_frame(self, frame):
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input image/frame from camera
            
        Returns:
            Processed frame with annotations and list of results
        """
        # Step 1: Detect persons in the frame
        person_results = self.person_model(frame)
        
        # Extract bounding boxes for persons
        person_boxes = []
        for result in person_results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Filter for person class (assuming class 0 is person)
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if cls_id == 0:  # Person class
                    person_boxes.append({
                        'box': box,
                        'confidence': conf
                    })
        
        # Prepare final results
        final_results = []
        processed_frame = frame.copy()
        
        # Step 2: Crop each person and classify apron
        for i, person in enumerate(person_boxes):
            x1, y1, x2, y2 = map(int, person['box'])
            
            # Crop person from image
            person_img = frame[y1:y2, x1:x2]
            
            if person_img.size == 0:
                continue  # Skip empty crops
                
            # Step 3: Classify apron on the cropped person
            apron_results = self.apron_model(person_img)
            
            # Get apron classification (assuming single classification per person)
            apron_cls = None
            apron_conf = 0
            for result in apron_results:
                if len(result.boxes) > 0:
                    apron_cls = int(result.boxes.cls[0].item())
                    apron_conf = result.boxes.conf[0].item()
            
            # Determine apron status
            apron_status = "apron" if apron_cls == 0 else "no_apron"  # Adjust class IDs as per your model
            
            # Store result
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
        
        return processed_frame, final_results

    def run_on_camera(self, camera_index=0):
        """
        Run the pipeline on live camera feed.
        
        Args:
            camera_index: Index of the camera to use (default 0)
        """
        cap = cv2.VideoCapture(camera_index)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame, results = self.process_frame(frame)
            
            # Display results
            cv2.imshow('Apron Detection', processed_frame)
            
            # Print results to console
            print("\nCurrent Frame Results:")
            for result in results:
                print(f"Person {result['person_id']}: {result['apron_status']} "
                      f"(Apron conf: {result['apron_confidence']:.2f}, "
                      f"Person conf: {result['person_confidence']:.2f})")
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize the pipeline with your trained models
    pipeline = ApronDetectionPipeline(
        person_model_path=r"D:\worksapce\Apron_Pipeline\Person_detector.pt",
        apron_model_path=r"D:\worksapce\Apron_Pipeline\Apron_detector_Model.pt"
    )
    
    # Run on camera (change to 1 if you have external camera)
    pipeline.run_on_camera(camera_index=0)