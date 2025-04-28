import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

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
            frame: Input image/frame
            
        Returns:
            processed_frame: Image with annotations
            final_results: List of detection results
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

    def process_image(self, image_path, output_dir=None):
        """
        Process a single image file.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save the processed image (optional)
            
        Returns:
            results: List of detection results
        """
        # Read the image
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return None
            
        # Process the image
        processed_frame, results = self.process_frame(frame)
        
        # Display results
        cv2.imshow('Apron Detection', processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the processed image if output directory is provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"processed_{Path(image_path).name}"
            cv2.imwrite(str(output_path), processed_frame)
            print(f"Processed image saved to: {output_path}")
        
        # Print results to console
        print("\nDetection Results:")
        print(f"Image: {image_path}")
        for result in results:
            print(f"Person {result['person_id']}: {result['apron_status']} "
                  f"(Apron conf: {result['apron_confidence']:.2f}, "
                  f"Person conf: {result['person_confidence']:.2f})")
        
        return results

    def process_directory(self, input_dir, output_dir=None):
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images (optional)
            
        Returns:
            all_results: Dictionary of results for all images
        """
        input_dir = Path(input_dir)
        image_paths = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
        
        all_results = {}
        for image_path in image_paths:
            results = self.process_image(image_path, output_dir)
            all_results[image_path.name] = results
            
        return all_results

if __name__ == "__main__":
    # Initialize the pipeline with your trained models
    pipeline = ApronDetectionPipeline(
        person_model_path=r"D:\worksapce\Apron_Pipeline\Person_detector.pt",
        apron_model_path=r"D:\worksapce\Apron_Pipeline\Apron_detector_Model.pt"
    )
    
    # Example usage:
    
    # 1. Process a single image
    image_path = r"D:\worksapce\Apron_Pipeline\test_image.jpg"
    results = pipeline.process_image(image_path, output_dir="output_images")
    
    # 2. Process all images in a directory
    # input_dir = "path/to/your/images"
    # all_results = pipeline.process_directory(input_dir, output_dir="output_images")