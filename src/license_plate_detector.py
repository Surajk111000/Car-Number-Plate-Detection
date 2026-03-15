"""
Specialized License Plate Detection using EasyOCR
Uses text detection to find license plate regions
This is better for license plate detection than general object detection
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class LicensePlateTextDetector:
    """
    Detect license plates by finding text regions with EasyOCR
    More reliable than general object detection for plates
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize text-based plate detector
        
        Args:
            confidence_threshold: Minimum confidence for text detection
        """
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR not installed. Install with: pip install easyocr")
        
        self.confidence_threshold = confidence_threshold
        self.reader = None
        self._load_reader()
    
    def _load_reader(self) -> None:
        """Load EasyOCR reader for text detection"""
        logger.info("Initializing EasyOCR for text detection...")
        try:
            # Create reader for text detection (not recognition, just bounding boxes)
            self.reader = easyocr.Reader(['en'], gpu=False)
            logger.info("✅ EasyOCR text detector loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect license plate regions using text detection
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with detections in standard format
        """
        if self.reader is None:
            logger.error("Reader not initialized")
            return self._empty_detections()
        
        try:
            # Run text detection
            results = self.reader.readtext(image, detail=1)  # detail=1 to get confidence
            
            if not results:
                logger.debug("No text regions found")
                return self._empty_detections()
            
            # Parse results
            detection_boxes = []
            detection_scores = []
            detection_classes = []
            
            height, width = image.shape[:2]
            
            for (bbox, text, confidence) in results:
                # Confidence is already 0-1, but text confidence might be low for plates
                # Use (confidence + 0.3) boost to account for license plate specifics
                boosted_conf = min(1.0, confidence + 0.3)
                
                if boosted_conf < self.confidence_threshold:
                    continue
                
                # bbox is list of 4 corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                try:
                    points = np.array(bbox)
                    y_min = np.min(points[:, 1]) / height
                    y_max = np.max(points[:, 1]) / height
                    x_min = np.min(points[:, 0]) / width
                    x_max = np.max(points[:, 0]) / width
                    
                    # Validate
                    y_min = max(0, min(1, y_min))
                    y_max = max(0, min(1, y_max))
                    x_min = max(0, min(1, x_min))
                    x_max = max(0, min(1, x_max))
                    
                    detection_boxes.append([y_min, x_min, y_max, x_max])
                    detection_scores.append(boosted_conf)
                    detection_classes.append(0)  # License plate class
                    
                    logger.debug(f"Found text region: '{text}' confidence: {confidence:.2f} (boosted: {boosted_conf:.2f})")
                
                except Exception as e:
                    logger.warning(f"Error processing detection box: {e}")
                    continue
            
            num_detections = len(detection_boxes)
            logger.info(f"Text detection found {num_detections} potential plates")
            
            return {
                'detection_boxes': np.array(detection_boxes, dtype=np.float32),
                'detection_scores': np.array(detection_scores, dtype=np.float32),
                'detection_classes': np.array(detection_classes, dtype=np.int64),
                'num_detections': num_detections,
                'is_pretrained': True,
                'model': 'EasyOCR-TextDetection'
            }
        
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return self._empty_detections()
    
    def get_plate_regions(self, image: np.ndarray, detections: Dict) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract plate regions from the image using detection boxes
        
        Args:
            image: Input image
            detections: Detection results from detect()
            
        Returns:
            List of tuples (cropped_plate, (x_min, y_min, x_max, y_max) in pixels)
        """
        MIN_PLATE_WIDTH = 30
        MIN_PLATE_HEIGHT = 15
        padding = 10
        
        if detections['num_detections'] == 0:
            return []
        
        height, width = image.shape[:2]
        plates = []
        
        for box in detections['detection_boxes']:
            ymin, xmin, ymax, xmax = box
            
            # Convert to pixel coordinates
            x_min = int(xmin * width)
            x_max = int(xmax * width)
            y_min = int(ymin * height)
            y_max = int(ymax * height)
            
            # Validate box dimensions
            plate_width = x_max - x_min
            plate_height = y_max - y_min
            
            if plate_width < MIN_PLATE_WIDTH or plate_height < MIN_PLATE_HEIGHT:
                logger.debug(f"Skipping detection: size {plate_width}x{plate_height}")
                continue
            
            # Add padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)
            
            # Extract plate region
            plate_image = image[y_min:y_max, x_min:x_max]
            
            if plate_image.size > 0:
                plates.append((plate_image, (x_min, y_min, x_max, y_max)))
        
        return plates
    
    def visualize(self, image: np.ndarray, detections: Dict) -> np.ndarray:
        """
        Draw detections on image
        
        Args:
            image: Input image
            detections: Detection results
            
        Returns:
            Image with drawn boxes
        """
        result = image.copy()
        height, width = image.shape[:2]
        
        for box, score in zip(detections['detection_boxes'], detections['detection_scores']):
            ymin, xmin, ymax, xmax = box
            
            x_min = int(xmin * width)
            x_max = int(xmax * width)
            y_min = int(ymin * height)
            y_max = int(ymax * height)
            
            # Draw rectangle
            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw label
            label = f"Text {score:.2f}"
            cv2.putText(result, label, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result
    
    @staticmethod
    def _empty_detections() -> Dict[str, Any]:
        """Return empty detection"""
        return {
            'detection_boxes': np.empty((0, 4), dtype=np.float32),
            'detection_scores': np.empty((0,), dtype=np.float32),
            'detection_classes': np.empty((0,), dtype=np.int64),
            'num_detections': 0,
            'is_pretrained': True,
            'model': 'EasyOCR-TextDetection'
        }
