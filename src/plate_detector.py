"""
Plate Detection Module
Handles real-time number plate detection using TensorFlow object detection models
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional, TYPE_CHECKING
import logging
import json
from pathlib import Path

if TYPE_CHECKING:
    import tensorflow as tf

logger = logging.getLogger(__name__)

# Configuration constants
MIN_PLATE_WIDTH = 30  # Minimum plate width in pixels
MIN_PLATE_HEIGHT = 15  # Minimum plate height in pixels
DEFAULT_PADDING = 10  # Padding around detected plates


class PlateDetector:
    """Detects car number plates in images using pre-trained TensorFlow model"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, 
                 nms_threshold: float = 0.5, padding: int = DEFAULT_PADDING):
        """
        Initialize the plate detector
        
        Args:
            model_path: Path to the pre-trained model checkpoint
            confidence_threshold: Minimum confidence score for detections (0-1)
            nms_threshold: Non-maximum suppression threshold for overlapping boxes
            padding: Padding around detected plates in pixels
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model_path = model_path
        self.padding = padding
        self.detect_fn = None
        self.detection_count = 0
        self.is_mock_mode = False
        self._load_model()
    
    def _is_mock_model(self) -> bool:
        """Check if this is a mock model directory"""
        try:
            config_path = Path(self.model_path) / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    return config.get("type") == "mock"
        except:
            pass
        return False
    
    def _load_model(self):
        """Load the pre-trained detection model"""
        try:
            # Check if it's a mock model
            if self._is_mock_model():
                logger.info(f"Loading MOCK model from {self.model_path}...")
                self.is_mock_mode = True
                logger.info("Mock model loaded (will return dummy detections for testing)")
                return
            
            # For real models, lazy-import TensorFlow
            logger.info(f"Loading model from {self.model_path}...")
            import tensorflow as tf
            self.detect_fn = tf.saved_model.load(self.model_path)
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_mock_detections(self, image: np.ndarray) -> Dict:
        """Return mock detections for testing without a real model"""
        height, width = image.shape[:2]
        
        # Return 2 dummy plate detections in normalized coordinates [ymin, xmin, ymax, xmax]
        detection_boxes = np.array([
            [0.2, 0.15, 0.55, 0.9],   # Top plate
            [0.35, 0.1, 0.65, 0.85],  # Bottom plate
        ], dtype=np.float32)
        
        detection_scores = np.array([0.95, 0.88], dtype=np.float32)
        detection_classes = np.array([0, 0], dtype=np.int64)
        
        return {
            'detection_boxes': detection_boxes,
            'detection_scores': detection_scores,
            'detection_classes': detection_classes,
            'num_detections': 2
        }
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect number plates in an image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Dictionary containing detections with boxes, scores, and classes
            
        Raises:
            ValueError: If image is invalid or empty
        """
        # Validate input
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detect()")
            return self._empty_detections()
        
        if len(image.shape) not in [2, 3]:
            logger.error(f"Invalid image shape: {image.shape}")
            raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")
        
        # Return mock detections if in mock mode
        if self.is_mock_mode:
            logger.debug("Using mock detections for testing")
            detections = self._get_mock_detections(image)
            detections = self._filter_detections(detections)
            detections = self._apply_nms(detections, image.shape)
            self.detection_count = detections['num_detections']
            return detections
        
        # Convert image to RGB for TensorFlow
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Lazy import TensorFlow for real model inference
        import tensorflow as tf
        
        # Convert to tensor and add batch dimension
        input_tensor = tf.convert_to_tensor(image_rgb)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        
        try:
            if self.detect_fn is None:
                raise RuntimeError("Detection model is not loaded")

            # Run detection
            detections = self.detect_fn(input_tensor)
            
            # Extract detections
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
            
            # Filter by confidence threshold
            detections = self._filter_detections(detections)
            
            # Apply Non-Maximum Suppression
            detections = self._apply_nms(detections, image.shape)
            
            self.detection_count = detections['num_detections']
            logger.debug(f"Detected {self.detection_count} plates with confidence > {self.confidence_threshold}")
            
            return detections
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return self._empty_detections()
    
    @staticmethod
    def _empty_detections() -> Dict:
        """Return empty detection dictionary"""
        return {
            'detection_boxes': np.empty((0, 4), dtype=np.float32),
            'detection_scores': np.empty((0,), dtype=np.float32),
            'detection_classes': np.empty((0,), dtype=np.float32),
            'num_detections': 0
        }
    
    def _filter_detections(self, detections: Dict) -> Dict:
        """
        Filter detections by confidence threshold
        
        Args:
            detections: Raw detections dictionary
            
        Returns:
            Filtered detections dictionary
        """
        scores = detections['detection_scores']
        valid_idx = scores > self.confidence_threshold
        
        filtered = {
            'detection_boxes': detections['detection_boxes'][valid_idx],
            'detection_scores': scores[valid_idx],
            'detection_classes': detections['detection_classes'][valid_idx],
            'num_detections': int(np.sum(valid_idx))
        }
        
        logger.debug(f"Filtered {len(scores)} detections, kept {filtered['num_detections']}")
        return filtered
    
    def _apply_nms(self, detections: Dict, image_shape: Tuple[int, ...]) -> Dict:
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            detections: Filtered detections
            image_shape: Shape of input image (height, width, ...)
            
        Returns:
            Detections after NMS
        """
        if detections['num_detections'] == 0:
            return detections
        
        # Convert normalized coordinates to pixel coordinates
        height, width = image_shape[:2]
        boxes = detections['detection_boxes'].copy()
        
        # Convert from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax) in pixels
        boxes_pixel = np.zeros_like(boxes)
        boxes_pixel[:, 0] = boxes[:, 1] * width  # xmin
        boxes_pixel[:, 1] = boxes[:, 0] * height  # ymin
        boxes_pixel[:, 2] = boxes[:, 3] * width  # xmax
        boxes_pixel[:, 3] = boxes[:, 2] * height  # ymax
        
        # Apply NMS
        scores = detections['detection_scores'].astype(np.float32)
        indices = self._nms(boxes_pixel, scores, self.nms_threshold)
        
        # Filter detections using NMS indices
        filtered = {
            'detection_boxes': detections['detection_boxes'][indices],
            'detection_scores': detections['detection_scores'][indices],
            'detection_classes': detections['detection_classes'][indices],
            'num_detections': len(indices)
        }
        
        logger.debug(f"NMS reduced detections from {detections['num_detections']} to {filtered['num_detections']}")
        return filtered
    
    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Non-Maximum Suppression algorithm
        
        Args:
            boxes: Bounding boxes in format (xmin, ymin, xmax, ymax)
            scores: Confidence scores for each box
            threshold: IOU threshold for suppression
            
        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return np.array([], dtype=int)
        
        # Calculate areas
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by score
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # Calculate IOU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            union = areas[i] + areas[order[1:]] - inter
            union = np.maximum(union, 1e-6)
            iou = inter / union
            
            # Keep boxes with IOU below threshold
            order = order[np.where(iou <= threshold)[0] + 1]
        
        return np.array(keep, dtype=int)
    
    def get_plate_regions(self, image: np.ndarray, detections: Dict) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract plate regions from the image using detection boxes
        
        Args:
            image: Input image
            detections: Detection results from detect()
            
        Returns:
            List of tuples (cropped_plate, (x_min, y_min, x_max, y_max) in pixels)
        """
        if detections['num_detections'] == 0:
            logger.info("No detections found")
            return []
        
        height, width = image.shape[:2]
        plates = []
        
        for box, score in zip(detections['detection_boxes'], detections['detection_scores']):
            # Normalize box coordinates (they are in [0, 1] range)
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
                logger.debug(f"Skipping detection: size {plate_width}x{plate_height} below minimum")
                continue
            
            # Add padding
            x_min = max(0, x_min - self.padding)
            y_min = max(0, y_min - self.padding)
            x_max = min(width, x_max + self.padding)
            y_max = min(height, y_max + self.padding)
            
            # Extract plate region
            plate = image[y_min:y_max, x_min:x_max]
            
            if plate.size > 0:
                plates.append((plate, (x_min, y_min, x_max, y_max)))
                logger.debug(f"Extracted plate region: {plate.shape}, confidence: {score:.2f}")
        
        logger.info(f"Extracted {len(plates)} valid plate regions")
        return plates
    
    def visualize(self, image: np.ndarray, detections: Dict, 
                  thickness: int = 2, font_scale: float = 0.6) -> np.ndarray:
        """
        Draw bounding boxes on the image with confidence scores
        
        Args:
            image: Input image
            detections: Detection results
            thickness: Line thickness for bounding boxes
            font_scale: Font scale for text
            
        Returns:
            Image with drawn bounding boxes
        """
        output_image = image.copy()
        
        if detections['num_detections'] == 0:
            logger.info("No detections to visualize")
            return output_image
        
        height, width = image.shape[:2]
        
        for idx, (box, score) in enumerate(zip(detections['detection_boxes'], 
                                               detections['detection_scores'])):
            ymin, xmin, ymax, xmax = box
            
            # Convert to pixel coordinates
            x_min = int(xmin * width)
            x_max = int(xmax * width)
            y_min = int(ymin * height)
            y_max = int(ymax * height)
            
            # Draw rectangle
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness)
            
            # Create label with index and confidence
            label = f"Plate {idx+1}: {score:.3f}"
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
            
            # Draw background rectangle for text
            cv2.rectangle(output_image, 
                         (x_min, y_min - text_size[1] - 10),
                         (x_min + text_size[0] + 10, y_min),
                         (0, 255, 0), -1)
            
            # Put confidence score
            cv2.putText(output_image, label, (x_min + 5, y_min - 5),
                       font, font_scale, (0, 0, 0), 1)
        
        # Add detection count at top
        count_text = f"Detections: {detections['num_detections']}"
        cv2.putText(output_image, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return output_image
    
    def get_detection_stats(self) -> Dict:
        """
        Get statistics about the last detection
        
        Returns:
            Dictionary with detection statistics
        """
        return {
            'total_detections': self.detection_count,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold
        }
