"""
Pretrained YOLOv8 Model for Plate Detection
Uses YOLOv8 Nano - lightweight and fast
"""

import logging
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLOv8 not installed. Install with: pip install ultralytics")


class PretrainedPlateDetector:
    """
    Use YOLOv8 pretrained model for plate detection
    Much better performance than training from scratch on synthetic data
    """
    
    def __init__(
        self,
        model_size: str = "nano",  # nano, small, medium, large, xlarge
        confidence_threshold: float = 0.5,
        device: str = "cpu"  # cpu, 0 (GPU)
    ):
        """
        Initialize pretrained detector
        
        Args:
            model_size: YOLOv8 model size (nano is fastest)
            confidence_threshold: Detection confidence threshold
            device: 'cpu' or GPU device number
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8 not installed. Run: pip install ultralytics")
        
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLOv8 pretrained model"""
        logger.info(f"Loading YOLOv8-{self.model_size} pretrained model...")
        
        try:
            # Map full names to abbreviations
            size_map = {
                'nano': 'n',
                'small': 's',
                'medium': 'm',
                'large': 'l',
                'xlarge': 'x',
                'n': 'n',  # Already abbreviated
                's': 's',
                'm': 'm',
                'l': 'l',
                'x': 'x'
            }
            
            size_abbr = size_map.get(self.model_size, 'n')
            model_name = f"yolov8{size_abbr}.pt"
            
            logger.info(f"Loading model: {model_name}")
            self.model = YOLO(model_name)
            self.model.to(self.device)
            
            params = sum(p.numel() for p in self.model.model.parameters()) / 1e6
            logger.info(f"✅ YOLOv8-{self.model_size} loaded successfully")
            logger.info(f"   Parameters: {params:.1f}M")
            logger.info(f"   Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect objects in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with detections
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            # Extract detections
            detections = self._parse_yolo_results(results, image.shape)
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return self._empty_detections()
    
    def get_plate_regions(self, image: np.ndarray, detections: Dict) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract plate regions from the image using detection boxes
        Compatible with PlateDetector interface
        
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
            logger.debug("No detections found")
            return []
        
        height, width = image.shape[:2]
        plates = []
        
        for box in detections['detection_boxes']:
            # Box is already in normalized coordinates [ymin, xmin, ymax, xmax]
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
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)
            
            # Extract plate region
            plate_image = image[y_min:y_max, x_min:x_max]
            
            if plate_image.size > 0:
                plates.append((plate_image, (x_min, y_min, x_max, y_max)))
        
        logger.debug(f"Extracted {len(plates)} plate regions")
        return plates
    
    def _parse_yolo_results(self, results, image_shape: Tuple) -> Dict[str, Any]:
        """Parse YOLOv8 results to standard format"""
        detection_boxes = []
        detection_scores = []
        detection_classes = []
        
        height, width = image_shape[:2]
        
        for result in results:
            if result.boxes is None:
                continue
            
            # Get boxes in normalized coordinates
            boxes = result.boxes.xywhn.cpu().numpy()  # normalized center format
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                # Convert from center format (x_center, y_center, width, height) 
                # to corner format (ymin, xmin, ymax, xmax)
                x_center, y_center, w, h = box
                
                y_min = max(0, y_center - h/2)
                y_max = min(1, y_center + h/2)
                x_min = max(0, x_center - w/2)
                x_max = min(1, x_center + w/2)
                
                detection_boxes.append([y_min, x_min, y_max, x_max])
                detection_scores.append(float(score))
                detection_classes.append(int(cls))
        
        num_detections = len(detection_boxes)
        
        return {
            'detection_boxes': np.array(detection_boxes, dtype=np.float32),
            'detection_scores': np.array(detection_scores, dtype=np.float32),
            'detection_classes': np.array(detection_classes, dtype=np.int64),
            'num_detections': num_detections,
            'is_pretrained': True,
            'model': 'YOLOv8'
        }
    
    @staticmethod
    def _empty_detections() -> Dict[str, Any]:
        """Return empty detection"""
        return {
            'detection_boxes': np.empty((0, 4), dtype=np.float32),
            'detection_scores': np.empty((0,), dtype=np.float32),
            'detection_classes': np.empty((0,), dtype=np.int64),
            'num_detections': 0,
            'is_pretrained': True,
            'model': 'YOLOv8'
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            return {'status': 'not_loaded'}
        
        return {
            'model': 'YOLOv8',
            'size': self.model_size,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'parameters_m': f"{sum(p.numel() for p in self.model.model.parameters())/1e6:.1f}M",
            'pretrained': True,
            'status': 'ready'
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("YOLOv8 Pretrained Model - Quick Test")
    print("=" * 80)
    
    # Load model (will auto-download first time)
    print("\n1️⃣  Loading YOLOv8 Nano model...")
    detector = PretrainedPlateDetector(model_size="nano", confidence_threshold=0.5)
    
    # Check if sample image exists
    sample_image_path = "sample_plate.jpg"
    if Path(sample_image_path).exists():
        print(f"\n2️⃣  Loading test image: {sample_image_path}")
        image = cv2.imread(sample_image_path)
        
        print("\n3️⃣  Running detection...")
        detections = detector.detect(image)
        
        print(f"\n📊 Results:")
        print(f"   Detections found: {detections['num_detections']}")
        print(f"   Model: {detections['model']}")
        print(f"   Pretrained: {detections['is_pretrained']}")
        
        if detections['num_detections'] > 0:
            print(f"\n   Detection Details:")
            for i, (box, score, cls) in enumerate(zip(
                detections['detection_boxes'],
                detections['detection_scores'],
                detections['detection_classes']
            )):
                print(f"   [{i+1}] Box: {box}, Score: {score:.2%}, Class: {cls}")
    else:
        print(f"⚠️  Sample image not found: {sample_image_path}")
    
    print(f"\n✅ Model Info:")
    print(f"   {detector.get_info()}")
