"""
OCR Module
Handles optical character recognition for number plate text extraction
"""

import easyocr
import numpy as np
import cv2
import re
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Fix for PIL/Pillow compatibility with EasyOCR
# EasyOCR uses deprecated PIL.Image.ANTIALIAS which was removed in Pillow 10+
try:
    from PIL import Image
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
        logger.info("Applied PIL.Image.ANTIALIAS compatibility fix")
except Exception as e:
    logger.warning(f"Failed to apply PIL compatibility fix: {e}")

# Configuration constants
PLATE_PATTERN = r'^[A-Z0-9\-\s]+$'  # Pattern for valid plate text
MIN_CONFIDENCE = 0.3
DEFAULT_MORPH_KERNEL = (5, 5)


class PlateOCR:
    """Performs OCR on license plate images to extract text"""
    
    def __init__(self, languages: List[str] = None, use_gpu: bool = None):
        """
        Initialize OCR reader
        
        Args:
            languages: List of language codes for OCR (default: ['en'])
            use_gpu: Use GPU for OCR. None = auto-detect, True = force GPU, False = force CPU
        """
        if languages is None:
            languages = ['en']
        
        self.languages = languages
        self.reader = None
        self.use_gpu = use_gpu if use_gpu is not None else self._check_gpu_availability()
        self._load_reader()
    
    @staticmethod
    def _check_gpu_availability() -> bool:
        """
        Check if GPU is available for OCR
        
        Returns:
            True if GPU is available, False otherwise
        """
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("No GPU detected, using CPU")
            return gpu_available
        except Exception as e:
            logger.warning(f"Could not check GPU availability: {e}, using CPU")
            return False
    
    def _load_reader(self):
        """Initialize EasyOCR reader"""
        try:
            logger.info(f"Initializing OCR reader for languages: {self.languages}, GPU={self.use_gpu}...")
            self.reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
            logger.info("OCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR reader: {e}")
            raise
    
    def extract_text(self, image: np.ndarray, 
                    confidence_threshold: float = MIN_CONFIDENCE,
                    clean_text: bool = True) -> Dict:
        """
        Extract text from a number plate image
        
        Args:
            image: Input plate image
            confidence_threshold: Minimum confidence for text recognition
            clean_text: Whether to clean and validate extracted text
            
        Returns:
            Dictionary with extracted text, confidence scores, and bounding boxes
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                logger.warning("Empty image provided to extract_text()")
                return self._empty_result()
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run OCR
            result = self.reader.readtext(processed_image)
            
            # Filter by confidence and format results
            extracted_text = {
                'text': [],
                'confidence': [],
                'boxes': [],
                'full_text': '',
                'cleaned_text': '',
                'avg_confidence': 0.0,
                'is_valid_plate': False
            }
            
            if not result:
                logger.info("No text detected in image")
                return extracted_text
            
            total_confidence = 0.0
            for detection in result:
                if len(detection) >= 3 and detection[2] >= confidence_threshold:
                    text = detection[1]
                    confidence = detection[2]
                    
                    extracted_text['text'].append(text)
                    extracted_text['confidence'].append(confidence)
                    extracted_text['boxes'].append(detection[0])
                    total_confidence += confidence
            
            if extracted_text['text']:
                # Combine all text
                extracted_text['full_text'] = ''.join(extracted_text['text'])
                extracted_text['avg_confidence'] = total_confidence / len(extracted_text['text'])
                
                # Clean text if requested
                if clean_text:
                    extracted_text['cleaned_text'] = self._clean_text(extracted_text['full_text'])
                    extracted_text['is_valid_plate'] = self._validate_plate_text(extracted_text['cleaned_text'])
                
                logger.info(f"Extracted text: {extracted_text['full_text']}, avg confidence: {extracted_text['avg_confidence']:.3f}")
            
            return extracted_text
        
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            result = self._empty_result()
            result['error'] = str(e)
            return result
    
    @staticmethod
    def _empty_result() -> Dict:
        """Return empty result dictionary"""
        return {
            'text': [],
            'confidence': [],
            'boxes': [],
            'full_text': '',
            'cleaned_text': '',
            'avg_confidence': 0.0,
            'is_valid_plate': False
        }
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean and standardize OCR output
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        # Convert to uppercase
        text = text.upper()
        
        # Remove special characters except common plate separators
        text = re.sub(r'[^A-Z0-9\-\s]', '', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def _validate_plate_text(text: str) -> bool:
        """
        Validate if text looks like a license plate
        
        Args:
            text: Cleaned text to validate
            
        Returns:
            True if text matches license plate pattern
        """
        if not text or len(text) < 3:
            return False
        
        # Check if text contains alphanumeric characters
        if not re.match(PLATE_PATTERN, text):
            return False
        
        # Check if it has at least some numbers and letters
        has_alpha = any(c.isalpha() for c in text)
        has_digit = any(c.isdigit() for c in text)
        
        return has_alpha and has_digit
    
    def _preprocess_image(self, image: np.ndarray, use_morphology: bool = True) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: Input image
            use_morphology: Whether to apply morphological operations
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Apply morphological operations if requested
        if use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DEFAULT_MORPH_KERNEL)
            # Closing to fill small holes
            morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=1)
            # Opening to remove noise
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
            denoised = morph
        
        # Apply adaptive threshold for better text separation
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Optional: Resize if image is too small
        height, width = thresh.shape
        if width < 100 or height < 30:
            scale = max(100 / width, 30 / height)
            thresh = cv2.resize(thresh, None, fx=scale, fy=scale, 
                               interpolation=cv2.INTER_CUBIC)
            logger.debug(f"Resized image by factor {scale:.2f}")
        
        return thresh
    
    def extract_with_visualization(self, image: np.ndarray, 
                                   show_confidence: bool = True) -> Tuple[Dict, np.ndarray]:
        """
        Extract text and return visualization with bounding boxes
        
        Args:
            image: Input plate image
            show_confidence: Whether to show confidence scores on visualization
            
        Returns:
            Tuple of (extraction results, visualization image)
        """
        result = self.extract_text(image)
        viz_image = image.copy()
        
        if not result['boxes']:
            logger.info("No text boxes to visualize")
            return result, viz_image
        
        # Sort boxes by x-coordinate for left-to-right reading order
        sorted_indices = sorted(range(len(result['boxes'])),
                               key=lambda i: np.mean(result['boxes'][i][:, 0]))
        
        # Draw bounding boxes on visualization
        for idx in sorted_indices:
            box = np.array(result['boxes'][idx]).astype(int)
            
            # Draw box
            cv2.polylines(viz_image, [box], True, (0, 255, 0), 2)
            
            # Draw text and confidence if requested
            if show_confidence:
                text_label = f"{result['text'][idx]} ({result['confidence'][idx]:.2f})"
                top_left = tuple(box[0])
                cv2.putText(viz_image, text_label, top_left,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Add full extracted text
        if result['full_text']:
            text_display = f"Detected: {result['cleaned_text'] or result['full_text']}"
            cv2.putText(viz_image, text_display, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add validity indicator
            if result['is_valid_plate']:
                validity_text = "✓ Valid Plate"
                color = (0, 255, 0)
            else:
                validity_text = "⚠ Needs Review"
                color = (0, 165, 255)
            
            cv2.putText(viz_image, validity_text, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result, viz_image
    
    def get_ocr_stats(self) -> Dict:
        """
        Get OCR reader statistics
        
        Returns:
            Dictionary with OCR statistics
        """
        return {
            'languages': self.languages,
            'using_gpu': self.use_gpu,
            'reader_available': self.reader is not None
        }
