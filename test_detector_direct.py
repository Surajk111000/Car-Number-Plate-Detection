#!/usr/bin/env python3
"""Debug: Test detector directly"""

import cv2
from src.license_plate_detector import LicensePlateTextDetector

print('Testing Detector Directly...')
print('=' * 70)

# Create detector
detector = LicensePlateTextDetector(confidence_threshold=0.5)

images = ['sample_plate.jpg', 'test_plate_ocr.jpg']

for img_path in images:
    print(f'\nTesting: {img_path}')
    print('-' * 70)
    
    img = cv2.imread(img_path)
    if img is None:
        print('  [ERROR] Failed to read')
        continue
    
    h, w = img.shape[:2]
    print(f'  Image: {w}x{h}')
    
    result = detector.detect(img)
    
    num_dets = result.get('num_detections', 0)
    print(f'  Detections: {num_dets}')
    
    if num_dets > 0:
        print(f'  Scores: {result.get("detection_scores", [])}')
        print(f'  Boxes: {result.get("detection_boxes", [])}')
    else:
        print('  [WARN] 0 detections')

print('\n' + '=' * 70)
print('[OK] Direct detector test complete')
