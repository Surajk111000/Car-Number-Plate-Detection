#!/usr/bin/env python3
"""Debug script to test PaddleOCR detection directly"""

import os
import cv2
import numpy as np

# Disable problematic oneDNN features
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['FLAGS_use_mkldnn'] = 'False'

from paddleocr import PaddleOCR

print('Testing PaddleOCR Directly on Images...')
print('=' * 70)

# Initialize OCR
print('Initializing PaddleOCR...')
ocr = PaddleOCR(lang='en')
print('[OK] PaddleOCR ready\n')

# Test images
images = ['sample_plate.jpg', 'test_plate_ocr.jpg']

for img_path in images:
    print(f'Testing: {img_path}')
    print('-' * 70)
    
    try:
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f'  [ERROR] Failed to read image')
            continue
        
        h, w = image.shape[:2]
        print(f'  Image size: {w}x{h} pixels')
        
        # Run OCR
        result = ocr.ocr(image)
        print(f'  Raw result length: {len(result) if result else 0}')
        
        if result and result[0]:
            print(f'  [OK] Text regions detected: {len(result[0])}')
            for i, line in enumerate(result[0][:10]):  # Show all detections
                bbox, (text, conf) = line
                points = np.array(bbox)
                area = int((points.max(axis=0) - points.min(axis=0)).prod())
                print(f'    [{i}] Text: "{text:20s}" | Conf: {conf:.4f} | Area: {area}px')
        else:
            print(f'  [WARN] No text detected in this image')
            
    except Exception as e:
        import traceback
        print(f'  [ERROR] {e}')
        traceback.print_exc()

print('\n' + '=' * 70)
print('[OK] Direct OCR test complete')
