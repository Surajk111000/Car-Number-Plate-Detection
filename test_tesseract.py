#!/usr/bin/env python3
"""Test Tesseract OCR directly on images"""

import cv2
import pytesseract
import numpy as np

print('Testing Tesseract OCR on Test Images')
print('=' * 70)

# Test images
images = ['sample_plate.jpg', 'test_plate_ocr.jpg']

for img_path in images:
    print(f'\nTesting: {img_path}')
    print('-' * 70)
    
    try:
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f'  [ERROR] Failed to read image')
            continue
        
        h, w = image.shape[:2]
        print(f'  Image size: {w}x{h} pixels')
        
        # Preprocess for better OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Test Tesseract
        try:
            # Get raw OCR output with detailed info
            data = pytesseract.image_to_data(filtered, output_type=pytesseract.Output.DICT)
            
            num_boxes = len(data['level'])
            print(f'  [OK] Tesseract found {num_boxes} text elements')
            
            if num_boxes > 0:
                # Show first 10 detections
                for i in range(min(10, num_boxes)):
                    text = data['text'][i]
                    conf = data['conf'][i]
                    
                    if text.strip() and int(conf) >= 0:
                        print(f'    [{i}] Text: "{text:15s}" | Conf: {conf}%')
            else:
                print('  [WARN] No text detected')
                
        except Exception as e:
            print(f'  [ERROR] Tesseract failed: {e}')
            
    except Exception as e:
        print(f'  [ERROR] {e}')
        import traceback
        traceback.print_exc()

print('\n' + '=' * 70)
print('[OK] Direct Tesseract test complete')
