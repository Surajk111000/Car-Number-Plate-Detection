#!/usr/bin/env python3
"""Direct test of EasyOCR without API"""

import cv2
import easyocr
import numpy as np

print('Testing EasyOCR Directly...')
print('=' * 70)

# Initialize reader
print('Initializing EasyOCR Reader...')
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='/tmp/easyocr')
print('[OK] Reader initialized\n')

images = ['sample_plate.jpg', 'test_plate_ocr.jpg']

for img_path in images:
    print(f'Testing: {img_path}')
    print('-' * 70)
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f'  [ERROR] Failed to read image')
            continue
        
        h, w = img.shape[:2]
        print(f'  Image: {w}x{h} pixels')
        
        # Test direct EasyOCR
        result = reader.readtext(img, detail=1)
        
        print(f'  Detections: {len(result)}')
        
        if result:
            for i, (bbox, text, conf) in enumerate(result[:10]):
                print(f'    [{i}] "{text:20s}" | Conf: {conf:.4f}')
        else:
            print('  [WARN] No text detected')
            
    except Exception as e:
        print(f'  [ERROR] {e}')
        import traceback
        traceback.print_exc()

print('\n' + '=' * 70)
print('[OK] Direct test complete')
