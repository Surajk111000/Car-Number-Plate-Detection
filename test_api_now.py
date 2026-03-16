#!/usr/bin/env python3
"""Test the API with updated EasyOCR detector"""

import requests
import time

# Give server time to start
time.sleep(3)

BASE_URL = 'http://localhost:5000'

# Test health endpoint first
print('Testing Health Endpoint...')
try:
    resp = requests.get(f'{BASE_URL}/health', timeout=10)
    print(f'  Status: {resp.status_code}')
    print(f'  Body: {resp.json()}')
except Exception as e:
    print(f'  [ERROR] {e}')
    exit(1)

print('\nTesting Plate Detection on 2 Images...')
print('=' * 70)

images = ['sample_plate.jpg', 'test_plate_ocr.jpg']

for img_path in images:
    print(f'\nImage: {img_path}')
    print('-' * 70)
    
    try:
        with open(img_path, 'rb') as f:
            files = {'image': f}
            resp = requests.post(f'{BASE_URL}/predict', files=files, timeout=30)
        
        if resp.status_code == 200:
            data = resp.json()
            num_dets = data.get('num_detections', 0)
            print(f'  [OK] Status 200')
            print(f'  Detections: {num_dets}')
            
            if num_dets > 0:
                print(f'  Scores: {data.get("detection_scores", [])}')
                if 'ocr_results' in data:
                    for i, ocr in enumerate(data['ocr_results'][:3]):
                        print(f'    [{i}] Text: "{ocr.get("full_text", "")}" Conf: {ocr.get("avg_confidence", 0):.2f}')
            else:
                print(f'  [WARN] No detections found')
        else:
            print(f'  [ERROR] Status {resp.status_code}')
            print(f'  Response: {resp.text}')
            
    except Exception as e:
        print(f'  [ERROR] {e}')

print('\n' + '=' * 70)
print('[OK] API test complete')
