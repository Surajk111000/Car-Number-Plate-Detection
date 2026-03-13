#!/usr/bin/env python
"""Debug OCR extraction with real license plate image."""

import time
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:5000"

def test_with_image(image_path):
    """Test API with image and show full response."""
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n📸 Testing with: {image_path}")
    print(f"⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    # Test health
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        health = resp.json()
        print(f"✓ API Health: {health['status']}")
    except Exception as e:
        print(f"❌ API not responding: {e}")
        return
    
    # Upload and predict
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            print(f"\n📤 Uploading image...")
            resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"\n✓ API Response:")
            print(f"  Plates detected: {data['summary']['detections']}")
            print(f"  Returned plates: {data['summary']['returned_plates']}")
            
            for idx, plate in enumerate(data['plates'], 1):
                print(f"\n  Plate {idx}:")
                print(f"    Detection Score: {plate['detection_score']:.4f}")
                print(f"    BBox: x={plate['bounding_box']['x_min']}-{plate['bounding_box']['x_max']}, "
                      f"y={plate['bounding_box']['y_min']}-{plate['bounding_box']['y_max']}")
                
                ocr = plate['ocr']
                print(f"    OCR:")
                print(f"      raw_text: '{ocr['raw_text']}'")
                print(f"      cleaned_text: '{ocr['cleaned_text']}'")
                print(f"      confidence: {ocr['avg_confidence']:.4f}")
                print(f"      is_valid: {ocr['is_valid_plate']}")
                if ocr['segments']:
                    print(f"      segments: {len(ocr['segments'])} chars")
                    for seg in ocr['segments'][:5]:
                        print(f"        - '{seg['text']}' ({seg['confidence']:.2f})")
        else:
            print(f"❌ API Error: {resp.status_code}")
            print(resp.text)
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    # Try different image paths
    image_paths = [
        "KL07AH9981.jpg",
        "./KL07AH9981.jpg",
        "g:/Projects/car-plate-improved/KL07AH9981.jpg",
    ]
    
    found = False
    for path in image_paths:
        if Path(path).exists():
            test_with_image(path)
            found = True
            break
    
    if not found:
        print("No KL07AH9981.jpg found. Looking for any jpg files...")
        jpg_files = list(Path(".").glob("*.jpg"))
        if jpg_files:
            print(f"Found: {jpg_files}")
            test_with_image(str(jpg_files[0]))
        else:
            print("No jpg files found in current directory")
            print("\n💡 Upload your KL07AH9981.jpg to the project root and run this script again")
            print(f"📍 Current directory: {Path.cwd()}")
