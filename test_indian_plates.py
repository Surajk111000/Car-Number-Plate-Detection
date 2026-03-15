#!/usr/bin/env python3
"""
Test script for testing the license plate detection and OCR system.
Upload your images and test them against the API.
"""

import requests
import json
import sys
import time
from pathlib import Path


def test_api_health():
    """Test if the API is ready"""
    print("🔍 Checking API health...")
    try:
        r = requests.get('http://localhost:5000/health', timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"✅ API is ready")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   OCR loaded: {data['ocr_loaded']}")
            return True
        else:
            print(f"❌ API returned status {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print(f"   Make sure FastAPI is running: python app.py")
        return False


def test_image(image_path: str):
    """Test a single image"""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    print(f"\n📷 Testing: {image_path.name} ({image_path.stat().st_size / 1024:.1f} KB)")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            start = time.time()
            r = requests.post('http://localhost:5000/predict', files=files, timeout=60)
            elapsed = time.time() - start
        
        if r.status_code != 200:
            print(f"❌ API error: {r.status_code}")
            print(r.text)
            return False
        
        result = r.json()
        print(f"✅ Success ({elapsed:.2f}s inference)")
        
        # Summary
        num_plates = len(result['plates'])
        print(f"   Detections: {num_plates}")
        
        if num_plates == 0:
            print("   ⚠️  No plates detected")
            return True
        
        # Details
        for i, plate in enumerate(result['plates'], 1):
            bbox = plate['bounding_box']
            score = plate['detection_score']
            ocr = plate.get('ocr', {})
            
            print(f"\n   Plate #{i}:")
            print(f"      Detection score: {score:.3f}")
            print(f"      Size: {bbox['width']}×{bbox['height']}px")
            print(f"      Position: ({bbox['x_min']}, {bbox['y_min']})")
            
            raw_text = ocr.get('raw_text', '')
            clean_text = ocr.get('cleaned_text', '')
            confidence = ocr.get('avg_confidence', 0)
            is_valid = ocr.get('is_valid_plate', False)
            
            if raw_text:
                print(f"      Raw OCR: \"{raw_text}\"")
                print(f"      Cleaned: \"{clean_text}\"")
                print(f"      Confidence: {confidence:.3f}")
                print(f"      Valid plate: {is_valid}")
            else:
                print(f"      ⚠️  No text detected")
                print(f"      Confidence: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main test function"""
    print("=" * 60)
    print("🚗 License Plate Detection & OCR Test")
    print("=" * 60)
    
    # Check health
    if not test_api_health():
        return 1
    
    # Test provided images
    test_images = [
        'sample_plate.jpg',
        'test_plate_ocr.jpg',
    ]
    
    success_count = 0
    for image_path in test_images:
        if Path(image_path).exists() and test_image(image_path):
            success_count += 1
    
    # Ask for user images
    print("\n" + "=" * 60)
    print("📸 Test Your Own Images")
    print("=" * 60)
    print("\nTo test your Indian license plate images:")
    print("1. Save them to this directory (e.g., indian_plate_1.jpg)")
    print("2. Run: python test_indian_plates.py <image_path>")
    print("\nExamples:")
    print("  python test_indian_plates.py indian_plate_1.jpg")
    print("  python test_indian_plates.py path/to/plate.jpg")
    
    # Test command-line arguments
    if len(sys.argv) > 1:
        print("\n" + "=" * 60)
        print("🧪 Testing provided images")
        print("=" * 60)
        for image_path in sys.argv[1:]:
            if test_image(image_path):
                success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Summary: {success_count} test(s) passed")
    print("=" * 60)
    
    return 0 if success_count > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
