"""
Test script for the Plate Recognition API
Creates a sample plate image and tests the /predict endpoint
"""

import base64
import io
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_sample_plate_image():
    """Generate a synthetic license plate image for testing"""
    # Create white image (plate background)
    img = Image.new("RGB", (500, 180), color="white")
    draw = ImageDraw.Draw(img)
    
    # Add colored border (simulating license plate border)
    border_color = (0, 48, 135)  # Deep blue
    draw.rectangle([5, 5, 495, 175], outline=border_color, width=5)
    
    # Add yellow/white section at top left (like EU plates)
    draw.rectangle([15, 15, 160, 65], fill=(255, 200, 0), outline=border_color, width=3)
    
    # Add EU text - larger size
    try:
        font_small = ImageFont.truetype("arial.ttf", 40)
        font_large = ImageFont.truetype("arial.ttf", 100)
    except:
        font_small = ImageFont.load_default()
        font_large = ImageFont.load_default()
    
    # EU text
    draw.text((40, 20), "EU", fill=(0, 0, 0), font=font_small)
    
    # Main registration text - larger and clearer
    plate_text = "ABC123XY"
    draw.text((170, 50), plate_text, fill=(0, 0, 0), font=font_large)
    
    return img


def save_sample_image(filepath):
    """Save sample image to file"""
    img = create_sample_plate_image()
    img.save(filepath)
    print(f"✅ Sample image created: {filepath}")
    return filepath


def test_api_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_api_predict(image_path):
    """Test prediction endpoint with sample image"""
    print("\n" + "="*60)
    print("Testing /predict endpoint")
    print("="*60)
    
    try:
        with open(image_path, "rb") as img_file:
            files = {"image": img_file}
            data = {"include_visualization": "true"}
            
            print(f"Uploading image: {image_path}")
            response = requests.post(
                "http://localhost:5000/predict",
                files=files,
                data=data,
                timeout=30
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ Detection Success!")
                print(f"  - Detections found: {result['summary']['detections']}")
                print(f"  - Returned plates: {result['summary']['returned_plates']}")
                print(f"  - Inference time: {result['summary']['inference_ms']}ms")
                
                if result["plates"]:
                    for plate in result["plates"]:
                        print(f"\n  Plate #{plate['plate_index']}:")
                        print(f"    - Detection score: {plate['detection_score']:.4f}")
                        print(f"    - Bounding box: {plate['bounding_box']}")
                        if "ocr" in plate:
                            print(f"    - OCR text: {plate['ocr']['cleaned_text']}")
                            print(f"    - OCR confidence: {plate['ocr']['avg_confidence']:.4f}")
                
                if "visualization_base64" in result:
                    print(f"\n  - Visualization image included (base64 encoded)")
            else:
                print(f"❌ Error: {response.json()}")
                return False
                
            return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_api_docs():
    """Show how to access interactive docs"""
    print("\n" + "="*60)
    print("Interactive API Documentation")
    print("="*60)
    print("\n📖 Swagger UI (Try it in browser):")
    print("   http://localhost:5000/docs")
    print("\n📖 ReDoc (Alternative docs):")
    print("   http://localhost:5000/redoc")
    print("\n📖 OpenAPI Schema:")
    print("   http://localhost:5000/openapi.json")


if __name__ == "__main__":
    print("🚀 Plate Recognition API - Test Suite\n")
    
    # Step 1: Create sample image
    sample_image = "sample_plate.jpg"
    save_sample_image(sample_image)
    
    # Step 2: Test health
    health_ok = test_api_health()
    
    # Step 3: Test prediction
    if health_ok:
        print("\n✅ API is healthy, testing predictions...")
        test_api_predict(sample_image)
    else:
        print("\n⚠️  API is not healthy. Set MODEL_PATH environment variable:")
        print("   $env:MODEL_PATH='path/to/your/model'")
        print("   Then restart the API server")
    
    # Step 4: Show docs
    test_api_docs()
    
    print("\n" + "="*60)
