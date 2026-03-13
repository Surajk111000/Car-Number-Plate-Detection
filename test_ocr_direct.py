"""
Direct OCR test - bypass detection and test OCR directly on images
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.ocr import PlateOCR


def create_clear_plate_image(filename="test_plate_ocr.jpg"):
    """Create a very clear plate image for OCR testing"""
    # Large image for better OCR
    img = Image.new("RGB", (800, 300), color="white")
    draw = ImageDraw.Draw(img)
    
    # Border
    draw.rectangle([10, 10, 790, 290], outline=(0, 48, 135), width=5)
    
    # EU section
    draw.rectangle([20, 20, 250, 120], fill=(255, 200, 0), outline=(0, 48, 135), width=3)
    
    try:
        font_eu = ImageFont.truetype("arial.ttf", 60)
        font_main = ImageFont.truetype("arial.ttf", 140)
    except:
        font_eu = ImageFont.load_default()
        font_main = ImageFont.load_default()
    
    # EU text
    draw.text((70, 35), "EU", fill=(0, 0, 0), font=font_eu)
    
    # Main plate text
    draw.text((280, 80), "ABC123XY", fill=(0, 0, 0), font=font_main)
    
    img.save(filename)
    print(f"✅ Created OCR test image: {filename}")
    return filename


def test_ocr_direct():
    """Test OCR directly on plate image"""
    print("\n" + "="*60)
    print("Direct OCR Test")
    print("="*60)
    
    # Create test image
    img_path = create_clear_plate_image("test_plate_ocr.jpg")
    
    # Load image with OpenCV
    image = cv2.imread(img_path)
    
    print(f"\n📷 Image shape: {image.shape}")
    print(f"📷 Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Initialize OCR
    print("\n🔄 Initializing OCR...")
    try:
        ocr = PlateOCR(languages=['en'], use_gpu=False)
        print("✅ OCR initialized")
    except Exception as e:
        print(f"❌ Failed to initialize OCR: {e}")
        return
    
    # Test full image OCR
    print("\n📖 Testing OCR on full image...")
    try:
        result = ocr.extract_text(image)
        print(f"✅ Full image OCR result:")
        print(f"   Raw text: '{result['full_text']}'")
        print(f"   Cleaned text: '{result['cleaned_text']}'")
        print(f"   Avg confidence: {result['avg_confidence']:.4f}")
        print(f"   Valid plate: {result['is_valid_plate']}")
        print(f"   Segments: {result['text']}")
        print(f"   Confidences: {result['confidence']}")
    except Exception as e:
        print(f"❌ OCR failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test EU region only
    print("\n📖 Testing OCR on EU region only (x: 20-250, y: 20-120)...")
    eu_region = image[20:120, 20:250]
    try:
        result_eu = ocr.extract_text(eu_region)
        print(f"✅ EU region OCR result:")
        print(f"   Text: '{result_eu['full_text']}'")
        print(f"   Confidence: {result_eu['avg_confidence']:.4f}")
    except Exception as e:
        print(f"❌ EU region OCR failed: {e}")
    
    # Test plate number region only
    print("\n📖 Testing OCR on plate number region (x: 280-800, y: 80-240)...")
    plate_region = image[80:240, 280:800]
    try:
        result_plate = ocr.extract_text(plate_region)
        print(f"✅ Plate number OCR result:")
        print(f"   Text: '{result_plate['full_text']}'")
        print(f"   Cleaned: '{result_plate['cleaned_text']}'")
        print(f"   Confidence: {result_plate['avg_confidence']:.4f}")
        print(f"   Valid: {result_plate['is_valid_plate']}")
    except Exception as e:
        print(f"❌ Plate region OCR failed: {e}")


if __name__ == "__main__":
    test_ocr_direct()
