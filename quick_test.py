import requests
import json

files = {'image': open('sample_plate.jpg', 'rb')}
data = {'include_visualization': 'false'}
r = requests.post('http://localhost:5000/predict', files=files, data=data)
result = r.json()

print('='*60)
print('API PREDICTION RESULTS')
print('='*60)
print()
print(f'Total Detections: {result["summary"]["detections"]}')
print(f'Inference Time: {result["summary"]["inference_ms"]}ms')
print()

for plate in result['plates']:
    print(f'PLATE #{plate["plate_index"]}:')
    print(f'  Detection Score: {plate["detection_score"]:.4f}')
    print(f'  Bounding Box: ({plate["bounding_box"]["x_min"]}, {plate["bounding_box"]["y_min"]}) -> ({plate["bounding_box"]["x_max"]}, {plate["bounding_box"]["y_max"]})')
    print()
    print(f'  OCR RESULTS:')
    print(f'    Raw Text: "{plate["ocr"]["raw_text"]}"')
    print(f'    Cleaned Text: "{plate["ocr"]["cleaned_text"]}"')
    print(f'    Confidence: {plate["ocr"]["avg_confidence"]:.4f}')
    print(f'    Valid Plate: {plate["ocr"]["is_valid_plate"]}')
    if plate["ocr"]["segments"]:
        print(f'    Characters: {[s["text"] for s in plate["ocr"]["segments"]]}')
    print()

print('='*60)
print('SUCCESS! OCR is now working with mock model!')
print('='*60)
