# FastAPI Plate Recognition - Test Examples

## 1. Using Python Test Script (Recommended)

```powershell
.venv\Scripts\python.exe test_api.py
```

This will:
- ✅ Create a sample plate image
- ✅ Check API health
- ✅ Send image to /predict endpoint
- ✅ Display results


## 2. Using PowerShell (curl equivalent)

### Test Health Endpoint
```powershell
(Invoke-WebRequest -UseBasicParsing http://localhost:5000/health).Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

### Test Predict Endpoint
```powershell
$form = @{
    image = Get-Item -Path "sample_plate.jpg"
    include_visualization = "true"
}

$response = Invoke-WebRequest -UseBasicParsing `
    -Uri "http://localhost:5000/predict" `
    -Method Post `
    -Form $form

$response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

### Save Response to File
```powershell
$form = @{
    image = Get-Item -Path "sample_plate.jpg"
    include_visualization = "true"
}

$response = Invoke-WebRequest -UseBasicParsing `
    -Uri "http://localhost:5000/predict" `
    -Method Post `
    -Form $form

$response.Content | Out-File -Path "api_response.json"
```


## 3. Using curl (Git Bash or WSL)

### Test Health
```bash
curl http://localhost:5000/health
```

### Test Predict
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@sample_plate.jpg" \
  -F "include_visualization=true"
```

### Save Response
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@sample_plate.jpg" \
  -F "include_visualization=true" > api_response.json
```


## 4. Using Browser

### Interactive Swagger UI
```
http://localhost:5000/docs
```
- Click "Try it out" on any endpoint
- Upload image directly in browser
- See live response


## 5. Using Postman or Insomnia

**URL:** `http://localhost:5000/predict`

**Method:** POST

**Form Data:**
- Key: `image` | Type: `File` | Value: `sample_plate.jpg`
- Key: `include_visualization` | Type: `Text` | Value: `true`


## Response Schema (Success)

```json
{
  "success": true,
  "image": {
    "width": 1920,
    "height": 1440,
    "channels": 3
  },
  "summary": {
    "detections": 2,
    "returned_plates": 2,
    "inference_ms": 150.45
  },
  "plates": [
    {
      "plate_index": 1,
      "detection_score": 0.95,
      "bounding_box": {
        "x_min": 100,
        "y_min": 200,
        "x_max": 300,
        "y_max": 350,
        "width": 200,
        "height": 150
      },
      "ocr": {
        "raw_text": "ABC123XY",
        "cleaned_text": "ABC123XY",
        "avg_confidence": 0.92,
        "is_valid_plate": true,
        "segments": [
          {"text": "A", "confidence": 0.98},
          {"text": "B", "confidence": 0.96},
          {"text": "C", "confidence": 0.94}
        ]
      }
    }
  ],
  "visualization_base64": "iVBORw0KGgoAAAANS..."
}
```


## Response Schema (Error - Model Not Loaded)

```json
{
  "detail": ["SERVICE NOT READY: MODEL_PATH environment variable is required"]
}
```

**Solution:**
```powershell
$env:MODEL_PATH="D:\path\to\your\saved_model"
.venv\Scripts\python.exe app.py
```


## 6. Image Requirements

- **Format:** JPG, PNG
- **Size:** 100x100 to 4000x4000 pixels
- **Color:** RGB or Grayscale
- **Max Size:** 8MB (default)


## 7. Environment Setup

### Required (to get predictions)
```powershell
$env:MODEL_PATH="path/to/tensorflow/savedmodel"
```

### Optional
```powershell
$env:CONFIDENCE_THRESHOLD="0.5"
$env:NMS_THRESHOLD="0.5"
$env:PADDING="10"
$env:ENABLE_OCR="true"
$env:OCR_USE_GPU="false"
$env:OCR_LANGUAGES="en"
```


## Files Generated

- `sample_plate.jpg` - Test image created by `test_api.py`
- `api_response.json` - API response (if saved)

