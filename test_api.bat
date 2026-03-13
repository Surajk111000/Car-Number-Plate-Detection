@echo off
REM Test FastAPI Plate Recognition API
REM Make sure API is running: .venv\Scripts\python.exe app.py

echo.
echo ================================================
echo FastAPI Plate Recognition - API Test
echo ================================================
echo.

REM Check if sample image exists
if not exist "sample_plate.jpg" (
    echo Creating sample image...
    .venv\Scripts\python.exe -c "from test_api import save_sample_image; save_sample_image('sample_plate.jpg')"
)

echo.
echo ================================================
echo 1. Testing /health endpoint
echo ================================================
echo.
curl -s http://localhost:5000/health | python -m json.tool
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not connect to API
    echo Make sure API is running: .venv\Scripts\python.exe app.py
    pause
    exit /b 1
)

echo.
echo ================================================
echo 2. Testing /predict endpoint with sample image
echo ================================================
echo.
curl -s -X POST http://localhost:5000/predict ^
  -F "image=@sample_plate.jpg" ^
  -F "include_visualization=false" | python -m json.tool

echo.
echo ================================================
echo 3. Full Python Test Suite
echo ================================================
echo.
.venv\Scripts\python.exe test_api.py

echo.
echo ================================================
echo All tests completed!
echo ================================================
echo.
echo API Swagger UI: http://localhost:5000/docs
echo API ReDoc:      http://localhost:5000/redoc
echo.
pause
