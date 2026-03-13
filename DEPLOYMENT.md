# Deployment on Render

This guide shows how to deploy the Car Plate Detection API on Render.

## Quick Start (5 minutes)

### Step 1: Connect GitHub Repository to Render

1. Go to [render.com](https://render.com)
2. Click **"New +"** → **"Web Service"**
3. Select **"Connect a repository"**
4. Find and select `Car-Number-Plate-Detection`
5. Click **"Connect"**

### Step 2: Configure Service

**Name:** `car-plate-detector` (or any name you prefer)

**Environment:** Python 3

**Build Command:** (Auto-detected from render.yaml)
```bash
pip install --upgrade pip && pip install -r requirements.txt && python create_mock_model.py
```

**Start Command:** (Auto-detected from render.yaml)
```bash
uvicorn src.api_service:app --host 0.0.0.0 --port $PORT
```

**Instance Type:** Free (or Starter+ for production)

### Step 3: Add Environment Variables

Click **"Advanced"** and add:

| Key | Value |
|-----|-------|
| `MODEL_PATH` | `mock_model` |
| `PYTHONUNBUFFERED` | `1` |

### Step 4: Deploy

Click **"Create Web Service"** and wait for deployment to complete.

Your API will be available at: `https://your-service-name.onrender.com`

---

## Using render.yaml

The `render.yaml` file in the repository automatically configures Render. No manual setup needed—just connect your GitHub repo.

### Automatic Configuration Includes:
- ✅ Python 3.11 environment
- ✅ Build: Install dependencies + create mock model
- ✅ Start: Uvicorn FastAPI server
- ✅ Environment variables set
- ✅ Free tier instance

---

## Testing Your Deployment

Once deployed, test the API:

```bash
# Health check
https://your-service-name.onrender.com/health

# API documentation (interactive)
https://your-service-name.onrender.com/docs

# Predict endpoint
POST https://your-service-name.onrender.com/predict
```

### Example using curl:
```bash
curl -X POST \
  -F "image=@path/to/license_plate.jpg" \
  https://your-service-name.onrender.com/predict
```

---

## Production Setup

For production deployments:

1. **Use Starter or higher plan** (Free tier sleeps after 15 min inactivity)
2. **Train a real detection model** instead of using `mock_model`
3. **Set `MODEL_PATH`** environment variable to your trained model path
4. **Enable ORM cache** for faster startup
5. **Configure CORS** in production domain

### Update MODEL_PATH for Real Model:
```bash
# In Render dashboard:
# Environment → Add variable
MODEL_PATH=/path/to/your/trained/model
```

---

## Troubleshooting

### Service won't start?
Check logs in Render dashboard:
1. Click service name
2. Click **"Logs"**
3. Look for error messages

Common issues:
- **Memory error:** Upgrade to Starter plan
- **Module not found:** Ensure `requirements.txt` is in repository
- **Model loading failed:** Check `MODEL_PATH` environment variable

### OCR returning empty?
- Current: Using mock model (returns fixed detections)
- Solution: Train real TensorFlow model for accurate license plate detection

### API too slow?
- Free tier: Limited resources
- Solution: Upgrade to Starter ($7/month) or higher tier

---

## Environment Variables Reference

```bash
# Required
MODEL_PATH=mock_model              # Path to detection model

# Optional
PYTHONUNBUFFERED=1                 # Show logs in real-time
CONFIDENCE_THRESHOLD=0.5           # Detection confidence minimum
OCR_LANGUAGES=en                   # OCR language (e.g., en, zh, ar)
ENABLE_OCR=true                    # Enable/disable OCR
```

---

## For Existing Real Model

If you have a trained TensorFlow model:

1. Upload model to cloud storage (AWS S3, Google Cloud Storage, etc.)
2. Update `MODEL_PATH` to download URL or cloud path
3. Modify `create_mock_model.py` to skip mock model creation if real model exists

Example:
```python
# Update create_mock_model.py
if not os.path.exists("real_trained_model"):
    # Download from S3 or create mock
    pass
```

---

## Support

For issues:
- Check Render logs
- Test locally with `uvicorn src.api_service:app --reload`
- Verify `requirements.txt` has all dependencies
- Ensure `mock_model/` directory exists or `MODEL_PATH` is set correctly

---

## Links
- **Render Docs:** https://render.com/docs
- **FastAPI Deployment:** https://fastapi.tiangolo.com/deployment/
- **GitHub Repo:** https://github.com/Surajk111000/Car-Number-Plate-Detection
