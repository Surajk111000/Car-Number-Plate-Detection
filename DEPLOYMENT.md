# 🚀 Deployment Guide

Deploy your License Plate Detection API to the cloud in minutes!

---

## 🌐 Render (Recommended - Free Tier Available!)

The easiest way to deploy. Auto-deploys from GitHub whenever you push!

### Prerequisites
- GitHub account with the repository pushed
- Render.com account (free)

### Step 1: Connect GitHub
1. Go to [render.com](https://render.com)
2. Click **"New +"** in the top right
3. Select **"Web Service"**
4. Click **"Connect a repository"**
5. Find and select `Car-Number-Plate-Detection`
6. Click **"Connect"**

### Step 2: Configure Service
- **Name:** `car-plate-detector` (choose any name)
- **Environment:** Python 3
- **Build Command:** *(Auto-detected from render.yaml)*
- **Start Command:** *(Auto-detected from render.yaml)*
- **Instance Type:** Free (Free tier available!)

### Step 3: Deploy
Click **"Create Web Service"** and wait for deployment.

**Your live URL:** `https://your-service-name.onrender.com`

✅ **Done!** API is live and auto-deploys on every GitHub push.

---

## ☁️ Heroku (Traditional Alternative)

### Prerequisites
- Heroku CLI installed
- Heroku account

### Step 1: Login to Heroku
```bash
heroku login
```

### Step 2: Create App
```bash
heroku create your-app-name
```

### Step 3: Deploy Code
```bash
git push heroku main
```

### Step 4: View Logs
```bash
heroku logs --tail
```

**Your live URL:** `https://your-app-name.herokuapp.com`

### View Live App
```bash
heroku open
```

---

## 🐳 Docker (Any Cloud Provider)

Works on AWS, Google Cloud, Azure, DigitalOcean, etc.

### Prerequisites
- Docker installed
- Docker Hub account

### Step 1: Build Image
```bash
docker build -t your-username/car-plate-detector .
```

### Step 2: Test Locally
```bash
docker run -p 5000:5000 your-username/car-plate-detector
```

Visit: `http://localhost:5000/docs`

### Step 3: Push to Docker Hub
```bash
docker push your-username/car-plate-detector
```

### Step 4: Deploy to Your Cloud Provider

**AWS:**
```bash
# Use ECR instead of Docker Hub
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker tag car-plate-detector:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/car-plate-detector:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/car-plate-detector:latest
```

**DigitalOcean:**
```bash
doctl apps create --spec app.yaml
```

---

## 📋 Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] `render.yaml` or `Procfile` configured
- [ ] `requirements.txt` up-to-date
- [ ] `.gitignore` excludes large files (models, cache)
- [ ] API starts without errors locally
- [ ] Health endpoint responds: `GET /health`
- [ ] Can make predictions: `POST /predict`

---

## 🧪 Test Your Deployment

Once live:

### Health Check
```bash
curl https://your-app-name.onrender.com/health
```

Should return:
```json
{
  "status": "ready",
  "model_loaded": true,
  "ocr_loaded": true
}
```

### Test Detection
```bash
curl -X POST https://your-app-name.onrender.com/predict \
  -F "image=@sample_plate.jpg"
```

### Interactive Docs
Visit: `https://your-app-name.onrender.com/docs`

---

## 🔧 Troubleshooting

### Build Fails
```
Check logs for missing dependencies
Make sure requirements.txt is complete
Verify render.yaml syntax
```

### API Won't Start
```
Check: python app.py works locally
Verify imports in src/ modules
Check environment variables
```

### Slow Response
```
Free tier has limitations
Consider upgrading to Starter+ on Render
Ensure models load on startup
```

### Out of Memory
```
Free tier: 512MB RAM
Large models might not fit
Consider paid tier for production
```

---

## 📊 Performance Tips

1. **Use Gunicorn Workers:**
   - Handles multiple requests
   - Better than direct uvicorn
   - Already configured in Procfile/render.yaml

2. **CPU Optimization:**
   - EasyOCR runs on CPU (no GPU on free tier)
   - Typical request: 2-3 seconds
   - Can optimize with caching

3. **Scaling:**
   - Free tier: Single instance
   - Paid tier: Multiple instances + load balancing
   - Auto-scale based on demand

---

## 🔐 Security Best Practices

1. **Environment Variables:**
   ```bash
   # Never commit secrets!
   .env  # Added to .gitignore
   ```

2. **API Rate Limiting:**
   - Consider adding rate limits for production
   - Prevent abuse

3. **CORS Configuration:**
   - Currently allows all origins
   - Restrict in production

---

## 📞 Support

Having issues? Check the main README.md for troubleshooting and links to GitHub issues.

**Happy deploying! 🚀**
