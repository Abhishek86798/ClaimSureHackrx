# Claimsure - Render Deployment Guide

## 🚀 **Deploy Claimsure to Render (Recommended)**

Render is the perfect platform for deploying Claimsure because it:
- ✅ **Free tier available** for testing and small projects
- ✅ **Excellent Python support** with automatic detection
- ✅ **Easy GitHub integration** with automatic deployments
- ✅ **SSL certificates** included automatically
- ✅ **Custom domains** supported
- ✅ **Environment variables** management
- ✅ **Build logs** for debugging

---

## 📋 **Prerequisites**

Before deploying, ensure you have:
- ✅ GitHub repository with Claimsure code
- ✅ Render account (free at [render.com](https://render.com))
- ✅ API keys for LLM services (Gemini, Claude, Hugging Face, OpenAI)

---

## 🎯 **Step-by-Step Deployment**

### **Step 1: Prepare Your Repository**

Your repository should already have these files:
- ✅ `requirements-minimal.txt` (for production)
- ✅ `runtime.txt` (Python 3.10.12)
- ✅ `run.py` (main application file)
- ✅ `src/main.py` (FastAPI app)

### **Step 2: Sign Up for Render**

1. **Go to** [render.com](https://render.com)
2. **Sign up** with your GitHub account
3. **Verify** your email address

### **Step 3: Create New Web Service**

1. **Click** "New +" button
2. **Select** "Web Service"
3. **Connect** your GitHub repository
4. **Choose** the repository: `Abhishek86798/ClaimSureHackrx`

### **Step 4: Configure Build Settings**

**Service Settings:**
- **Name**: `claimsure-app` (or your preferred name)
- **Environment**: `Python 3`
- **Region**: Choose closest to your users
- **Branch**: `master` (or your main branch)
- **Root Directory**: Leave empty (root of repository)

**Build & Deploy Settings:**
- **Build Command**: `pip install -r requirements-minimal.txt`
- **Start Command**: `python run.py`
- **Plan**: Free (or paid for more resources)

### **Step 5: Set Environment Variables**

**Required Environment Variables:**
```
GOOGLE_AI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_claude_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

**How to add them:**
1. **Click** "Environment" tab
2. **Add** each variable with its value
3. **Save** the configuration

### **Step 6: Deploy**

1. **Click** "Create Web Service"
2. **Wait** for build to complete (5-10 minutes)
3. **Check** build logs for any errors
4. **Access** your app at the provided URL

---

## 🔧 **Configuration Files**

### **requirements-minimal.txt** (Already Updated)
```txt
# Claimsure - Minimal Requirements for Production
# Compatible with Python 3.8+

# Core Framework
fastapi>=0.100.0,<0.120.0
uvicorn>=0.20.0,<0.40.0
pydantic>=2.0.0,<3.0.0
python-multipart>=0.0.6,<1.0.0
python-dotenv>=1.0.0,<2.0.0

# LLM Services (Core)
anthropic>=0.60.0,<0.70.0
google-generativeai>=0.8.0,<1.0.0
openai>=1.0.0,<2.0.0
huggingface-hub>=0.16.0,<1.0.0

# Document Processing
PyPDF2>=3.0.0,<4.0.0
python-docx>=0.8.11,<1.0.0

# Vector Database
faiss-cpu>=1.7.4,<2.0.0
numpy>=1.21.0,<2.0.0
sentence-transformers>=2.2.0,<3.0.0

# Web UI
streamlit>=1.25.0,<2.0.0
requests>=2.28.0,<3.0.0

# Utilities
aiofiles>=23.0.0,<24.0.0
python-json-logger>=2.0.0,<3.0.0
```

### **runtime.txt** (Already Updated)
```txt
python-3.10.12
```

### **run.py** (Main Entry Point)
```python
import uvicorn
from src.main import app

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
```

---

## 🐳 **Docker Compatibility**

### **Docker Build Issues Fixed**

The project now includes Docker-specific configurations to avoid common build issues:

**✅ Fixed Issues:**
- **Windows-specific packages** removed (pywin32, pywinpty)
- **Linux-compatible requirements** in `requirements-docker.txt`
- **Proper system dependencies** in Dockerfile
- **Health checks** for container monitoring

**Docker Files:**
- `Dockerfile` - Linux-compatible container configuration
- `requirements-docker.txt` - Linux-specific dependencies
- `docker-compose.yml` - Multi-service orchestration
- `build-docker.sh` - Build and test script

### **Docker Build Commands**

```bash
# Build Docker image
docker build -t claimsure:latest .

# Run container
docker run -d --name claimsure -p 8000:8000 claimsure:latest

# Test with Docker Compose
docker-compose up -d
```

---

## 🚨 **Common Issues & Solutions**

### **Issue 1: Build Fails with Python Version**
**Error**: `python-build: definition not found: python-3.10.12`
**Solution**: 
- ✅ Already fixed with `runtime.txt`
- ✅ Render supports Python 3.10.12

### **Issue 2: Missing Dependencies**
**Error**: `ModuleNotFoundError: No module named 'fastapi'`
**Solution**:
- ✅ Use `requirements-minimal.txt` (not `requirements.txt`)
- ✅ Check build logs for specific missing packages

### **Issue 3: Environment Variables Not Set**
**Error**: `KeyError: 'GOOGLE_AI_API_KEY'`
**Solution**:
- ✅ Add all required environment variables in Render dashboard
- ✅ Ensure no typos in variable names

### **Issue 4: Port Issues**
**Error**: `Address already in use`
**Solution**:
- ✅ Use port 8000 (Render's default)
- ✅ Set `API_HOST=0.0.0.0` in environment variables

### **Issue 5: Memory Issues**
**Error**: `MemoryError` or build timeout
**Solution**:
- ✅ Upgrade to paid plan for more resources
- ✅ Use `requirements-minimal.txt` (lighter dependencies)

### **Issue 6: Docker Build Fails**
**Error**: `pywin32==311` not found
**Solution**:
- ✅ Use `requirements-docker.txt` for Docker builds
- ✅ Windows-specific packages excluded
- ✅ Linux-compatible dependencies only

---

## 🔍 **Testing Your Deployment**

### **Health Check**
```bash
curl https://your-app-name.onrender.com/health
```
**Expected Response**: `{"status": "healthy"}`

### **API Test**
```bash
curl -X POST https://your-app-name.onrender.com/hackrx/run \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "sample_insurance_policy.txt",
    "questions": ["What is covered under this policy?"]
  }'
```

### **Web Interface**
Visit: `https://your-app-name.onrender.com:8501`

---

## 📊 **Render Plans Comparison**

| Plan | Price | Memory | CPU | Build Time | Sleep |
|------|-------|--------|-----|------------|-------|
| **Free** | $0/month | 512MB | Shared | 10 min | After 15 min |
| **Starter** | $7/month | 512MB | Shared | 10 min | Never |
| **Standard** | $25/month | 1GB | Shared | 10 min | Never |
| **Pro** | $50/month | 2GB | Dedicated | 10 min | Never |

**Recommendation**: Start with Free plan, upgrade if needed.

---

## 🔄 **Automatic Deployments**

Render automatically deploys when you:
- ✅ **Push** to the connected branch (master)
- ✅ **Update** environment variables
- ✅ **Redeploy** manually from dashboard

**Deployment Process**:
1. **Build**: Install dependencies from `requirements-minimal.txt`
2. **Start**: Run `python run.py`
3. **Health Check**: Verify `/health` endpoint responds
4. **Live**: Your app is accessible at the provided URL

---

## 🛠️ **Monitoring & Logs**

### **View Logs**
1. **Go to** your service dashboard
2. **Click** "Logs" tab
3. **Monitor** real-time logs

### **Health Monitoring**
- ✅ **Automatic health checks** every 30 seconds
- ✅ **Email notifications** for failures
- ✅ **Uptime monitoring** included

### **Performance Metrics**
- ✅ **Response times**
- ✅ **Memory usage**
- ✅ **CPU usage**
- ✅ **Request count**

---

## 🎉 **Success Checklist**

After deployment, verify:
- ✅ **Build completed** without errors
- ✅ **Health endpoint** responds: `/health`
- ✅ **API endpoint** works: `/hackrx/run`
- ✅ **Environment variables** are set correctly
- ✅ **Logs** show no errors
- ✅ **SSL certificate** is active (https://)

---

## 📞 **Need Help?**

**Render Support**:
- 📧 Email: support@render.com
- 📚 Docs: [docs.render.com](https://docs.render.com)
- 💬 Community: [community.render.com](https://community.render.com)

**Common Commands**:
```bash
# Check deployment status
curl https://your-app-name.onrender.com/health

# Test API endpoint
curl -X POST https://your-app-name.onrender.com/hackrx/run \
  -H "Content-Type: application/json" \
  -d '{"documents": "test", "questions": ["test"]}'

# View logs (in Render dashboard)
# Go to Logs tab in your service
```

---

## 🚀 **Next Steps**

1. **Deploy** to Render using this guide
2. **Test** all endpoints
3. **Monitor** performance and logs
4. **Scale** if needed (upgrade plan)
5. **Custom domain** (optional)

**Your Claimsure app will be live at**: `https://your-app-name.onrender.com` 🎉
