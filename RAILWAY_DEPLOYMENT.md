# 🚂 Railway Deployment Guide for Claimsure

> **Optimized deployment guide to avoid build timeouts and ensure successful Railway deployment**

## 🎯 **Quick Deploy to Railway**

**Deploy Claimsure in 5 minutes with Railway's optimized build:**

1. **Fork/Clone** this repository
2. **Sign up** at [railway.app](https://railway.app)
3. **Create New Project** and connect your repository
4. **Set environment variables** (API keys)
5. **Deploy** automatically

## 📋 **Prerequisites**

- GitHub account with repository access
- Railway account (free tier available)
- API keys for LLM services (optional for basic functionality)

## 🚀 **Step-by-Step Deployment**

### **Step 1: Prepare Repository**

Ensure your repository has these files:
```
├── Dockerfile.railway          # Railway-optimized Dockerfile
├── requirements-railway.txt    # Lightweight requirements
├── app.py                      # Main entry point
├── src/                        # Application code
└── .env.example               # Environment template
```

### **Step 2: Create Railway Project**

1. **Visit** [railway.app](https://railway.app)
2. **Sign in** with GitHub
3. **Click** "New Project"
4. **Select** "Deploy from GitHub repo"
5. **Choose** your Claimsure repository

### **Step 3: Configure Build Settings**

In Railway dashboard:

1. **Go to** Settings tab
2. **Set Build Command:**
   ```bash
   docker build -f Dockerfile.railway -t claimsure .
   ```
3. **Set Start Command:**
   ```bash
   python app.py
   ```

### **Step 4: Set Environment Variables**

Add these environment variables in Railway:

```bash
# Required
PORT=8000
PYTHONPATH=/app

# Optional API Keys (for full functionality)
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_claude_key
HUGGINGFACE_API_KEY=your_hf_key

# Optional Configuration
LOG_LEVEL=INFO
CACHE_DIR=/tmp/embeddings_cache
```

### **Step 5: Deploy**

1. **Click** "Deploy" button
2. **Wait** for build to complete (should be under 10 minutes)
3. **Check** deployment logs for success

## 🔧 **Optimization Features**

### **🚀 Build Optimizations**
- ✅ **Pre-built Python image**: Uses `python:3.10.12-slim`
- ✅ **Lightweight requirements**: Minimal dependencies
- ✅ **Docker layer caching**: Optimized for faster rebuilds
- ✅ **CPU-only FAISS**: Avoids GPU dependencies
- ✅ **Minimal system packages**: Only essential dependencies

### **💾 Memory Optimizations**
- ✅ **Batch processing**: Process chunks in small batches
- ✅ **Garbage collection**: Automatic memory cleanup
- ✅ **Memory monitoring**: Track usage and optimize
- ✅ **CPU-only mode**: Avoid GPU memory usage

### **⚡ Performance Features**
- ✅ **FastAPI**: High-performance async framework
- ✅ **Uvicorn**: Fast ASGI server
- ✅ **Optimized embeddings**: Efficient vector search
- ✅ **Caching**: Embedding and model caching

## 📊 **Resource Requirements**

### **Free Tier Limits**
- **Build time**: 10-15 minutes max
- **Memory**: 512MB RAM
- **Storage**: 1GB
- **Bandwidth**: 100GB/month

### **Recommended Settings**
- **Instance type**: Standard (1GB RAM)
- **Auto-scaling**: Disabled (for cost control)
- **Health checks**: Enabled

## 🔍 **Troubleshooting**

### **Build Timeout Issues**

**Problem**: Build takes longer than 15 minutes
**Solution**: 
1. Check `requirements-railway.txt` is being used
2. Ensure `Dockerfile.railway` is specified
3. Clear Railway cache and redeploy

### **Memory Issues**

**Problem**: Application runs out of memory
**Solution**:
1. Check memory usage in Railway logs
2. Reduce batch sizes in code
3. Upgrade to higher memory tier

### **Port Issues**

**Problem**: Application not accessible
**Solution**:
1. Ensure `PORT=8000` is set
2. Check Railway domain is generated
3. Verify health check endpoint works

### **API Key Issues**

**Problem**: LLM services not working
**Solution**:
1. Verify API keys are set correctly
2. Check API key permissions
3. Test with Railway's environment variable editor

## 📈 **Monitoring & Logs**

### **Railway Dashboard**
- **Deployments**: Track build and deployment status
- **Logs**: Real-time application logs
- **Metrics**: CPU, memory, and network usage
- **Health**: Application health status

### **Application Endpoints**
```bash
# Health check
GET https://your-app.railway.app/health

# API documentation
GET https://your-app.railway.app/docs

# Main endpoint
POST https://your-app.railway.app/hackrx/run
```

## 🔄 **Continuous Deployment**

### **Automatic Deploys**
- **Push to main**: Automatically triggers deployment
- **Branch protection**: Prevent accidental deployments
- **Rollback**: Easy rollback to previous versions

### **Environment Management**
- **Development**: Use feature branches
- **Staging**: Test before production
- **Production**: Main branch deployment

## 💰 **Cost Optimization**

### **Free Tier Usage**
- **Build time**: Under 10 minutes
- **Runtime**: 500 hours/month
- **Storage**: 1GB included
- **Bandwidth**: 100GB/month

### **Paid Tier Benefits**
- **Faster builds**: No timeout limits
- **More memory**: Up to 8GB RAM
- **Custom domains**: Professional URLs
- **Team collaboration**: Multiple users

## 🛠️ **Customization**

### **Environment Variables**
```bash
# Add custom configuration
CUSTOM_MODEL_PATH=/app/models
DEBUG_MODE=false
LOG_LEVEL=INFO
```

### **Build Optimization**
```dockerfile
# Add custom build steps
RUN pip install --no-cache-dir -r requirements-railway.txt
RUN python -m spacy download en_core_web_sm
```

## 📞 **Support**

### **Railway Support**
- **Documentation**: [railway.app/docs](https://railway.app/docs)
- **Discord**: [railway.app/discord](https://railway.app/discord)
- **GitHub**: [railwayapp/railway](https://github.com/railwayapp/railway)

### **Claimsure Support**
- **Issues**: GitHub repository issues
- **Documentation**: README.md and guides
- **Community**: Project discussions

## 🎉 **Success Checklist**

- ✅ Repository connected to Railway
- ✅ Environment variables configured
- ✅ Build completed successfully
- ✅ Application accessible via URL
- ✅ Health check endpoint working
- ✅ API documentation available
- ✅ Test query working

**Your Claimsure application is now deployed and ready to use! 🚀**



