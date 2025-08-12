# Claimsure Deployment Guide

## 🚨 **Important: Netlify is NOT Suitable for Python Applications**

**Netlify** is designed for:
- ✅ Static websites (HTML, CSS, JavaScript)
- ✅ Node.js applications
- ✅ JAMstack applications

**Netlify is NOT suitable for:**
- ❌ Python applications (like Claimsure)
- ❌ FastAPI backends
- ❌ Streamlit applications
- ❌ Applications requiring server-side processing

---

## 🎯 **Recommended Deployment Platforms for Claimsure**

### Option 1: **Render** (Recommended - Free Tier Available)
```bash
# 1. Connect your GitHub repository to Render
# 2. Create a new Web Service
# 3. Configure build settings:

Build Command: pip install -r requirements-minimal.txt
Start Command: python run.py

# 4. Set environment variables:
GOOGLE_AI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
HUGGINGFACE_API_KEY=your_key
OPENAI_API_KEY=your_key
```

**Advantages:**
- ✅ Free tier available
- ✅ Python support out of the box
- ✅ Easy GitHub integration
- ✅ Automatic deployments
- ✅ SSL certificates included

### Option 2: **Railway** (Recommended - Free Tier Available)
```bash
# 1. Connect your GitHub repository
# 2. Railway will auto-detect Python
# 3. Set environment variables in dashboard
# 4. Deploy automatically
```

**Advantages:**
- ✅ Free tier available
- ✅ Excellent Python support
- ✅ Simple deployment process
- ✅ Built-in monitoring

### Option 3: **Heroku** (Paid - Professional)
```bash
# Create Procfile
echo "web: python run.py" > Procfile

# Deploy
heroku create claimsure-app
heroku config:set GOOGLE_AI_API_KEY="your_key"
heroku config:set ANTHROPIC_API_KEY="your_key"
heroku config:set HUGGINGFACE_API_KEY="your_key"
heroku config:set OPENAI_API_KEY="your_key"
git push heroku master
```

### Option 4: **DigitalOcean App Platform**
```bash
# 1. Connect GitHub repository
# 2. Select Python as runtime
# 3. Configure build and run commands
# 4. Set environment variables
```

### Option 5: **AWS Elastic Beanstalk**
```bash
# 1. Create application
# 2. Upload code or connect Git
# 3. Configure environment
# 4. Deploy
```

---

## 🚀 **Quick Deployment with Render**

### Step 1: Prepare Your Repository
Ensure your repository has:
- ✅ `requirements-minimal.txt` (for production)
- ✅ `runtime.txt` (Python 3.11.9)
- ✅ `run.py` (main application file)
- ✅ Environment variables documented

### Step 2: Deploy to Render
1. **Sign up** at [render.com](https://render.com)
2. **Connect** your GitHub repository
3. **Create** a new Web Service
4. **Configure**:
   - **Name**: `claimsure-app`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements-minimal.txt`
   - **Start Command**: `python run.py`
   - **Plan**: Free (or paid for more resources)

### Step 3: Set Environment Variables
In Render dashboard, add:
```
GOOGLE_AI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_claude_api_key
HUGGINGFACE_API_KEY=your_hf_api_key
OPENAI_API_KEY=your_openai_api_key
ENVIRONMENT=production
```

### Step 4: Deploy
Click "Create Web Service" and wait for deployment.

---

## 🚀 **Quick Deployment with Railway**

### Step 1: Deploy to Railway
1. **Sign up** at [railway.app](https://railway.app)
2. **Connect** your GitHub repository
3. **Railway** will auto-detect Python
4. **Set** environment variables in dashboard
5. **Deploy** automatically

### Step 2: Configure Environment
Add these variables in Railway dashboard:
```
GOOGLE_AI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_claude_api_key
HUGGINGFACE_API_KEY=your_hf_api_key
OPENAI_API_KEY=your_openai_api_key
ENVIRONMENT=production
```

---

## 🔧 **Local Development Setup**

For local development and testing:

```bash
# 1. Clone repository
git clone https://github.com/Abhishek86798/ClaimSureHackrx.git
cd ClaimSureHackrx

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements-minimal.txt

# 4. Set environment variables
cp env_template.txt .env
# Edit .env with your API keys

# 5. Run application
python run.py
```

---

## 📊 **Platform Comparison**

| Platform | Free Tier | Python Support | Ease of Use | Cost |
|----------|-----------|----------------|-------------|------|
| **Render** | ✅ Yes | ✅ Excellent | ⭐⭐⭐⭐⭐ | $7/month |
| **Railway** | ✅ Yes | ✅ Excellent | ⭐⭐⭐⭐⭐ | $5/month |
| **Heroku** | ❌ No | ✅ Good | ⭐⭐⭐⭐ | $7/month |
| **DigitalOcean** | ❌ No | ✅ Good | ⭐⭐⭐ | $5/month |
| **AWS** | ❌ No | ✅ Good | ⭐⭐ | Variable |
| **Netlify** | ✅ Yes | ❌ No | ❌ N/A | Free |

---

## 🎯 **Recommendation**

**For immediate deployment:**
1. **Use Render** - Best free tier, excellent Python support
2. **Use Railway** - Simple deployment, good free tier
3. **Avoid Netlify** - Not suitable for Python applications

---

## 🚨 **Why Netlify Failed**

Netlify's build process:
1. Detects your repository as a Node.js project
2. Tries to run `npm install` (which doesn't exist)
3. Fails because there's no `package.json`
4. Cannot run Python applications

**Solution**: Use a platform designed for Python applications.

---

## 📞 **Need Help?**

If you need assistance with deployment:
1. Check the platform's documentation
2. Ensure all environment variables are set
3. Verify Python version compatibility
4. Check build logs for specific errors

**Recommended Next Step**: Deploy to Render or Railway for the best experience with Python applications.
