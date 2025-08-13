# Claimsure - LLM-Powered Insurance Document Query System

[![Python](https://img.shields.io/badge/Python-3.10.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Intelligent document processing and querying system for insurance documents using hybrid LLM processing**

## 🚀 **Quick Deploy Options**

### **Option 1: Railway (Recommended - Fastest)**
**Deploy Claimsure in 3-5 minutes with Railway's optimized build:**

1. **Fork/Clone** this repository
2. **Sign up** at [railway.app](https://railway.app)
3. **Create New Project** and connect your repository
4. **Set environment variables** (API keys)
5. **Deploy** automatically

📖 **[Complete Railway Deployment Guide](RAILWAY_DEPLOYMENT.md)**

### **Option 2: Render (Alternative)**
**Deploy Claimsure in 5 minutes with Render's free tier:**

1. **Fork/Clone** this repository
2. **Sign up** at [render.com](https://render.com)
3. **Create Web Service** and connect your repository
4. **Set environment variables** (API keys)
5. **Deploy** automatically

📖 **[Complete Render Deployment Guide](RENDER_DEPLOYMENT.md)**

---

## 🎯 **Features**

- 🔍 **Hybrid LLM Processing**: Claude 3.5 Sonnet, Gemini, Hugging Face, OpenAI GPT-3.5
- 📄 **Multi-format Support**: PDF, DOCX, TXT documents
- 🧠 **Semantic Search**: FAISS vector embeddings for intelligent retrieval
- 🤖 **Query Understanding**: Intent classification and entity extraction
- 💡 **Decision Logic**: AI-powered reasoning over retrieved clauses
- 🌐 **RESTful API**: FastAPI backend with comprehensive endpoints
- 🎨 **Web Interface**: Streamlit UI for document upload and querying
- 🔄 **Fallback Systems**: Robust error handling and service redundancy

---

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Query         │    │   Response      │
│   Upload        │───▶│   Processing    │───▶│   Generation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chunking &    │    │   Semantic      │    │   JSON          │
│   Embedding     │    │   Retrieval     │    │   Formatting    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🚀 **Deployment Options**

### **1. Render (Recommended - Free)**
```bash
# Build Command: pip install -r requirements-minimal.txt
# Start Command: python run.py
```
- ✅ **Free tier available**
- ✅ **Automatic deployments**
- ✅ **SSL certificates included**
- 📖 **[Deployment Guide](RENDER_DEPLOYMENT.md)**

### **2. Railway (Simple - Free)**
```bash
# Auto-detects Python
# No configuration needed
```
- ✅ **Free tier available**
- ✅ **Simple deployment process**

### **3. Heroku (Professional - Paid)**
```bash
# Uses Procfile: web: python run.py
# Requires paid plan
```

### **4. Local Development**
```bash
git clone https://github.com/Abhishek86798/ClaimSureHackrx.git
cd ClaimSureHackrx
pip install -r requirements-minimal.txt
python run.py
```

---

## 📋 **Prerequisites**

- **Python 3.10+** (3.10.12 recommended)
- **API Keys** for LLM services:
  - Google Gemini API
  - Anthropic Claude API
  - Hugging Face API
  - OpenAI API (optional)

---

## 🔧 **Installation**

### **For Production (Render)**
```bash
# Use requirements-minimal.txt (already configured)
pip install -r requirements-minimal.txt
```

### **For Development**
```bash
# Use full requirements.txt
pip install -r requirements.txt
```

### **Environment Setup**
```bash
# Copy environment template
cp env_template.txt .env

# Edit .env with your API keys
GOOGLE_AI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_claude_key
HUGGINGFACE_API_KEY=your_hf_key
OPENAI_API_KEY=your_openai_key
```

---

## 🚀 **Quick Start**

### **1. Start the API Server**
```bash
python run.py
```
**API will be available at**: `http://localhost:8000`

### **2. Start the Web Interface**
```bash
streamlit run app/webapp.py
```
**Web UI will be available at**: `http://localhost:8501`

### **3. Test the API**
```bash
# Health check
curl http://localhost:8000/health

# Process documents and questions
curl -X POST http://localhost:8000/hackrx/run \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "sample_insurance_policy.txt",
    "questions": ["What is covered under this policy?"]
  }'
```

---

## 📚 **API Endpoints**

### **Health Check**
```http
GET /health
```
**Response**: `{"status": "healthy"}`

### **Process Documents & Questions**
```http
POST /hackrx/run
Content-Type: application/json

{
  "documents": "document_url_or_content",
  "questions": ["question1", "question2"]
}
```

### **Statistics**
```http
GET /stats
```
**Response**: System statistics and service status

---

## 🔧 **Configuration**

### **Environment Variables**
| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_AI_API_KEY` | Gemini API key | Yes |
| `ANTHROPIC_API_KEY` | Claude API key | Yes |
| `HUGGINGFACE_API_KEY` | Hugging Face API key | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Optional |
| `ENVIRONMENT` | Environment (production/development) | No |
| `API_HOST` | API host (0.0.0.0 for production) | No |
| `API_PORT` | API port (8000) | No |

### **LLM Service Priority**
1. **Hugging Face Enhanced** (primary)
2. **Claude 3.5 Sonnet** (fallback)
3. **Gemini** (fallback)
4. **Local Mistral** (offline fallback)
5. **OpenAI GPT-3.5** (last resort)

---

## 🧪 **Testing**

### **Run Deployment Check**
```bash
python deploy.py
```

### **Test API Endpoints**
```bash
# Health check
curl http://localhost:8000/health

# Process sample document
curl -X POST http://localhost:8000/hackrx/run \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "sample_insurance_policy.txt",
    "questions": ["What is the waiting period for pre-existing diseases?"]
  }'
```

---

## 📁 **Project Structure**

```
ClaimSureHackrx/
├── src/
│   ├── main.py                 # FastAPI application
│   ├── core/
│   │   ├── hybrid_processor.py # LLM service orchestration
│   │   ├── embeddings.py       # Vector embeddings
│   │   ├── query_processing.py # Query understanding
│   │   ├── retrieval.py        # Semantic search
│   │   ├── logic_evaluator.py  # Decision logic
│   │   └── services/           # LLM service integrations
│   ├── document_loader.py      # Document parsing
│   └── text_chunker.py         # Text chunking
├── app/
│   └── webapp.py              # Streamlit web interface
├── requirements-minimal.txt    # Production dependencies
├── requirements.txt           # Development dependencies
├── runtime.txt                # Python version specification
├── run.py                     # Application entry point
├── RENDER_DEPLOYMENT.md       # Render deployment guide
└── README.md                  # This file
```

---

## 🔄 **Development Workflow**

1. **Clone** the repository
2. **Install** dependencies: `pip install -r requirements.txt`
3. **Set** environment variables in `.env`
4. **Run** locally: `python run.py`
5. **Test** endpoints and functionality
6. **Deploy** to Render using the deployment guide

---

## 🚨 **Troubleshooting**

### **Common Issues**

**Build Fails on Render:**
- ✅ Use `requirements-minimal.txt` (not `requirements.txt`)
- ✅ Check Python version in `runtime.txt`
- ✅ Verify environment variables are set

**API Keys Not Working:**
- ✅ Check API key format and validity
- ✅ Ensure environment variables are set correctly
- ✅ Verify API quotas and billing

**Memory Issues:**
- ✅ Upgrade to paid Render plan
- ✅ Use minimal requirements file
- ✅ Monitor resource usage

---

## 📊 **Performance**

- **Response Time**: < 5 seconds for typical queries
- **Memory Usage**: ~512MB (free tier)
- **Concurrent Requests**: 10+ (depends on plan)
- **Document Size**: Up to 50MB per document

---

## 🤝 **Contributing**

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** thoroughly
5. **Submit** a pull request

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **FastAPI** for the web framework
- **Streamlit** for the web interface
- **Hugging Face** for open-source models
- **Anthropic** for Claude API
- **Google** for Gemini API
- **OpenAI** for GPT API

---

## 📞 **Support**

- 📖 **[Render Deployment Guide](RENDER_DEPLOYMENT.md)**
- 🐛 **[Issues](https://github.com/Abhishek86798/ClaimSureHackrx/issues)**
- 📧 **Email**: [Your Email]

---

**Ready to deploy?** 🚀 **[Start with Render](RENDER_DEPLOYMENT.md)**
