# Claimsure - LLM-Powered Document Query System

A comprehensive Python-based system for intelligent document processing and querying using Large Language Models (LLMs). The system combines vector embeddings, semantic search, and AI-powered response generation to provide an intuitive document management solution.

## Features

- **📄 Document Processing**: Support for PDF, DOCX, and TXT files with intelligent chunking
- **🔍 Vector Search**: High-dimensional embeddings using sentence transformers for semantic search
- **🤖 LLM Integration**: OpenAI GPT models for context-aware response generation
- **🚀 FastAPI Backend**: RESTful API with comprehensive documentation
- **💻 Streamlit Frontend**: Modern web interface with drag-and-drop upload
- **🗄️ Vector Database**: Pinecone integration for scalable vector storage
- **⚙️ Configurable**: Highly customizable processing parameters and thresholds

## Project Structure

```
Claimsure/
├── src/                    # Main application source code
│   ├── document_loader.py  # Document loading and parsing
│   ├── text_chunker.py     # Text chunking and processing
│   ├── embeddings.py       # Vector embedding generation
│   ├── vector_store.py     # Vector database operations
│   ├── llm.py             # LLM interface and text generation
│   ├── query_processor.py # Query processing orchestration
│   ├── api.py             # FastAPI application
│   ├── models.py          # Pydantic models for API
│   └── ui.py              # Streamlit web interface
├── utils/                  # Utility functions and helpers
├── core/                   # Core business logic
├── tests/                  # Test files
├── data/                   # Data storage and sample files
├── config.py              # Configuration constants
├── requirements.txt        # Python dependencies
├── run.py                 # Main application runner
├── run_ui.py              # Streamlit UI launcher
└── test_api.py            # API testing script
```

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Claimsure
```

2. **Create and activate virtual environment:**

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
# Create .env file with your API keys
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
```

**💡 Quick Activation (Windows):**
```powershell
.\activate_venv.ps1
```

## Quick Start

### 1. Start the API Server

```bash
# Start the FastAPI backend
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 2. Start the Web Interface

```bash
# In a new terminal, start the Streamlit UI
python run_ui.py
```

The web interface will be available at:
- **UI**: http://localhost:8501

### 3. Test the System

```bash
# Test the API endpoints
python test_api.py
```

## Usage

### Web Interface

1. **Upload Documents**: Use the drag-and-drop interface to upload PDF, DOCX, or TXT files
2. **Process Documents**: Click "Process Documents" to chunk and embed your files
3. **Query Documents**: Ask natural language questions about your documents
4. **View Results**: See AI-generated responses with source citations

### API Usage

#### Upload Documents
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["data/sample_document.txt"],
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

#### Query Documents
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "top_k": 5,
    "similarity_threshold": 0.7
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/upload` | POST | Upload and process documents |
| `/query` | POST | Process single query |
| `/query/batch` | POST | Process multiple queries |
| `/system/info` | GET | System information |
| `/system/stats` | GET | System statistics |

## Configuration

The system is highly configurable through `config.py`:

- **Document Processing**: Chunk size, overlap, file size limits
- **Embedding Model**: Model selection and batch sizes
- **Vector Database**: Pinecone configuration and settings
- **LLM Settings**: Model selection, temperature, max tokens
- **API Configuration**: Host, port, reload settings
- **Search Parameters**: Top-K results, similarity thresholds

## Development

### Running Tests

```bash
# Test the API
python test_api.py

# Run unit tests
python -m pytest tests/
```

### Code Quality

```bash
# Format code
black .
isort .

# Type checking
mypy src/
```

## Architecture

The system follows a modular architecture:

1. **Document Loader** → Loads and parses documents
2. **Text Chunker** → Splits documents into chunks
3. **Embedding Generator** → Creates vector embeddings
4. **Vector Store** → Stores and retrieves embeddings
5. **LLM Interface** → Generates AI responses
6. **Query Processor** → Orchestrates the complete pipeline
7. **API Layer** → Provides REST endpoints
8. **Web Interface** → User-friendly UI

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.
