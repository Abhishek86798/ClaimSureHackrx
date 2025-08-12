# Docker Troubleshooting Guide

## 🐳 **Common Docker Build Issues & Solutions**

### **Issue 1: pip command not found**
**Error**: `/bin/bash: line 1: pip: command not found`

**Solutions**:
1. **Use explicit Python commands**:
   ```dockerfile
   RUN python -m pip install --no-cache-dir --upgrade pip
   RUN python -m pip install --no-cache-dir -r requirements.txt
   ```

2. **Use full Python image**:
   ```dockerfile
   FROM python:3.10.12  # Instead of python:3.10.12-slim
   ```

3. **Check Python installation**:
   ```dockerfile
   RUN python --version
   RUN which python
   RUN which pip
   ```

### **Issue 2: Build context too large**
**Error**: `failed to solve: failed to compute cache key`

**Solutions**:
1. **Use .dockerignore**:
   ```dockerignore
   .git
   __pycache__
   *.pyc
   venv/
   .env
   data/
   logs/
   ```

2. **Copy only necessary files**:
   ```dockerfile
   COPY requirements.txt .
   COPY src/ ./src/
   COPY app.py .
   ```

### **Issue 3: Memory issues during build**
**Error**: `MemoryError` or build timeout

**Solutions**:
1. **Increase Docker memory** (Docker Desktop settings)
2. **Use multi-stage builds**:
   ```dockerfile
   FROM python:3.10.12 as builder
   COPY requirements.txt .
   RUN pip install --user -r requirements.txt
   
   FROM python:3.10.12-slim
   COPY --from=builder /root/.local /root/.local
   ```

### **Issue 4: System dependencies missing**
**Error**: `fatal error: Python.h: No such file or directory`

**Solutions**:
1. **Install build tools**:
   ```dockerfile
   RUN apt-get update && apt-get install -y \
       gcc \
       g++ \
       python3-dev \
       && rm -rf /var/lib/apt/lists/*
   ```

### **Issue 5: Port already in use**
**Error**: `Address already in use`

**Solutions**:
1. **Use different port**:
   ```bash
   docker run -p 8001:8000 claimsure:latest
   ```

2. **Stop existing containers**:
   ```bash
   docker stop $(docker ps -q)
   ```

---

## 🔧 **Docker Build Commands**

### **Build with specific Dockerfile**
```bash
# Use simple Dockerfile
docker build -f Dockerfile.simple -t claimsure:latest .

# Use main Dockerfile
docker build -f Dockerfile -t claimsure:latest .
```

### **Build with no cache**
```bash
docker build --no-cache -t claimsure:latest .
```

### **Build with progress**
```bash
docker build --progress=plain -t claimsure:latest .
```

### **Build with specific platform**
```bash
docker build --platform linux/amd64 -t claimsure:latest .
```

---

## 🧪 **Testing Docker Builds**

### **Test build step by step**
```bash
# Build with verbose output
docker build --progress=plain -t claimsure:latest .

# Check image layers
docker history claimsure:latest

# Run interactive shell
docker run -it claimsure:latest /bin/bash
```

### **Test application**
```bash
# Run container
docker run -d --name claimsure-test -p 8000:8000 claimsure:latest

# Check logs
docker logs claimsure-test

# Test health endpoint
curl http://localhost:8000/health

# Stop container
docker stop claimsure-test
docker rm claimsure-test
```

---

## 📋 **Dockerfile Best Practices**

### **1. Use specific base image**
```dockerfile
FROM python:3.10.12  # Specific version
```

### **2. Set working directory**
```dockerfile
WORKDIR /app
```

### **3. Copy requirements first**
```dockerfile
COPY requirements.txt .
RUN pip install -r requirements.txt
```

### **4. Copy application code**
```dockerfile
COPY . .
```

### **5. Use non-root user**
```dockerfile
RUN adduser --disabled-password --gecos '' appuser
USER appuser
```

### **6. Expose port**
```dockerfile
EXPOSE 8000
```

### **7. Health check**
```dockerfile
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

---

## 🚀 **Quick Fix Commands**

### **Clean up Docker**
```bash
# Remove all containers
docker rm -f $(docker ps -aq)

# Remove all images
docker rmi -f $(docker images -q)

# Remove all volumes
docker volume prune -f

# Remove all networks
docker network prune -f

# Clean everything
docker system prune -a -f
```

### **Rebuild from scratch**
```bash
# Clean build
docker build --no-cache -t claimsure:latest .

# Test immediately
docker run -d --name test -p 8000:8000 claimsure:latest
curl http://localhost:8000/health
docker stop test && docker rm test
```

---

## 📞 **Need Help?**

**Common Issues**:
1. **Docker not running** - Start Docker Desktop
2. **Permission denied** - Run with sudo (Linux) or as admin (Windows)
3. **Port conflicts** - Use different ports
4. **Memory issues** - Increase Docker memory allocation

**Debug Commands**:
```bash
# Check Docker version
docker --version

# Check Docker info
docker info

# Check running containers
docker ps

# Check all containers
docker ps -a

# Check images
docker images

# Check logs
docker logs <container_name>
```

---

## 🎯 **Recommended Dockerfile**

Use `Dockerfile.simple` for the most reliable builds:

```dockerfile
FROM python:3.10.12
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
```

This Dockerfile is simple, reliable, and should work in most environments.
