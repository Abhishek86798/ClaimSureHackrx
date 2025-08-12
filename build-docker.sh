#!/bin/bash

# Claimsure Docker Build Script
# This script builds and tests the Docker image

set -e

echo "🐳 Building Claimsure Docker Image..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image using the simple Dockerfile
echo "📦 Building with Dockerfile.simple..."
docker build -f Dockerfile.simple -t claimsure:latest .

echo "✅ Docker image built successfully!"

# Test the image
echo "🧪 Testing Docker image..."

# Run a quick test to ensure the image works
docker run --rm -d --name claimsure-test -p 8000:8000 claimsure:latest

# Wait for the container to start
echo "⏳ Waiting for container to start..."
sleep 15

# Test health endpoint
echo "🔍 Testing health endpoint..."
if curl -f http://localhost:8000/health; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
    echo "📋 Container logs:"
    docker logs claimsure-test
fi

# Stop the test container
docker stop claimsure-test

echo "🎉 Docker build and test completed successfully!"
echo ""
echo "🚀 To run the container:"
echo "   docker run -d --name claimsure -p 8000:8000 claimsure:latest"
echo ""
echo "🌐 Access the API at: http://localhost:8000"
echo "📖 API docs at: http://localhost:8000/docs"
