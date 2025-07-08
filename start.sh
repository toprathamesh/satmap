#!/bin/bash

# Change Detection Platform - Startup Script

echo "🚀 Starting Change Detection Platform..."

# Set default environment variables
export FLASK_ENV=production
export FLASK_DEBUG=false
export DEVICE=cpu
export PORT=5000

# Check if we're in development or production
if [ "$NODE_ENV" = "development" ]; then
    echo "🔧 Running in development mode"
    export FLASK_DEBUG=true
    export CORS_ORIGINS="http://localhost:8080,http://localhost:3000"
else
    echo "🌐 Running in production mode"
    export CORS_ORIGINS="https://change-detection-frontend.onrender.com"
fi

# Create directories
echo "📁 Creating necessary directories..."
mkdir -p images/satellite images/demo images/results models

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "📦 Setting up virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Activate virtual environment
source venv/bin/activate

# Run the application
echo "🌟 Starting Flask application..."
cd backend
python app.py 