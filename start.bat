@echo off
echo 🚀 Starting Change Detection Platform...

REM Set default environment variables
set FLASK_ENV=production
set FLASK_DEBUG=false
set DEVICE=cpu
set PORT=5000
set CORS_ORIGINS=http://localhost:8080,http://localhost:3000

REM Create directories
echo 📁 Creating necessary directories...
if not exist "images" mkdir images
if not exist "images\satellite" mkdir images\satellite
if not exist "images\demo" mkdir images\demo
if not exist "images\results" mkdir images\results
if not exist "models" mkdir models

REM Run the application
echo 🌟 Starting Flask application...
cd backend
python app.py

pause 