
# 🛰️  Google Earth Engine Setup Guide

## Option 1: Service Account (Recommended for Production)

1. Go to Google Cloud Console (https://console.cloud.google.com/)
2. Create a new project or select existing project
3. Enable Earth Engine API
4. Create a Service Account:
   - Go to IAM & Admin > Service Accounts
   - Click "Create Service Account"
   - Download the JSON key file
5. Set environment variable:
   ```bash
   export GEE_SERVICE_ACCOUNT_KEY="/path/to/your/service-account-key.json"
   ```

## Option 2: User Authentication

1. Install Earth Engine:
   ```bash
   pip install earthengine-api
   ```

2. Authenticate:
   ```bash
   earthengine authenticate
   ```

3. Initialize (this should work automatically)

## Option 3: Demo Mode (Fallback)

If GEE authentication fails, the system automatically falls back to demo mode with synthetic satellite images.

## Testing Authentication

Run this test script to verify your setup:
```bash
python test_real_change_detection.py
```

The script will show whether you're connected to real GEE or using demo mode.
