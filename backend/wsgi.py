#!/usr/bin/env python3
"""
WSGI entry point for production deployment
"""

import os
import sys
from app import create_app

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Create the Flask application
app = create_app()

if __name__ == "__main__":
    # For development only
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port) 