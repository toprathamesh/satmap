#!/usr/bin/env python3
"""
WSGI entry point for the Change Detection Platform
"""

import os
from app import create_app

# Create Flask application
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 