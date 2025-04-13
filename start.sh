#!/bin/bash
# Step 1: Run feature extraction
echo "Starting feature extraction..."
python offline.py

# Step 2: Start Flask server with Gunicorn
echo "Starting web server..."
gunicorn --workers=1 --threads=1 --timeout=120 server:app
