#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Start the Flask app using Gunicorn
# Replace 'main:app' if your app object is named differently or in another file
exec gunicorn app:app --workers 4 --bind 0.0.0.0:8000
