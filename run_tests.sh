#!/bin/bash

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt

# Run environment test
python src/test_environment.py

# Run minimal implementation test
python src/test_minimal.py 