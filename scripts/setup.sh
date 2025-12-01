#!/bin/bash

echo "================================================"
echo "Drug Recommendation System - Setup Script"
echo "================================================"

# Create directories
echo -e "\n[1/5] Creating directories..."
mkdir -p data models database drug_recommendation_system

# Check Python version
echo -e "\n[2/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo -e "\n[3/5] Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo -e "\n[4/5] Downloading NLTK data..."
python3 << EOF
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
print("✓ NLTK data downloaded")
EOF

# Instructions for dataset
echo -e "\n[5/5] Setup complete!"
echo ""
echo "================================================"
echo "NEXT STEPS:"
echo "================================================"
echo ""
echo "1. Download the dataset:"
echo "   URL: https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com"
echo "   Place files in data/ directory:"
echo "   - data/drugsComTrain_raw.tsv"
echo "   - data/drugsComTest_raw.tsv"
echo ""
echo "2. Train the models:"
echo "   python train_models.py"
echo ""
echo "3. Initialize Reflex:"
echo "   reflex init"
echo ""
echo "4. Run the application:"
echo "   reflex run"
echo ""
echo "================================================"
