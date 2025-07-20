#!/bin/bash
# Quick installation script for macOS/Linux

set -e  # Exit on any error

echo "🚀 Quick Install - Sistema de Búsqueda Semántica"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "requirements-minimal.txt" ]; then
    echo "❌ Error: Execute this script from the project directory"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $python_version"

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install minimal requirements
echo "📦 Installing minimal requirements..."
python3 -m pip install -r requirements-minimal.txt

# Test basic functionality
echo "🧪 Testing installation..."
python3 -c "
import whisper
import sentence_transformers
import streamlit
print('✅ Core libraries imported successfully')

# Test Whisper
model = whisper.load_model('tiny')
print('✅ Whisper model loaded')

# Test sentence-transformers
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print('✅ Sentence transformer loaded')
"

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📚 Next steps:"
echo "1. Run the web app: streamlit run streamlit_app.py"
echo "2. Or try examples: python example_usage.py"
echo ""
echo "⚠️  Note: This is a minimal installation."
echo "   For full features (audio embeddings), install:"
echo "   pip install tensorflow tensorflow-hub"