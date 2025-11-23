#!/bin/bash
# Full installation script for macOS/Linux with compatibility checks

set -e  # Exit on any error

echo "🚀 Full Install - Sistema de Búsqueda Semántica"
echo "==============================================="

# Check if we're in the right directory
if [ ! -f "requirements-minimal.txt" ]; then
    echo "❌ Error: Execute this script from the project directory"
    exit 1
fi

# Check Python version compatibility
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $python_version"

# Check if Python version is compatible with TensorFlow
if python -c "import sys; major, minor = sys.version_info[:2]; exit(0 if (major == 3 and 9 <= minor <= 12) else 1)" 2>/dev/null; then
    echo "✅ Python version is compatible with TensorFlow"
    TENSORFLOW_COMPATIBLE=true
else
    echo "⚠️  Python $python_version is not compatible with TensorFlow"
    echo "   TensorFlow supports Python 3.9-3.12, you have Python $python_version"
    echo "   Continuing with installation, but TensorFlow will be skipped..."
    TENSORFLOW_COMPATIBLE=false
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install core requirements first
echo "📦 Installing core requirements..."
python -m pip install -r requirements-minimal.txt

# Check if TensorFlow is already installed
echo "🔍 Checking TensorFlow installation..."
if python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} already installed')" 2>/dev/null; then
    echo "✅ TensorFlow is already installed"
    TENSORFLOW_INSTALLED=true
elif [ "$TENSORFLOW_COMPATIBLE" = true ]; then
    echo "📦 Installing TensorFlow and TensorFlow Hub..."
    if python -m pip install tensorflow tensorflow-hub; then
        echo "✅ TensorFlow installed successfully"
        TENSORFLOW_INSTALLED=true
    else
        echo "⚠️  TensorFlow installation failed, continuing without it"
        TENSORFLOW_INSTALLED=false
    fi
else
    echo "⚠️  Skipping TensorFlow installation due to Python version incompatibility"
    TENSORFLOW_INSTALLED=false
fi

# Test basic functionality
echo "🧪 Testing installation..."
TF_CPP_MIN_LOG_LEVEL=3 python -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import whisper
import sentence_transformers
print('✅ Core libraries imported successfully')

# Test Whisper (only import, not loading model to avoid download)
try:
    # Just verify it can be imported
    print('✅ Whisper available')
except Exception as e:
    print(f'⚠️  Whisper test failed: {e}')

# Test sentence-transformers (only import, not loading model to avoid download)
try:
    from sentence_transformers import SentenceTransformer
    print('✅ Sentence Transformers available')
except Exception as e:
    print(f'⚠️  Sentence transformer test failed: {e}')
"

# Test TensorFlow if installed
if [ "$TENSORFLOW_INSTALLED" = true ]; then
    echo "🧪 Testing TensorFlow installation..."
    TF_CPP_MIN_LOG_LEVEL=3 python -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    print('✅ TensorFlow and TensorFlow Hub imported successfully')
    print(f'   TensorFlow version: {tf.__version__}')
except Exception as e:
    print(f'⚠️  TensorFlow test failed: {e}')
"
fi

echo ""
echo "🎉 Installation completed!"
echo ""
echo "📚 Next steps:"
echo "1. Run the web app: streamlit run streamlit_app.py"
echo "2. Or try the command line: ./run_cmd_line.sh"
echo "3. Or try examples: python example_usage.py"
echo ""

if [ "$TENSORFLOW_INSTALLED" = true ]; then
    echo "✅ Full installation with audio embeddings (YAMNet) support"
else
    echo "⚠️  Limited installation - audio embeddings (YAMNet) not available"
    echo "   To enable full features, use Python 3.9-3.12 and run:"
    echo "   pip install tensorflow tensorflow-hub"
fi