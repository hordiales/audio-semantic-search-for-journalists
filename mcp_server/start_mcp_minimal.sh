#!/bin/bash

# Minimal MCP server startup script for testing
# This version uses only essential dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$SCRIPT_DIR"

echo "🧪 Starting Minimal MCP Server for Testing"
echo "=========================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Setup minimal environment
if [ ! -f ".venv_minimal/pyvenv.cfg" ]; then
    echo "📦 Setting up minimal environment..."
    uv venv .venv_minimal --python 3.11
    source .venv_minimal/bin/activate
    uv pip install -r requirements_minimal.txt
else
    source .venv_minimal/bin/activate
fi

# Parse arguments
dataset_dir="${1:-$PROJECT_ROOT/dataset}"

echo "🚀 Starting minimal MCP server"
echo "📁 Dataset directory: $dataset_dir"

# Simple test - just check if MCP is working
python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT/src')

try:
    from mcp.server import Server
    print('✅ MCP import successful')
    
    # Test basic dataset loading
    import pandas as pd
    import os
    dataset_file = '$dataset_dir/final/complete_dataset.pkl'
    if os.path.exists(dataset_file):
        df = pd.read_pickle(dataset_file)
        print(f'✅ Dataset loaded: {len(df)} segments')
    else:
        print(f'❌ Dataset not found: {dataset_file}')
        
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"

echo "🎉 Minimal test completed successfully!"
echo ""
echo "Now you can restart Claude Desktop and the full MCP server should work."
echo "The full server might take 1-2 minutes to load all ML models on first run."