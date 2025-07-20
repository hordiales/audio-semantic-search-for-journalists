#!/bin/bash

# UV-compatible startup script for Audio Search MCP Server
# This script ensures proper environment setup and runs the server with uv

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to the MCP server directory
cd "$SCRIPT_DIR"

# Function to check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        echo "❌ uv is not installed. Please install it first:"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

# Function to setup the project with uv
setup_project() {
    if [ -t 1 ]; then
        echo "🔧 Setting up project with uv..."
    fi
    
    # Install dependencies using requirements.txt instead of pyproject.toml
    if [ ! -f ".venv/pyvenv.cfg" ] || [ "requirements_full.txt" -nt ".venv/pyvenv.cfg" ]; then
        if [ -t 1 ]; then
            echo "📦 Installing dependencies with uv..."
        fi
        uv venv --python 3.11 >/dev/null 2>&1
        source .venv/bin/activate
        uv pip install -r requirements_full.txt >/dev/null 2>&1
    fi
}

# Function to run the server
run_server() {
    local dataset_dir="${1:-$PROJECT_ROOT/dataset}"
    
    if [ -t 1 ]; then
        echo "🚀 Starting Audio Search MCP Server with uv"
        echo "📁 Dataset directory: $dataset_dir"
        echo "📁 Project root: $PROJECT_ROOT"
    fi
    
    # Check if dataset exists
    if [ ! -d "$dataset_dir" ]; then
        if [ -t 1 ]; then
            echo "❌ Dataset directory not found: $dataset_dir"
            echo "Please ensure the dataset is available before starting the MCP server."
        fi
        exit 1
    fi
    
    # Activate virtual environment and run the server
    source .venv/bin/activate
    # Set environment variable to suppress verbose output in MCP mode
    export MCP_MODE=1
    exec python start_uv.py --dataset-dir "$dataset_dir"
}

# Main execution
main() {
    # Only show output if not being called by Claude Desktop
    if [ -t 1 ]; then
        echo "🎵 Audio Search MCP Server (UV Runner)"
        echo "======================================"
    fi
    
    check_uv
    setup_project
    
    # Parse arguments
    dataset_dir=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset-dir)
                dataset_dir="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [--dataset-dir PATH]"
                echo ""
                echo "Options:"
                echo "  --dataset-dir PATH    Path to dataset directory (default: ../dataset)"
                echo "  -h, --help           Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use -h or --help for usage information"
                exit 1
                ;;
        esac
    done
    
    run_server "$dataset_dir"
}

# Run main function with all arguments
main "$@"