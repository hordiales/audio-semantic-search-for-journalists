#!/usr/bin/env python3
"""
UV-compatible startup script for Audio Search MCP Server
"""

import sys
import os
import subprocess
from pathlib import Path
import logging

# Configure logging to stderr
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stderr)

def get_project_root():
    """Get the project root directory"""
    current_dir = Path(__file__).parent
    # Go up one level to reach the main project directory
    return current_dir.parent

def setup_environment():
    """Setup the environment for running with uv"""
    project_root = get_project_root()
    src_dir = project_root / "src"
    
    # Add src directory to Python path
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(src_dir)
    
    return project_root

def run_with_uv():
    """Run the MCP server using uv"""
    project_root = setup_environment()
    dataset_dir = project_root / "dataset"
    
    # Check if dataset exists
    if not dataset_dir.exists():
        logging.error(f"❌ Dataset directory not found: {dataset_dir}")
        logging.error("Please ensure the dataset is available before starting the MCP server.")
        sys.exit(1)
    
    # Import and run the server
    try:
        from server import AudioSearchMCPServer
        
        server = AudioSearchMCPServer()
        server.run(str(dataset_dir))
        
    except ImportError as e:
        logging.error(f"❌ Import error: {e}")
        logging.error("Please ensure all dependencies are installed with: uv sync")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Error starting server: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Audio Search MCP Server with UV")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Path to the dataset directory (default: ../dataset)"
    )
    
    args = parser.parse_args()
    
    # Setup environment
    project_root = setup_environment()
    
    # Determine dataset directory
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        dataset_dir = project_root / "dataset"
    
    # Check if dataset exists
    if not dataset_dir.exists():
        logging.error(f"❌ Dataset directory not found: {dataset_dir}")
        logging.error("Please ensure the dataset is available before starting the MCP server.")
        sys.exit(1)
    
    # Import and run the server
    try:
        from server import AudioSearchMCPServer
        
        # Only print status if running in terminal (not MCP mode)
        if sys.stdout.isatty():
            logging.info(f"🚀 Starting Audio Search MCP Server with UV")
            logging.info(f"📁 Dataset directory: {dataset_dir}")
            logging.info(f"🐍 Python path: {sys.path[0]}")
        
        # Redirect stdout during initialization to avoid JSON parse errors
        if not sys.stdout.isatty():
            # Save original stdout
            original_stdout = sys.stdout
            # Redirect stdout to stderr during initialization
            sys.stdout = sys.stderr
            
            server = AudioSearchMCPServer()
            
            # Restore stdout for MCP communication
            sys.stdout = original_stdout
            
            server.run(str(dataset_dir))
        else:
            server = AudioSearchMCPServer()
            server.run(str(dataset_dir))
        
    except ImportError as e:
        logging.error(f"❌ Import error: {e}")
        logging.error("Please ensure all dependencies are installed with: uv sync")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()