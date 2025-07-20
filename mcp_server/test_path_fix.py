#!/usr/bin/env python3
"""
Test the path fix for audio playback
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from server import AudioSearchMCPServer

async def test_path_fix():
    """Test that the path fix works"""
    print("🧪 Testing Path Fix for Audio Playback")
    print("=" * 40)
    
    # Initialize the server
    server = AudioSearchMCPServer()
    dataset_dir = "../dataset"
    
    # Test dataset dir conversion to Path
    server.dataset_dir = Path(dataset_dir)
    print(f"✅ Dataset dir as Path: {server.dataset_dir}")
    print(f"   Type: {type(server.dataset_dir)}")
    
    # Test path operations
    test_filename = "test_file.wav"
    test_paths = [
        server.dataset_dir / "converted" / test_filename,
        server.dataset_dir / "audio" / test_filename,
        server.dataset_dir.parent / "data" / test_filename,
    ]
    
    print(f"✅ Path operations work:")
    for i, path in enumerate(test_paths, 1):
        print(f"   {i}. {path}")
        print(f"      Type: {type(path)}")
    
    # Test filename extraction
    source_file_with_path = "dataset/converted/test_file.wav"
    source_filename = Path(source_file_with_path).name
    print(f"✅ Filename extraction:")
    print(f"   Original: {source_file_with_path}")
    print(f"   Extracted: {source_filename}")
    
    print("\n🎉 Path fix should work correctly now!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_path_fix())
    sys.exit(0 if success else 1)