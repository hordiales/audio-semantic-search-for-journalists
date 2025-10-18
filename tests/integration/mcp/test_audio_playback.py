#!/usr/bin/env python3
"""
Test script for audio playback functionality
"""

import sys
import os
import asyncio
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import ensure_sys_path, SRC_ROOT, PROJECT_ROOT

ensure_sys_path([SRC_ROOT])

from server import AudioSearchMCPServer

async def test_audio_playback():
    """Test the audio playback functionality"""
    print("🧪 Testing Audio Playback Functionality")
    print("=" * 40)
    
    # Initialize the server
    server = AudioSearchMCPServer()
    dataset_dir = str(PROJECT_ROOT / "dataset")
    
    if not await server.initialize_client(dataset_dir):
        print("❌ Failed to initialize server")
        return False
    
    print("✅ Server initialized successfully")
    
    # Test OS detection
    player = server._get_audio_player_command()
    print(f"🔧 Detected audio player: {player}")
    
    if not player:
        print("❌ No audio player found")
        return False
    
    # Get a sample audio file from the dataset
    df = server.client.df
    if df.empty:
        print("❌ No audio segments in dataset")
        return False
    
    # Get the first segment
    sample = df.iloc[0]
    source_file = sample['source_file']
    start_time = sample['start_time']
    end_time = sample['end_time']
    
    print(f"🎵 Testing with sample segment:")
    print(f"   File: {source_file}")
    print(f"   Time: {start_time:.1f}s - {end_time:.1f}s")
    
    # Test the playback function
    args = {
        "source_file": source_file,
        "start_time": start_time,
        "end_time": end_time,
        "segment_index": 0
    }
    
    try:
        result = await server._handle_play_audio_segment(args)
        print("✅ Audio playback test completed")
        print("📝 Result:")
        for content in result:
            print(content.text)
        return True
    except Exception as e:
        print(f"❌ Audio playback test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_audio_playback())
    sys.exit(0 if success else 1)
