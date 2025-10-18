#!/usr/bin/env python3
"""
Test script to validate that the music search fix works
"""

import sys
import pandas as pd
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import ensure_sys_path, SRC_ROOT

ensure_sys_path([SRC_ROOT])

from improved_audio_search import ImprovedAudioSearch

def test_music_keywords():
    """Test that music-related keywords are properly mapped"""
    search_engine = ImprovedAudioSearch()
    
    print("🎵 Testing music keyword mapping...")
    
    # Check if music keywords are properly loaded
    music_keywords = search_engine.get_keywords_for_class('music')
    print(f"Music keywords: {music_keywords}")
    
    # Check if English terms are included
    english_terms = ['music', 'song', 'songs']
    for term in english_terms:
        if term in music_keywords:
            print(f"✅ '{term}' found in music keywords")
        else:
            print(f"❌ '{term}' NOT found in music keywords")
    
    # Test search logic with mock data
    print("\n🔍 Testing search logic...")
    
    # Create mock dataframe
    mock_data = [
        {"text": "La música de fondo era hermosa", "source_file": "test1.wav", "start_time": 0, "end_time": 10, "duration": 10},
        {"text": "I love this music so much", "source_file": "test2.wav", "start_time": 0, "end_time": 10, "duration": 10},
        {"text": "The song was playing loudly", "source_file": "test3.wav", "start_time": 0, "end_time": 10, "duration": 10},
        {"text": "No audio content here", "source_file": "test4.wav", "start_time": 0, "end_time": 10, "duration": 10}
    ]
    
    mock_df = pd.DataFrame(mock_data)
    
    # Test search for "music"
    results = search_engine.search_audio_by_text(mock_df, "music", k=5)
    
    print(f"Found {len(results)} results for 'music' query:")
    for result in results:
        print(f"  - Score: {result['score']:.2f} | Text: {result['text']}")
        print(f"    Matched keywords: {result.get('matched_keywords', [])}")
    
    if len(results) > 0:
        print("✅ Search for 'music' now returns results!")
    else:
        print("❌ Search for 'music' still returns no results")
    
    return len(results) > 0

if __name__ == "__main__":
    success = test_music_keywords()
    if success:
        print("\n🎉 Fix validated successfully!")
        sys.exit(0)
    else:
        print("\n❌ Fix validation failed!")
        sys.exit(1)
