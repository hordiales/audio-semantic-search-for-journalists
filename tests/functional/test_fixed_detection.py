#!/usr/bin/env python3
"""
Test the fixed YAMNet detection on a few segments
"""

import pandas as pd
import sys
from pathlib import Path
import logging

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import ensure_sys_path, SRC_ROOT

ensure_sys_path([SRC_ROOT])

from detect_audio_events import AudioEventDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_segment_detection():
    """Test detection on individual segments"""
    
    # Load dataset 
    df = pd.read_pickle("dataset/final/complete_dataset.pkl")
    logger.info(f"Dataset loaded: {len(df)} segments")
    
    # Test on first few segments with different time ranges
    detector = AudioEventDetector()
    
    test_segments = [0, 1, 2, 50, 100]  # Test different segments
    
    for idx in test_segments:
        if idx >= len(df):
            continue
            
        row = df.iloc[idx]
        logger.info(f"\n=== Testing Segment {idx} ===")
        logger.info(f"File: {row['source_file']}")
        logger.info(f"Time: {row['start_time']:.1f}s - {row['end_time']:.1f}s ({row['duration']:.1f}s)")
        logger.info(f"Text: {row['text'][:100]}...")
        
        try:
            # Test the fixed detection
            events_result = detector.detect_events_in_audio(
                row['source_file'], 
                row['start_time'], 
                row['end_time']
            )
            
            if events_result['processing_success']:
                logger.info(f"Processing successful:")
                logger.info(f"  Duration: {events_result['audio_duration']:.2f}s")
                logger.info(f"  Frames: {events_result['total_frames']}")
                logger.info(f"  Max confidence: {events_result['max_confidence']:.4f}")
                
                detected_events = []
                events_detected = events_result['events_detected']
                for event_name, event_data in events_detected.items():
                    if event_data.get('detected'):
                        detected_events.append(f"{event_name}({event_data.get('max_confidence', 0):.3f})")
                
                if detected_events:
                    logger.info(f"  Detected: {', '.join(detected_events)}")
                else:
                    logger.info(f"  No events detected")
            else:
                logger.error(f"Processing failed: {events_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error testing segment {idx}: {e}")

if __name__ == "__main__":
    print("🔍 Testing Fixed YAMNet Detection")
    print("=" * 50)
    test_segment_detection()
