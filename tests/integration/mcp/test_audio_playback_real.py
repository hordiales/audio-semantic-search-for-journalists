#!/usr/bin/env python3
"""
Test real audio playback with the exact parameters that caused the error
"""

import asyncio
from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import PROJECT_ROOT, SRC_ROOT, ensure_sys_path

ensure_sys_path([SRC_ROOT])

from server import AudioSearchMCPServer


async def test_real_playback():
    """Test with the exact parameters that caused the error"""
    print("🧪 Testing Real Audio Playback")
    print("=" * 30)

    # Initialize the server
    server = AudioSearchMCPServer()
    server.dataset_dir = Path(PROJECT_ROOT / "dataset")

    # Test parameters from the error log
    args = {
        "source_file": "dataset/converted/Ernesto Tenembaum sobre el vínculo entre la empresa Tech Security SRL con el Banco Nación. [68HlGgpKLRA].wav",
        "start_time": 211.6,
        "end_time": 214.3
    }

    print("📝 Testing with parameters:")
    print(f"   source_file: {args['source_file']}")
    print(f"   start_time: {args['start_time']}")
    print(f"   end_time: {args['end_time']}")

    try:
        # Extract filename
        source_filename = Path(args['source_file']).name
        print(f"✅ Extracted filename: {source_filename}")

        # Test path operations
        possible_paths = [
            server.dataset_dir / "converted" / source_filename,
            server.dataset_dir / "audio" / source_filename,
            server.dataset_dir.parent / "data" / source_filename,
        ]

        print("✅ Generated paths:")
        for path in possible_paths:
            exists = path.exists()
            print(f"   {path} -> {'EXISTS' if exists else 'NOT FOUND'}")

        # Test the actual function
        result = await server._handle_play_audio_segment(args)
        print("✅ Function executed successfully!")
        print("📄 Result:")
        for content in result:
            print(content.text[:200] + "..." if len(content.text) > 200 else content.text)

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_real_playback())
    print(f"\n{'🎉 SUCCESS!' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)
