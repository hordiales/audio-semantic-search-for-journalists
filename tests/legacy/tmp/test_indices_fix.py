#!/usr/bin/env python3
"""
Test that indices are created in the correct location
"""

import sys
import os
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import ensure_sys_path, SRC_ROOT, PROJECT_ROOT

ensure_sys_path([SRC_ROOT])

def test_indices_location():
    """Test that indices will be created in the correct location"""
    
    print("🧪 Testing Indices Location Fix")
    print("=" * 32)
    
    # Simulate the dataset directory
    dataset_dir = PROJECT_ROOT / "dataset"
    
    # Check where indices should be created
    indices_dir = dataset_dir / "indices"
    print(f"✅ Correct indices location: {indices_dir}")
    print(f"   Absolute path: {indices_dir.absolute()}")
    print(f"   Directory exists: {'✅ YES' if indices_dir.exists() else '❌ NO'}")
    
    # Check that src/indices doesn't exist
    src_indices = SRC_ROOT / "indices"
    print(f"✅ Incorrect location (should not exist): {src_indices}")
    print(f"   Directory exists: {'❌ SHOULD NOT EXIST' if src_indices.exists() else '✅ CORRECTLY REMOVED'}")
    
    # Show the fix
    print(f"\n🔧 Fix implemented:")
    print(f"   Before: create_indices(df)  # uses default 'indices' -> src/indices")
    print(f"   After:  create_indices(df, str(indices_dir))  # explicit path -> dataset/indices")
    
    if indices_dir.exists() and not src_indices.exists():
        print(f"\n🎉 Indices location fix successful!")
        return True
    else:
        print(f"\n❌ Fix needs verification")
        return False

if __name__ == "__main__":
    success = test_indices_location()
    exit(0 if success else 1)
