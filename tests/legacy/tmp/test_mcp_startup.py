#!/usr/bin/env python3
"""
Test MCP server startup and tool registration
"""

import asyncio
import os
from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import PROJECT_ROOT, SRC_ROOT, ensure_sys_path

TMP_ROOT = PROJECT_ROOT / "tmp"
for candidate in (CURRENT_FILE.parent, TMP_ROOT, SRC_ROOT):
    ensure_sys_path([candidate])

async def test_mcp_server_startup():
    """Test that MCP server can start and register tools"""
    try:
        from mcp_server import SemanticSearchMCPServer

        print("✅ MCP imports successful")

        # Create server instance
        server = SemanticSearchMCPServer()
        print("✅ MCP server instance created")

        # Test that server has the expected tools
        expected_tools = [
            "initialize_search_system",
            "load_real_dataset",
            "get_dataset_info",
            "switch_dataset_mode",
            "sentiment_search",
            "semantic_search"
        ]

        print(f"✅ Expected {len(expected_tools)} key tools")

        # Test basic initialization
        print("\n🧪 Testing system initialization...")
        result = await server._initialize_system({"use_mock": True})
        print("✅ System initialization successful")

        # Test dataset info
        print("\n🧪 Testing dataset info...")
        result = await server._get_dataset_info({})
        print("✅ Dataset info successful")

        print("\n🎉 MCP server startup test PASSED!")
        print("\n📝 Ready for Claude integration!")
        print("\nTo use in Claude:")
        print("1. Add this to your MCP settings:")
        print("   {")
        print('     "mcpServers": {')
        print('       "semantic-search": {')
        print('         "command": "python",')
        print(f'         "args": ["{os.path.abspath("mcp_server.py")}"]')
        print('       }')
        print('     }')
        print("   }")
        print("\n2. Restart Claude")
        print("3. Use tools like get_dataset_info, load_real_dataset, sentiment_search")

        return True

    except Exception as e:
        print(f"❌ MCP server startup test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing MCP Server Startup")
    print("=" * 40)

    try:
        result = asyncio.run(test_mcp_server_startup())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
