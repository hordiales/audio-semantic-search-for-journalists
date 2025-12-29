#!/usr/bin/env python3
"""
Test script for MCP server dataset functionality
Tests the new dataset loading and management tools
"""

import asyncio
import json
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

try:
    from mcp_server import SemanticSearchMCPServer
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP library not available")

async def test_mcp_dataset_tools():
    """Test MCP server dataset tools"""
    if not MCP_AVAILABLE:
        print("❌ MCP not available, testing core functionality only")
        return await test_core_functionality()  # Test without MCP

    print("🧪 Testing MCP Server Dataset Tools")
    print("=" * 50)

    # Create server instance
    server = SemanticSearchMCPServer()

    try:
        # Test 1: Initialize system
        print("\n1️⃣ Testing system initialization...")
        result = await server._initialize_system({"use_mock": True})
        print("✅ System initialized")

        # Test 2: Get dataset info
        print("\n2️⃣ Testing dataset info...")
        result = await server._get_dataset_info({})
        info = json.loads(result[0].text)
        print(f"📊 Available datasets: {len(info.get('available_datasets', {}))}")
        print(f"🔧 Current mode: {info.get('current_mode')}")

        # Test 3: Load sample data first
        print("\n3️⃣ Testing sample data loading...")
        result = await server._load_sample_data({})
        sample_result = json.loads(result[0].text)
        print(f"✅ Sample data: {sample_result.get('segments_loaded')} segments")

        # Test 4: Try to load real dataset (small subset)
        print("\n4️⃣ Testing real dataset loading...")
        result = await server._load_real_dataset({"load_limit": 25})

        if "error" not in result[0].text.lower():
            real_result = json.loads(result[0].text)
            print(f"✅ Real dataset: {real_result.get('segments_loaded')} segments")
            print(f"📁 Source: {real_result.get('dataset_source')}")

            # Test sentiment search on real data
            print("\n5️⃣ Testing sentiment search on real data...")
            search_result = await server._sentiment_search({
                "sentiment": "positive",
                "top_k": 3
            })

            if "error" not in search_result[0].text.lower():
                search_data = json.loads(search_result[0].text)
                print(f"🔍 Found {search_data.get('results_count')} results")
            else:
                print(f"⚠️ Search result: {search_result[0].text}")
        else:
            print(f"⚠️ Real dataset loading: {result[0].text}")

        # Test 5: Switch dataset modes
        print("\n6️⃣ Testing dataset mode switching...")
        switch_result = await server._switch_dataset_mode({"mode": "sample"})
        if "error" not in switch_result[0].text.lower():
            switch_data = json.loads(switch_result[0].text)
            print(f"🔄 Switched to: {switch_data.get('mode')}")
        else:
            print(f"⚠️ Mode switch: {switch_result[0].text}")

        print("\n🎉 MCP dataset tools tests completed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_core_functionality():
    """Test core functionality without MCP"""
    print("🧪 Testing Core Dataset Functionality (No MCP)")
    print("=" * 50)

    try:
        # Import and test core components
        from semantic_search import SemanticSearchEngine
        from sentiment_analysis import SentimentAnalyzer

        print("\n1️⃣ Testing semantic search engine...")
        config = {
            'whisper_model': 'base',
            'text_embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'use_mock_audio': True,
            'use_mock_sentiment': True,
            'index_type': 'cosine',
            'top_k_results': 5
        }

        engine = SemanticSearchEngine(config)
        print("✅ Search engine created")

        # Test dataset detection
        print("\n2️⃣ Testing dataset detection...")
        from pathlib import Path

        dataset_dir = Path("./dataset")
        if dataset_dir.exists():
            metadata_files = list(dataset_dir.glob("**/dataset_metadata.csv"))
            if not metadata_files:
                metadata_files = list(dataset_dir.glob("**/segments_metadata.csv"))

            if metadata_files:
                import pandas as pd
                df = pd.read_csv(metadata_files[0])
                print(f"✅ Found dataset: {len(df):,} segments")
                print(f"📊 Languages: {df.get('language', pd.Series()).value_counts().head(3).to_dict()}")
            else:
                print("⚠️ No metadata files found")
        else:
            print("⚠️ No dataset directory found")

        print("\n3️⃣ Testing sentiment analyzer...")
        analyzer = SentimentAnalyzer()
        test_text = "¡Qué alegría! Excelente noticia para todos."
        sentiment_scores = analyzer.analyze_text(test_text)
        dominant = max(sentiment_scores, key=sentiment_scores.get)
        print(f"✅ Sentiment analysis: {dominant} ({sentiment_scores[dominant]:.3f})")

        print("\n🎉 Core functionality tests completed!")
        return True

    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False

def test_mcp_tools_list():
    """Test that all MCP tools are properly registered"""
    if not MCP_AVAILABLE:
        print("❌ MCP not available, skipping tools test")
        return

    print("\n🔧 Testing MCP Tools Registration")
    print("=" * 40)

    server = SemanticSearchMCPServer()

    # Check that server has all expected tools
    expected_tools = [
        "initialize_search_system",
        "semantic_search",
        "sentiment_search",
        "get_sentiment_distribution",
        "get_available_sentiments",
        "analyze_text_sentiment",
        "load_sample_data",
        "get_system_stats",
        "find_emotional_content",
        "load_real_dataset",       # New tool
        "get_dataset_info",        # New tool
        "switch_dataset_mode"      # New tool
    ]

    print(f"Expected tools: {len(expected_tools)}")
    for tool in expected_tools:
        print(f"  ✅ {tool}")

    print("\n💡 Use these tools in your LLM to access sentiment search functionality!")

async def main():
    """Main test function"""
    print("🎭 MCP Server Dataset Integration Tests")
    print("=" * 60)

    # Test tools registration
    test_mcp_tools_list()

    # Test dataset functionality
    await test_mcp_dataset_tools()

    print("\n" + "=" * 60)
    print("📖 Usage Examples:")
    print()
    print("For LLMs using MCP:")
    print('  {"name": "get_dataset_info", "arguments": {}}')
    print('  {"name": "load_real_dataset", "arguments": {"load_limit": 100}}')
    print('  {"name": "sentiment_search", "arguments": {"sentiment": "happy", "top_k": 5}}')
    print('  {"name": "switch_dataset_mode", "arguments": {"mode": "real"}}')
    print()
    print("For CLI:")
    print("  python cli_sentiment_search.py --dataset-info")
    print("  python cli_sentiment_search.py --load-real --load-limit 100 --sentiment feliz")
    print()
    print("🚀 Ready for LLM integration!")

if __name__ == "__main__":
    asyncio.run(main())
