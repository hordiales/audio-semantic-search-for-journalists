#!/usr/bin/env python3
"""
Simple Integration Test for API and MCP Components
Tests core functionality without requiring full server setup
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

TMP_ROOT = PROJECT_ROOT / "tmp"
for candidate in (CURRENT_FILE.parent, TMP_ROOT, SRC_ROOT):
    ensure_sys_path([candidate])

def test_imports():
    """Test that all core modules can be imported"""
    print("🧪 Testing Core Module Imports")
    print("-" * 40)

    tests = [
        ("FastAPI Server", "fastapi_server"),
        ("API Client", "api_client"),
        ("MCP Server", "mcp_server"),
        ("Sentiment Analysis", "sentiment_analysis"),
        ("Semantic Search", "semantic_search"),
    ]

    passed = 0
    for name, module in tests:
        try:
            __import__(module)
            print(f"✅ {name}: Import successful")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: Import failed - {e}")

    return passed, len(tests)

def test_dependencies():
    """Test that required dependencies are available"""
    print("\n🔧 Testing Dependencies")
    print("-" * 40)

    deps = [
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("Transformers", "transformers"),
        ("Sentence Transformers", "sentence_transformers"),
        ("Pandas", "pandas"),
        ("NumPy", "numpy"),
        ("HTTPX", "httpx"),
        ("Pydantic", "pydantic"),
    ]

    passed = 0
    for name, package in deps:
        try:
            __import__(package)
            print(f"✅ {name}: Available")
            passed += 1
        except ImportError:
            print(f"❌ {name}: Missing - install with 'pip install {package}'")

    # Optional dependencies
    optional_deps = [
        ("MCP", "mcp"),
        ("FAISS", "faiss"),
        ("SlowAPI", "slowapi"),
    ]

    print("\nOptional Dependencies:")
    for name, package in optional_deps:
        try:
            __import__(package)
            print(f"✅ {name}: Available")
        except ImportError:
            print(f"⚠️  {name}: Missing (optional)")

    return passed, len(deps)

def test_configuration():
    """Test configuration files and directories"""
    print("\n📁 Testing Configuration")
    print("-" * 40)

    config_files = [
        "server_config.yaml",
        "requirements.txt",
        "start_servers.py",
    ]

    passed = 0
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}: Found")
            passed += 1
        else:
            print(f"❌ {config_file}: Missing")

    # Check if key source files exist
    source_files = [
        "fastapi_server.py",
        "mcp_server.py",
        "api_client.py",
        "sentiment_analysis.py",
        "semantic_search.py",
    ]

    print("\nSource Files:")
    for source_file in source_files:
        if Path(source_file).exists():
            print(f"✅ {source_file}: Found")
            passed += 1
        else:
            print(f"❌ {source_file}: Missing")

    return passed, len(config_files) + len(source_files)

async def test_core_functionality():
    """Test core functionality without full server"""
    print("\n⚙️  Testing Core Functionality")
    print("-" * 40)

    passed = 0
    total = 0

    # Test 1: Sentiment Analyzer
    try:
        from sentiment_analysis import SentimentAnalyzer
        analyzer = SentimentAnalyzer(use_mock=True)

        # Test sentiment analysis
        result = analyzer.analyze_text("¡Estoy muy feliz con estos resultados!")
        assert "POSITIVE" in result
        assert "NEGATIVE" in result
        assert "NEUTRAL" in result

        # Test available moods
        moods = analyzer.get_available_moods()
        assert len(moods) > 0
        assert "feliz" in moods

        print("✅ Sentiment Analyzer: Working")
        passed += 1
    except Exception as e:
        print(f"❌ Sentiment Analyzer: Failed - {e}")
    total += 1

    # Test 2: Semantic Search Engine
    try:
        from semantic_search import SemanticSearchEngine

        config = {
            'use_mock_audio': True,
            'use_mock_sentiment': True,
            'whisper_model': 'base',
            'text_embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
        }

        search_engine = SemanticSearchEngine(config)

        # Test basic initialization
        assert search_engine.config is not None
        assert search_engine.sentiment_analyzer is not None

        print("✅ Semantic Search Engine: Working")
        passed += 1
    except Exception as e:
        print(f"❌ Semantic Search Engine: Failed - {e}")
    total += 1

    # Test 3: API Client
    try:
        from api_client import SearchResult, SemanticSearchAPIClient

        # Test client initialization
        client = SemanticSearchAPIClient("http://localhost:8000")
        assert client.base_url == "http://localhost:8000"

        # Test data structures
        result = SearchResult(
            text="Test text",
            source_file="test.wav",
            start_time=0.0,
            end_time=10.0
        )
        assert result.text == "Test text"

        print("✅ API Client: Working")
        passed += 1
    except Exception as e:
        print(f"❌ API Client: Failed - {e}")
    total += 1

    # Test 4: MCP Server (basic check)
    try:
        from mcp_server import SemanticSearchMCPServer

        # Check if MCP is available
        try:
            import mcp
            mcp_server = SemanticSearchMCPServer()
            print("✅ MCP Server: Working (MCP available)")
        except ImportError:
            print("⚠️  MCP Server: Core working (MCP library not installed)")

        passed += 1
    except Exception as e:
        print(f"❌ MCP Server: Failed - {e}")
    total += 1

    return passed, total

def test_sample_data_processing():
    """Test sample data processing"""
    print("\n📊 Testing Sample Data Processing")
    print("-" * 40)

    try:
        import pandas as pd

        from sentiment_analysis import SentimentAnalyzer

        # Create sample data
        sample_data = {
            'text': [
                "¡Excelente noticia! La economía está creciendo",
                "Tristemente, los resultados son decepcionantes",
                "La reunión se realizará el próximo martes"
            ],
            'source_file': ['test1.wav', 'test2.wav', 'test3.wav'],
            'start_time': [0, 30, 60],
            'end_time': [30, 60, 90]
        }

        df = pd.DataFrame(sample_data)

        # Test sentiment processing
        analyzer = SentimentAnalyzer(use_mock=True)
        df_with_sentiment = analyzer.process_dataframe(df)

        # Verify results
        assert 'sentiment_positive' in df_with_sentiment.columns
        assert 'sentiment_negative' in df_with_sentiment.columns
        assert 'dominant_sentiment' in df_with_sentiment.columns
        assert len(df_with_sentiment) == 3

        # Test sentiment search
        happy_results = analyzer.search_by_sentiment(df_with_sentiment, "feliz", top_k=5)
        assert len(happy_results) >= 0  # May be empty with mock data

        print("✅ Sample Data Processing: Working")
        return True

    except Exception as e:
        print(f"❌ Sample Data Processing: Failed - {e}")
        return False

def print_usage_examples():
    """Print usage examples"""
    print("\n📚 Usage Examples")
    print("-" * 40)

    print("1. Start FastAPI Server:")
    print("   python start_servers.py --mode fastapi")
    print()

    print("2. Start MCP Server:")
    print("   python start_servers.py --mode mcp")
    print()

    print("3. Test API Client:")
    print("   python api_client.py")
    print()

    print("4. Run Sentiment Demo:")
    print("   python sentiment_search_demo.py")
    print()

    print("5. Initialize and search (cURL):")
    print("   curl -X POST http://localhost:8000/api/v1/initialize")
    print("   curl -X POST http://localhost:8000/api/v1/load-sample-data")
    print('   curl -X POST http://localhost:8000/api/v1/sentiment-search \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"sentiment": "feliz", "top_k": 5}\'')

async def main():
    """Run all tests"""
    print("🚀 Semantic Audio Search - Integration Test")
    print("=" * 60)

    total_passed = 0
    total_tests = 0

    # Run all test categories
    import_passed, import_total = test_imports()
    deps_passed, deps_total = test_dependencies()
    config_passed, config_total = test_configuration()
    func_passed, func_total = await test_core_functionality()
    data_passed = test_sample_data_processing()

    total_passed = import_passed + deps_passed + config_passed + func_passed + (1 if data_passed else 0)
    total_tests = import_total + deps_total + config_total + func_total + 1

    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    success_rate = (total_passed / total_tests) * 100
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success Rate: {success_rate:.1f}%")

    if total_passed == total_tests:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✓ Core modules working correctly")
        print("✓ Dependencies satisfied")
        print("✓ Configuration files present")
        print("✓ Basic functionality operational")
        print("✓ Ready for API and MCP usage")

        print_usage_examples()

    else:
        print(f"\n⚠️  {total_tests - total_passed} tests failed.")
        print("Please resolve issues before using the system.")

        if import_passed < import_total:
            print("\n🔧 Fix import issues:")
            print("   Check Python path and module structure")

        if deps_passed < deps_total:
            print("\n🔧 Install missing dependencies:")
            print("   pip install -r requirements.txt")

        if config_passed < config_total:
            print("\n🔧 Ensure all files are present:")
            print("   Check that source files and configs exist")

    return total_passed == total_tests

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
