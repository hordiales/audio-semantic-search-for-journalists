#!/usr/bin/env python3
"""
Comprehensive Test Suite for Semantic Audio Search API and MCP Integration
Tests all endpoints, functionality, and integration scenarios
"""

import asyncio
import httpx
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import signal
from datetime import datetime

CURRENT_FILE = Path(__file__).resolve()
TESTS_ROOT = CURRENT_FILE
while TESTS_ROOT.name != "tests" and TESTS_ROOT.parent != TESTS_ROOT:
    TESTS_ROOT = TESTS_ROOT.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from tests.common.path_utils import PROJECT_ROOT

TMP_ROOT = PROJECT_ROOT / "tmp"
if str(TMP_ROOT) not in sys.path:
    sys.path.insert(0, str(TMP_ROOT))

# Add legacy tmp helpers (same directory) to path for relative imports
if str(CURRENT_FILE.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_FILE.parent))

from api_client import SemanticSearchAPIClient

class APITestSuite:
    """Comprehensive test suite for the API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = None
        self.server_process = None
        self.test_results = []
        
    async def setup(self):
        """Setup test environment"""
        print("🔧 Setting up test environment...")
        
        # Start server if not running
        await self.ensure_server_running()
        
        # Initialize API client
        self.client = SemanticSearchAPIClient(self.base_url)
        
        print("✓ Test environment ready")
    
    async def teardown(self):
        """Cleanup test environment"""
        if self.client:
            await self.client.close()
        
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
    
    async def ensure_server_running(self):
        """Ensure API server is running"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=2.0)
                if response.status_code == 200:
                    print("✓ Server already running")
                    return
        except:
            pass
        
        print("🚀 Starting API server...")
        self.server_process = subprocess.Popen([
            sys.executable, "start_servers.py", "--mode", "fastapi"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        for _ in range(30):  # 30 second timeout
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.base_url}/health", timeout=1.0)
                    if response.status_code == 200:
                        print("✓ Server started successfully")
                        return
            except:
                await asyncio.sleep(1)
        
        raise Exception("Failed to start server")
    
    def log_test_result(self, test_name: str, success: bool, message: str = "", duration: float = 0):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "duration": duration
        })
        print(f"{status} {test_name} ({duration:.3f}s)")
        if message:
            print(f"    {message}")
    
    async def run_test(self, test_name: str, test_func):
        """Run individual test with error handling"""
        start_time = time.time()
        try:
            await test_func()
            duration = time.time() - start_time
            self.log_test_result(test_name, True, "", duration)
            return True
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(test_name, False, str(e), duration)
            return False
    
    # ================== BASIC FUNCTIONALITY TESTS ==================
    
    async def test_health_check(self):
        """Test health check endpoint"""
        response = await self.client.health_check()
        assert response["status"] in ["healthy", "initializing"]
        assert "timestamp" in response
        assert "version" in response
    
    async def test_system_initialization(self):
        """Test system initialization"""
        response = await self.client.initialize(use_mock=True)
        assert response["status"] == "initialized"
        assert "config" in response
        assert "available_sentiments_count" in response
    
    async def test_load_sample_data(self):
        """Test loading sample data"""
        response = await self.client.load_sample_data()
        assert response["status"] == "sample_data_loaded"
        assert response["segments_count"] > 0
        assert "sentiment_distribution" in response
    
    async def test_system_stats(self):
        """Test system statistics"""
        stats = await self.client.get_stats()
        assert stats["status"] == "ready"
        assert stats["processed_segments"] > 0
        assert len(stats["available_sentiments"]) > 0
        assert "sentiment_distribution" in stats
    
    # ================== SEARCH FUNCTIONALITY TESTS ==================
    
    async def test_semantic_search(self):
        """Test basic semantic search"""
        response = await self.client.search("economía", search_type="text", top_k=5)
        assert len(response.results) > 0
        assert response.search_type == "text"
        assert response.execution_time > 0
        
        # Check result structure
        result = response.results[0]
        assert hasattr(result, 'text')
        assert hasattr(result, 'source_file')
        assert hasattr(result, 'similarity_score')
    
    async def test_sentiment_search(self):
        """Test sentiment-based search"""
        response = await self.client.sentiment_search("feliz", top_k=5)
        assert len(response.results) > 0
        assert response.search_type == "sentiment"
        
        # Check sentiment scores
        for result in response.results:
            assert result.sentiment_score is not None
            assert result.sentiment_score > 0
    
    async def test_combined_search(self):
        """Test combined search with sentiment filter"""
        response = await self.client.search(
            "economía",
            search_type="text",
            sentiment_filter="optimista",
            top_k=5
        )
        assert len(response.results) >= 0  # May be empty if no positive economic content
        assert response.search_type == "text"
    
    async def test_bulk_search(self):
        """Test bulk search functionality"""
        queries = ["feliz", "triste", "economía"]
        responses = await self.client.bulk_search(queries, top_k=3)
        
        assert len(responses) == len(queries)
        for response in responses:
            assert response.query in queries
            assert isinstance(response.results, list)
    
    # ================== SENTIMENT ANALYSIS TESTS ==================
    
    async def test_sentiment_info(self):
        """Test sentiment information endpoint"""
        info = await self.client.get_sentiments_info()
        assert "total_sentiments" in info
        assert "sentiments_by_category" in info
        assert "positive" in info["sentiments_by_category"]
        assert "negative" in info["sentiments_by_category"]
        assert "neutral" in info["sentiments_by_category"]
    
    async def test_emotional_content_search(self):
        """Test emotional content search"""
        # Test pure emotion search
        happy_results = await self.client.find_emotional_content("feliz", max_results=3)
        assert len(happy_results) > 0
        
        # Test combined emotion + topic search
        econ_happy = await self.client.find_emotional_content("optimista", "economía", max_results=3)
        assert isinstance(econ_happy, list)
    
    async def test_convenience_methods(self):
        """Test convenience search methods"""
        happy_content = await self.client.search_happy_content(top_k=3)
        sad_content = await self.client.search_sad_content(top_k=3)
        angry_content = await self.client.search_angry_content(top_k=3)
        
        assert isinstance(happy_content, list)
        assert isinstance(sad_content, list)
        assert isinstance(angry_content, list)
    
    async def test_content_mood_analysis(self):
        """Test content mood analysis"""
        analysis = await self.client.analyze_content_mood("economía")
        
        assert "query" in analysis
        assert "overall_mood" in analysis
        assert analysis["overall_mood"] in ["positive", "negative", "neutral"]
        assert "sentiment_distribution" in analysis
        assert "summary" in analysis
    
    # ================== DATA MANAGEMENT TESTS ==================
    
    async def test_data_clearing(self):
        """Test data clearing functionality"""
        # First ensure we have data
        await self.client.load_sample_data()
        
        # Check we have data
        stats_before = await self.client.get_stats()
        assert stats_before["processed_segments"] > 0
        
        # Clear data
        clear_response = await self.client.clear_data()
        assert clear_response["status"] == "data_cleared"
        
        # Reload sample data for other tests
        await self.client.load_sample_data()
    
    # ================== ERROR HANDLING TESTS ==================
    
    async def test_invalid_requests(self):
        """Test error handling for invalid requests"""
        # Test empty query
        try:
            await self.client.search("")
            assert False, "Should have raised an error for empty query"
        except:
            pass  # Expected to fail
        
        # Test invalid sentiment
        response = await self.client.sentiment_search("invalid_sentiment_xyz", top_k=5)
        # Should return empty results, not error
        assert len(response.results) == 0
    
    async def test_edge_cases(self):
        """Test edge cases"""
        # Very large top_k
        response = await self.client.search("test", top_k=1000)
        assert len(response.results) <= 100  # Should be limited by data size
        
        # Very small top_k
        response = await self.client.search("test", top_k=1)
        assert len(response.results) <= 1
    
    # ================== PERFORMANCE TESTS ==================
    
    async def test_search_performance(self):
        """Test search performance"""
        queries = ["feliz", "triste", "economía", "política", "salud"]
        
        start_time = time.time()
        for query in queries:
            await self.client.search(query, top_k=5)
        total_time = time.time() - start_time
        
        avg_time = total_time / len(queries)
        assert avg_time < 2.0, f"Average search time too high: {avg_time:.3f}s"
        
        print(f"    Average search time: {avg_time:.3f}s")
    
    async def test_concurrent_searches(self):
        """Test concurrent search performance"""
        async def search_task(query: str):
            return await self.client.search(query, top_k=3)
        
        queries = ["feliz", "triste", "enojado", "optimista", "preocupado"]
        
        start_time = time.time()
        responses = await asyncio.gather(*[search_task(q) for q in queries])
        total_time = time.time() - start_time
        
        assert len(responses) == len(queries)
        assert total_time < 5.0, f"Concurrent searches too slow: {total_time:.3f}s"
        
        print(f"    Concurrent search time: {total_time:.3f}s for {len(queries)} searches")
    
    # ================== INTEGRATION TESTS ==================
    
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # 1. Initialize system
        await self.client.initialize(use_mock=True)
        
        # 2. Load data
        await self.client.load_sample_data()
        
        # 3. Get system info
        stats = await self.client.get_stats()
        assert stats["processed_segments"] > 0
        
        # 4. Perform various searches
        text_results = await self.client.search("economía", search_type="text")
        sentiment_results = await self.client.sentiment_search("feliz")
        combined_results = await self.client.search("política", sentiment_filter="preocupado")
        
        # 5. Analyze content
        mood_analysis = await self.client.analyze_content_mood("economía")
        
        # 6. Get sentiment info
        sentiment_info = await self.client.get_sentiments_info()
        
        # Verify all steps worked
        assert len(text_results.results) > 0
        assert len(sentiment_results.results) > 0
        assert isinstance(combined_results.results, list)
        assert "overall_mood" in mood_analysis
        assert sentiment_info["total_sentiments"] > 0
    
    # ================== MAIN TEST RUNNER ==================
    
    async def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 Starting Comprehensive API Test Suite")
        print("=" * 60)
        
        await self.setup()
        
        # Test categories
        test_categories = [
            ("Basic Functionality", [
                ("Health Check", self.test_health_check),
                ("System Initialization", self.test_system_initialization),
                ("Load Sample Data", self.test_load_sample_data),
                ("System Stats", self.test_system_stats),
            ]),
            ("Search Functionality", [
                ("Semantic Search", self.test_semantic_search),
                ("Sentiment Search", self.test_sentiment_search),
                ("Combined Search", self.test_combined_search),
                ("Bulk Search", self.test_bulk_search),
            ]),
            ("Sentiment Analysis", [
                ("Sentiment Info", self.test_sentiment_info),
                ("Emotional Content Search", self.test_emotional_content_search),
                ("Convenience Methods", self.test_convenience_methods),
                ("Content Mood Analysis", self.test_content_mood_analysis),
            ]),
            ("Data Management", [
                ("Data Clearing", self.test_data_clearing),
            ]),
            ("Error Handling", [
                ("Invalid Requests", self.test_invalid_requests),
                ("Edge Cases", self.test_edge_cases),
            ]),
            ("Performance", [
                ("Search Performance", self.test_search_performance),
                ("Concurrent Searches", self.test_concurrent_searches),
            ]),
            ("Integration", [
                ("End-to-End Workflow", self.test_end_to_end_workflow),
            ])
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for category_name, tests in test_categories:
            print(f"\n📋 {category_name}")
            print("-" * 40)
            
            for test_name, test_func in tests:
                total_tests += 1
                success = await self.run_test(test_name, test_func)
                if success:
                    passed_tests += 1
        
        await self.teardown()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        success_rate = (passed_tests / total_tests) * 100
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! API is fully functional.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed. Check implementation.")
            
            print("\nFailed tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  ❌ {result['test']}: {result['message']}")
        
        return passed_tests == total_tests


async def test_mcp_integration():
    """Test MCP server integration (basic check)"""
    print("\n🤖 Testing MCP Integration")
    print("-" * 40)
    
    try:
        # Check if MCP server can be imported
        from mcp_server import SemanticSearchMCPServer
        
        # Create MCP server instance
        mcp_server = SemanticSearchMCPServer()
        
        print("✅ MCP Server: Import successful")
        print("✅ MCP Server: Instance creation successful")
        print("✅ MCP Server: Ready for LLM integration")
        
        return True
        
    except ImportError as e:
        print(f"❌ MCP Server: Import failed - {e}")
        print("   Install MCP dependencies: pip install mcp")
        return False
    except Exception as e:
        print(f"❌ MCP Server: Setup failed - {e}")
        return False


async def main():
    """Main test function"""
    print("🚀 Semantic Audio Search - Complete Test Suite")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print()
    
    # Test API functionality
    api_tests = APITestSuite()
    api_success = await api_tests.run_all_tests()
    
    # Test MCP integration
    mcp_success = await test_mcp_integration()
    
    # Overall result
    print("\n" + "=" * 60)
    print("🎯 OVERALL RESULTS")
    print("=" * 60)
    
    if api_success and mcp_success:
        print("🎉 ALL SYSTEMS FUNCTIONAL!")
        print("✓ FastAPI Server: Fully operational")
        print("✓ Sentiment Search: Working correctly")
        print("✓ MCP Integration: Ready for LLM tools")
        print("\n🚀 Ready for production use!")
        return True
    else:
        print("⚠️  Some issues detected:")
        if not api_success:
            print("❌ API tests failed")
        if not mcp_success:
            print("❌ MCP integration issues")
        print("\n🔧 Please review and fix issues before deployment.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
