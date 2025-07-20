#!/usr/bin/env python3
"""
Test script for the MCP server to verify it loads correctly
"""

import sys
import os
import asyncio

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from server import AudioSearchMCPServer

async def test_server():
    """Test server initialization"""
    print("🧪 Testing MCP Server initialization...")
    
    server = AudioSearchMCPServer()
    
    # Test initialization
    dataset_dir = "../dataset"
    success = await server.initialize_client(dataset_dir)
    
    if success:
        print("✅ Server initialized successfully!")
        print(f"📊 Dataset loaded: {len(server.client.df)} segments")
        print(f"🎭 Sentiment analysis: {'✅ Enabled' if server.client.sentiment_enabled else '❌ Disabled'}")
        print(f"🎵 Hybrid search: {'✅ Enabled' if hasattr(server.client, 'hybrid_search_enabled') and server.client.hybrid_search_enabled else '❌ Disabled'}")
        
        # Test a simple semantic search
        print("\n🔍 Testing semantic search...")
        try:
            results = await server._handle_semantic_search({"query": "economía", "k": 3})
            print(f"✅ Semantic search test successful: {len(results)} results")
            if results and len(results[0].text) > 100:
                print(f"📝 Sample result: {results[0].text[:100]}...")
        except Exception as e:
            print(f"❌ Semantic search test failed: {e}")
        
        return True
    else:
        print("❌ Server initialization failed!")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_server())
    sys.exit(0 if success else 1)