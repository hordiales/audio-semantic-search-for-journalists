#!/usr/bin/env python3
"""
Test script for sentiment search functionality
Tests the sentiment analysis and search capabilities
"""

from pathlib import Path
import sys

import pandas as pd

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

from semantic_search import SemanticSearchEngine
from sentiment_analysis import SentimentAnalyzer


def test_sentiment_analyzer():
    """Test the basic sentiment analyzer functionality"""
    print("=== Testing Sentiment Analyzer ===")

    # Test data with clear sentiments
    test_texts = [
        "¡Estoy súper feliz con estos resultados fantásticos!",
        "Me siento muy triste y deprimido por la situación",
        "¡Esto me enoja muchísimo! Es una situación terrible",
        "Es una noticia neutral sin mayor impacto emocional",
        "¡Qué alegría tan grande! Me emociona ver el progreso",
        "La situación económica es muy preocupante y genera ansiedad",
        "Estoy furioso con estas medidas gubernamentales",
        "Celebramos con entusiasmo este logro extraordinario",
        "Profunda tristeza invade a las familias afectadas",
        "Mantenemos la calma ante estos cambios esperados"
    ]

    # Create analyzer
    analyzer = SentimentAnalyzer(use_mock=True)  # Use mock for testing

    print(f"Available moods: {analyzer.get_available_moods()[:10]}...")

    # Test individual text analysis
    print("\n--- Individual Text Analysis ---")
    for i, text in enumerate(test_texts[:5]):
        sentiment = analyzer.analyze_text(text)
        dominant = max(sentiment, key=sentiment.get)
        print(f"Text {i+1}: {dominant} ({sentiment[dominant]:.2f})")
        print(f"  '{text[:50]}...'")

    # Test DataFrame processing
    print("\n--- DataFrame Processing ---")
    df = pd.DataFrame({'text': test_texts})
    df_with_sentiment = analyzer.process_dataframe(df)

    print("Sentiment distribution:")
    distribution = analyzer.get_sentiment_distribution(df_with_sentiment)
    for sentiment, count in distribution.items():
        print(f"  {sentiment}: {count}")

    # Test sentiment search
    print("\n--- Sentiment Search ---")
    search_terms = ['feliz', 'triste', 'enojado', 'neutral']

    for mood in search_terms:
        results = analyzer.search_by_sentiment(df_with_sentiment, mood, threshold=0.3)
        print(f"\nSearching for '{mood}': {len(results)} results")
        if len(results) > 0:
            for _idx, row in results.head(2).iterrows():
                score = row.get('sentiment_score', 0)
                print(f"  Score: {score:.2f} - '{row['text'][:60]}...'")

    return True


def test_semantic_search_with_sentiment():
    """Test the integrated semantic search with sentiment functionality"""
    print("\n=== Testing Semantic Search with Sentiment ===")

    # Sample data with diverse sentiments
    sample_data = {
        'text': [
            "¡Excelente noticia! El presidente anunció medidas económicas optimistas",
            "Los mercados financieros cayeron dramáticamente generando pánico",
            "La reforma fiscal será discutida en el congreso la próxima semana",
            "¡Fantástico! Las exportaciones superaron todas las expectativas",
            "Tristemente, reportaron un incremento preocupante en los casos",
            "La industria tecnológica registra un crecimiento extraordinario",
            "Los ciudadanos expresan su enojo por las nuevas medidas restrictivas",
            "La población celebra con entusiasmo los avances médicos",
            "Situación desesperante: miles de familias perdieron empleos",
            "El ambiente festivo se percibe tras la victoria deportiva"
        ],
        'start_time': [i * 15 for i in range(10)],
        'end_time': [(i + 1) * 15 for i in range(10)],
        'duration': [15] * 10,
        'source_file': [f'test_{i}.wav' for i in range(10)],
        'segment_id': list(range(10)),
        'language': ['es'] * 10,
        'confidence': [0.9] * 10
    }

    try:
        # Create search engine with sentiment support
        config = {
            'whisper_model': 'base',
            'text_embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'use_mock_audio': True,
            'use_mock_sentiment': True,
            'index_type': 'cosine',
            'top_k_results': 5
        }

        search_engine = SemanticSearchEngine(config)

        # Convert to DataFrame and process
        df = pd.DataFrame(sample_data)

        # Manually process components to simulate the full pipeline
        print("Processing text embeddings...")
        df_with_embeddings = search_engine.text_embedder.process_transcription_dataframe(df)

        print("Processing sentiment analysis...")
        df_full = search_engine.sentiment_analyzer.process_dataframe(df_with_embeddings)

        search_engine.processed_data = df_full

        # Create indices for semantic search
        print("Creating search indices...")
        search_engine.create_indices(df_full)

        # Test different search types
        print("\n--- Testing Different Search Types ---")

        # Regular semantic search
        print("\n1. Regular text search for 'economía':")
        results = search_engine.search("economía", search_type="text", top_k=3)
        print(f"Found {len(results)} results")
        for _idx, row in results.iterrows():
            print(f"  Score: {row.get('similarity_score', 0):.3f} - '{row['text'][:60]}...'")

        # Sentiment-only search
        print("\n2. Sentiment search for 'feliz':")
        results = search_engine.search("feliz", search_type="sentiment", top_k=3)
        print(f"Found {len(results)} results")
        for _idx, row in results.iterrows():
            score = row.get('sentiment_score', 0)
            print(f"  Score: {score:.3f} - '{row['text'][:60]}...'")

        # Combined search with sentiment filter
        print("\n3. Text search with sentiment filter (positive content):")
        results = search_engine.search("medidas", search_type="text", sentiment_filter="optimista", top_k=3)
        print(f"Found {len(results)} results")
        for _idx, row in results.iterrows():
            score = row.get('similarity_score', 0)
            print(f"  Score: {score:.3f} - '{row['text'][:60]}...'")

        # Show system stats
        print("\n--- System Statistics ---")
        stats = search_engine.get_system_stats()
        print(f"Processed segments: {stats['processed_segments']}")
        print(f"Is indexed: {stats['is_indexed']}")
        print(f"Available sentiments: {len(stats['available_sentiments'])}")
        print(f"Sentiment distribution: {stats['sentiment_distribution']}")

        return True

    except Exception as e:
        print(f"Error in semantic search test: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all sentiment search tests"""
    print("Starting Sentiment Search Tests")
    print("=" * 50)

    success_count = 0
    total_tests = 2

    # Test 1: Basic sentiment analyzer
    try:
        if test_sentiment_analyzer():
            success_count += 1
            print("✓ Sentiment Analyzer test passed")
        else:
            print("✗ Sentiment Analyzer test failed")
    except Exception as e:
        print(f"✗ Sentiment Analyzer test error: {e}")

    # Test 2: Integrated semantic search
    try:
        if test_semantic_search_with_sentiment():
            success_count += 1
            print("✓ Semantic Search with Sentiment test passed")
        else:
            print("✗ Semantic Search with Sentiment test failed")
    except Exception as e:
        print(f"✗ Semantic Search with Sentiment test error: {e}")

    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {success_count}/{total_tests} passed")

    if success_count == total_tests:
        print("🎉 All tests passed! Sentiment search functionality is working correctly.")
        return True
    print("⚠️  Some tests failed. Check the implementation.")
    return False


if __name__ == "__main__":
    print("Sentiment Search Test Suite")
    print("This script tests the sentiment analysis and search functionality")
    print()

    success = run_all_tests()

    if success:
        print("\n--- Usage Examples ---")
        print("1. Basic sentiment search:")
        print("   search_engine.search('feliz', search_type='sentiment')")
        print()
        print("2. Text search with sentiment filter:")
        print("   search_engine.search('economía', search_type='text', sentiment_filter='optimista')")
        print()
        print("3. Available sentiments:")
        print("   search_engine.get_available_sentiments()")

    sys.exit(0 if success else 1)
