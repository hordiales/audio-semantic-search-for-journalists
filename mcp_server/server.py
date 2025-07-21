#!/usr/bin/env python3
"""
MCP Server for Audio Semantic Search for Journalists
Exposes audio dataset search functionality as MCP tools for Claude Desktop
"""

import asyncio
import sys
import os
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
import mcp.types as types

# Import our audio search components
from query_client import AudioDatasetClient

class AudioSearchMCPServer:
    def __init__(self):
        self.server = Server("audio-search")
        self.client: Optional[AudioDatasetClient] = None
        self.dataset_dir = None
        self.logger = logging.getLogger(__name__)
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all MCP handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="semantic_search",
                    description="Search audio segments using semantic text search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text (e.g., 'economía política', 'crisis económica')"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="audio_search",
                    description="Search audio segments using audio keyword/class search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Audio-related query (e.g., 'applause', 'music', 'speech')"
                            },
                            "k": {
                                "type": "integer", 
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="sentiment_search",
                    description="Search audio segments by sentiment/emotion",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "description": "Sentiment to search for (positive, negative, neutral, joy, anger, fear, etc.)"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)", 
                                "default": 5
                            }
                        },
                        "required": ["sentiment"]
                    }
                ),
                Tool(
                    name="hybrid_search",
                    description="Combined text and audio search with weighted scoring",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for both text and audio content"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            },
                            "text_weight": {
                                "type": "number",
                                "description": "Weight for text search (0.0-1.0, default: 0.7)",
                                "default": 0.7
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="mood_search",
                    description="Search with text query filtered by sentiment",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Text search query"
                            },
                            "sentiment": {
                                "type": "string", 
                                "description": "Sentiment filter (positive, negative, neutral, etc.)"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query", "sentiment"]
                    }
                ),
                Tool(
                    name="browse_dataset",
                    description="Browse random segments from the dataset",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "count": {
                                "type": "integer",
                                "description": "Number of segments to browse (default: 10)",
                                "default": 10
                            }
                        }
                    }
                ),
                Tool(
                    name="dataset_stats",
                    description="Get statistics about the audio dataset",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="find_text",
                    description="Find segments containing specific text",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to search for in transcriptions"
                            }
                        },
                        "required": ["text"]
                    }
                ),
                Tool(
                    name="get_similar",
                    description="Find segments similar to a specific segment by index",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "Index of the reference segment"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of similar segments to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["index"]
                    }
                ),
                Tool(
                    name="analyze_sentiment",
                    description="Analyze sentiment distribution for segments about a topic",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Topic to analyze sentiment for"
                            }
                        },
                        "required": ["topic"]
                    }
                ),
                Tool(
                    name="list_sentiments",
                    description="List available sentiment categories",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_capabilities",
                    description="Get system capabilities and status",
                    inputSchema={
                        "type": "object", 
                        "properties": {}
                    }
                ),
                Tool(
                    name="check_status",
                    description="Check if the MCP server is ready and fully initialized",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="play_audio_segment",
                    description="Play an audio segment from the dataset using the system's audio player",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_file": {
                                "type": "string",
                                "description": "Name of the source audio file (without path)"
                            },
                            "start_time": {
                                "type": "number",
                                "description": "Start time of the segment in seconds"
                            },
                            "end_time": {
                                "type": "number", 
                                "description": "End time of the segment in seconds"
                            },
                            "segment_index": {
                                "type": "integer",
                                "description": "Optional: Index of segment from search results to play",
                                "minimum": 0
                            }
                        },
                        "required": ["source_file", "start_time", "end_time"]
                    }
                ),
                Tool(
                    name="process_youtube_url",
                    description="Download audio from YouTube URL and add it to the dataset pipeline",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "youtube_url": {
                                "type": "string",
                                "description": "YouTube URL to download and process (e.g., https://www.youtube.com/watch?v=VIDEO_ID)"
                            },
                            "title": {
                                "type": "string",
                                "description": "Optional custom title for the downloaded audio (if not provided, will use video title)",
                                "default": ""
                            }
                        },
                        "required": ["youtube_url"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
            """Handle tool calls"""
            
            if not self.client:
                # Check if we're still initializing
                if hasattr(self, 'dataset_dir'):
                    return [types.TextContent(
                        type="text",
                        text="⏳ El MCP server está inicializándose... Por favor, espera unos segundos e intenta de nuevo.\n\n🔄 Cargando modelos de IA y dataset de audio..."
                    )]
                else:
                    return [types.TextContent(
                        type="text",
                        text="❌ Error: Audio search client not initialized. Please ensure the dataset is available."
                    )]
            
            try:
                if name == "semantic_search":
                    return await self._handle_semantic_search(arguments)
                elif name == "audio_search":
                    return await self._handle_audio_search(arguments)
                elif name == "sentiment_search":
                    return await self._handle_sentiment_search(arguments)
                elif name == "hybrid_search":
                    return await self._handle_hybrid_search(arguments)
                elif name == "mood_search":
                    return await self._handle_mood_search(arguments)
                elif name == "browse_dataset":
                    return await self._handle_browse_dataset(arguments)
                elif name == "dataset_stats":
                    return await self._handle_dataset_stats(arguments)
                elif name == "find_text":
                    return await self._handle_find_text(arguments)
                elif name == "get_similar":
                    return await self._handle_get_similar(arguments)
                elif name == "analyze_sentiment":
                    return await self._handle_analyze_sentiment(arguments)
                elif name == "list_sentiments":
                    return await self._handle_list_sentiments(arguments)
                elif name == "get_capabilities":
                    return await self._handle_get_capabilities(arguments)
                elif name == "check_status":
                    return await self._handle_check_status(arguments)
                elif name == "play_audio_segment":
                    return await self._handle_play_audio_segment(arguments)
                elif name == "process_youtube_url":
                    return await self._handle_process_youtube_url(arguments)
                else:
                    return [types.TextContent(
                        type="text",
                        text=f"❌ Unknown tool: {name}"
                    )]
                    
            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=f"❌ Error executing {name}: {str(e)}"
                )]
    
    async def _handle_semantic_search(self, args: dict) -> list[types.TextContent]:
        """Handle semantic text search"""
        query = args["query"]
        k = args.get("k", 5)
        
        results = self.client.search_text(query, k)
        
        if not results:
            return [types.TextContent(
                type="text",
                text=f"❌ No se encontraron resultados para: '{query}'"
            )]
        
        response = f"🔍 Búsqueda semántica: '{query}'\n"
        response += f"✅ Encontrados {len(results)} segmentos relevantes\n\n"
        
        for result in results:
            response += f"**Segmento {result['rank']}** (Score: {result['score']:.3f})\n"
            response += f"📄 Archivo: {result['source_file']}\n"
            response += f"⏰ Tiempo: {result['start_time']:.1f}s - {result['end_time']:.1f}s\n"
            response += f"📝 Texto: {result['text']}\n"
            if 'sentiment' in result:
                response += f"🎭 Sentimiento: {result['sentiment']}\n"
            response += "\n---\n\n"
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_audio_search(self, args: dict) -> list[types.TextContent]:
        """Handle audio keyword search"""
        query = args["query"]
        k = args.get("k", 5)
        
        results = self.client.search_audio(query, k)
        
        if not results:
            return [types.TextContent(
                type="text",
                text=f"❌ No se encontraron segmentos con audio relevante para: '{query}'"
            )]
        
        response = f"🎵 Búsqueda de audio: '{query}'\n"
        response += f"✅ Encontrados {len(results)} segmentos con contenido de audio relevante\n\n"
        
        for result in results:
            response += f"**Segmento {result['rank']}** (Score: {result['score']:.3f})\n"
            response += f"📄 Archivo: {result['source_file']}\n"
            response += f"⏰ Tiempo: {result['start_time']:.1f}s - {result['end_time']:.1f}s\n"
            response += f"📝 Texto: {result['text']}\n"
            if 'audio_class' in result:
                response += f"🎵 Clase de audio: {result['audio_class']}\n"
            response += "\n---\n\n"
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_sentiment_search(self, args: dict) -> list[types.TextContent]:
        """Handle sentiment search"""
        sentiment = args["sentiment"]
        k = args.get("k", 5)
        
        if not self.client.sentiment_enabled:
            return [types.TextContent(
                type="text",
                text="❌ El análisis de sentimientos no está habilitado"
            )]
        
        results = self.client.sentiment_search_engine.search_by_sentiment(sentiment, k)
        
        if not results:
            return [types.TextContent(
                type="text", 
                text=f"❌ No se encontraron segmentos con sentimiento: '{sentiment}'"
            )]
        
        response = f"🎭 Búsqueda por sentimiento: '{sentiment}'\n"
        response += f"✅ Encontrados {len(results)} segmentos\n\n"
        
        for result in results:
            response += f"**Segmento {result['rank']}** (Score: {result['score']:.3f})\n"
            response += f"📄 Archivo: {result['source_file']}\n"
            response += f"⏰ Tiempo: {result['start_time']:.1f}s - {result['end_time']:.1f}s\n"
            response += f"📝 Texto: {result['text']}\n"
            response += f"🎭 Sentimiento: {result['sentiment']} (Confianza: {result.get('sentiment_confidence', 'N/A')})\n"
            response += "\n---\n\n"
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_hybrid_search(self, args: dict) -> list[types.TextContent]:
        """Handle hybrid search"""
        query = args["query"]
        k = args.get("k", 5)
        text_weight = args.get("text_weight", 0.7)
        
        results = self.client.search_combined(query, k, text_weight)
        
        if not results:
            return [types.TextContent(
                type="text",
                text=f"❌ No se encontraron resultados para búsqueda híbrida: '{query}'"
            )]
        
        response = f"🔄 Búsqueda híbrida: '{query}'\n"
        response += f"⚖️ Peso texto: {text_weight:.1f}, Peso audio: {1-text_weight:.1f}\n"
        response += f"✅ Encontrados {len(results)} segmentos\n\n"
        
        for result in results:
            response += f"**Segmento {result['rank']}** (Score combinado: {result['score']:.3f})\n"
            response += f"📊 Score texto: {result.get('text_score', 0):.3f} | Score audio: {result.get('audio_score', 0):.3f}\n"
            response += f"📄 Archivo: {result['source_file']}\n"
            response += f"⏰ Tiempo: {result['start_time']:.1f}s - {result['end_time']:.1f}s\n"
            response += f"📝 Texto: {result['text']}\n"
            response += "\n---\n\n"
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_mood_search(self, args: dict) -> list[types.TextContent]:
        """Handle mood search (text query with sentiment filter)"""
        query = args["query"]
        sentiment = args["sentiment"]
        k = args.get("k", 5)
        
        if not self.client.sentiment_enabled:
            return [types.TextContent(
                type="text",
                text="❌ El análisis de sentimientos no está habilitado"
            )]
        
        results = self.client.sentiment_search_engine.search_with_sentiment_filter(
            query, sentiment, k
        )
        
        if not results:
            return [types.TextContent(
                type="text",
                text=f"❌ No se encontraron resultados para '{query}' con sentimiento '{sentiment}'"
            )]
        
        response = f"🎭🔍 Búsqueda con filtro de sentimiento\n"
        response += f"📝 Consulta: '{query}'\n"
        response += f"🎭 Sentimiento: '{sentiment}'\n"
        response += f"✅ Encontrados {len(results)} segmentos\n\n"
        
        for result in results:
            response += f"**Segmento {result['rank']}** (Score: {result['score']:.3f})\n"
            response += f"📄 Archivo: {result['source_file']}\n"
            response += f"⏰ Tiempo: {result['start_time']:.1f}s - {result['end_time']:.1f}s\n"
            response += f"📝 Texto: {result['text']}\n"
            response += f"🎭 Sentimiento: {result['sentiment']}\n"
            response += "\n---\n\n"
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_browse_dataset(self, args: dict) -> list[types.TextContent]:
        """Handle dataset browsing"""
        count = args.get("count", 10)
        
        import random
        total_segments = len(self.client.df)
        indices = random.sample(range(total_segments), min(count, total_segments))
        
        response = f"📊 Explorando {len(indices)} segmentos aleatorios del dataset\n"
        response += f"📈 Total de segmentos en el dataset: {total_segments}\n\n"
        
        for i, idx in enumerate(indices, 1):
            row = self.client.df.iloc[idx]
            response += f"**Segmento {i}** (Índice: {idx})\n"
            response += f"📄 Archivo: {row['source_file']}\n"
            response += f"⏰ Tiempo: {row['start_time']:.1f}s - {row['end_time']:.1f}s\n"
            response += f"📝 Texto: {row['text']}\n"
            if 'sentiment' in row:
                response += f"🎭 Sentimiento: {row['sentiment']}\n"
            response += "\n---\n\n"
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_dataset_stats(self, args: dict) -> list[types.TextContent]:
        """Handle dataset statistics"""
        df = self.client.df
        
        response = "📊 Estadísticas del Dataset de Audio\n"
        response += "=" * 40 + "\n\n"
        
        response += f"📈 **Total de segmentos:** {len(df)}\n"
        response += f"📁 **Archivos únicos:** {df['source_file'].nunique()}\n"
        
        # Duración total
        total_duration = (df['end_time'] - df['start_time']).sum()
        response += f"⏱️ **Duración total:** {total_duration:.1f} segundos ({total_duration/60:.1f} minutos)\n"
        
        # Duración promedio de segmentos
        avg_duration = (df['end_time'] - df['start_time']).mean()
        response += f"📊 **Duración promedio por segmento:** {avg_duration:.1f} segundos\n"
        
        # Estadísticas de texto
        text_lengths = df['text'].str.len()
        response += f"📝 **Longitud promedio de texto:** {text_lengths.mean():.0f} caracteres\n"
        response += f"📝 **Texto más corto:** {text_lengths.min()} caracteres\n"
        response += f"📝 **Texto más largo:** {text_lengths.max()} caracteres\n"
        
        # Sentimientos si están disponibles
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            response += f"\n🎭 **Distribución de sentimientos:**\n"
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(df)) * 100
                response += f"  • {sentiment}: {count} ({percentage:.1f}%)\n"
        
        # Archivos más representados
        file_counts = df['source_file'].value_counts().head(5)
        response += f"\n📁 **Archivos con más segmentos:**\n"
        for file, count in file_counts.items():
            response += f"  • {file}: {count} segmentos\n"
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_find_text(self, args: dict) -> list[types.TextContent]:
        """Handle text finding"""
        text = args["text"]
        
        # Search for text in transcriptions
        matches = self.client.df[self.client.df['text'].str.contains(text, case=False, na=False)]
        
        if matches.empty:
            return [types.TextContent(
                type="text",
                text=f"❌ No se encontraron segmentos que contengan: '{text}'"
            )]
        
        response = f"🔍 Búsqueda de texto: '{text}'\n"
        response += f"✅ Encontrados {len(matches)} segmentos\n\n"
        
        for idx, (_, row) in enumerate(matches.head(10).iterrows(), 1):
            response += f"**Segmento {idx}** (Índice: {row.name})\n"
            response += f"📄 Archivo: {row['source_file']}\n"
            response += f"⏰ Tiempo: {row['start_time']:.1f}s - {row['end_time']:.1f}s\n"
            response += f"📝 Texto: {row['text']}\n"
            response += "\n---\n\n"
        
        if len(matches) > 10:
            response += f"... y {len(matches) - 10} segmentos más\n"
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_get_similar(self, args: dict) -> list[types.TextContent]:
        """Handle finding similar segments"""
        index = args["index"]
        k = args.get("k", 5)
        
        if index >= len(self.client.df) or index < 0:
            return [types.TextContent(
                type="text",
                text=f"❌ Índice inválido: {index}. Debe estar entre 0 y {len(self.client.df)-1}"
            )]
        
        reference_row = self.client.df.iloc[index]
        reference_text = reference_row['text']
        
        # Use semantic search with the reference text
        similar_results = self.client.search_text(reference_text, k + 1)  # +1 to exclude self
        
        # Filter out the reference segment itself
        similar_results = [r for r in similar_results if r.get('index', -1) != index][:k]
        
        response = f"🔍 Segmentos similares al índice {index}\n"
        response += f"📝 **Texto de referencia:** {reference_text}\n\n"
        
        if not similar_results:
            response += "❌ No se encontraron segmentos similares\n"
        else:
            response += f"✅ Encontrados {len(similar_results)} segmentos similares\n\n"
            
            for result in similar_results:
                response += f"**Segmento {result['rank']}** (Similitud: {result['score']:.3f})\n"
                response += f"📄 Archivo: {result['source_file']}\n"
                response += f"⏰ Tiempo: {result['start_time']:.1f}s - {result['end_time']:.1f}s\n"
                response += f"📝 Texto: {result['text']}\n"
                response += "\n---\n\n"
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_analyze_sentiment(self, args: dict) -> list[types.TextContent]:
        """Handle sentiment analysis for a topic"""
        topic = args["topic"]
        
        if not self.client.sentiment_enabled:
            return [types.TextContent(
                type="text",
                text="❌ El análisis de sentimientos no está habilitado"
            )]
        
        analysis = self.client.sentiment_search_engine.analyze_sentiment_for_topic(topic)
        
        if not analysis:
            return [types.TextContent(
                type="text",
                text=f"❌ No se encontraron segmentos relacionados con: '{topic}'"
            )]
        
        response = f"🎭 Análisis de sentimientos para: '{topic}'\n"
        response += "=" * 50 + "\n\n"
        
        response += f"📊 **Total de segmentos analizados:** {analysis['total_segments']}\n\n"
        
        response += "📈 **Distribución de sentimientos:**\n"
        for sentiment, data in analysis['sentiment_distribution'].items():
            count = data['count']
            percentage = data['percentage']
            response += f"  • {sentiment}: {count} segmentos ({percentage:.1f}%)\n"
        
        response += f"\n🎯 **Sentimiento predominante:** {analysis['dominant_sentiment']}\n"
        response += f"📊 **Score promedio:** {analysis['average_confidence']:.3f}\n\n"
        
        response += "💡 **Ejemplos por sentimiento:**\n"
        for sentiment, examples in analysis['examples'].items():
            response += f"\n**{sentiment.upper()}:**\n"
            for example in examples[:2]:  # Show top 2 examples
                response += f"  • {example['text'][:100]}...\n"
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_list_sentiments(self, args: dict) -> list[types.TextContent]:
        """Handle listing available sentiments"""
        if not self.client.sentiment_enabled:
            return [types.TextContent(
                type="text",
                text="❌ El análisis de sentimientos no está habilitado"
            )]
        
        sentiments = self.client.sentiment_search_engine.get_available_sentiments()
        
        response = "🎭 Sentimientos disponibles en el dataset\n"
        response += "=" * 40 + "\n\n"
        
        response += "📊 **Sentimientos básicos:**\n"
        basic_sentiments = ['positive', 'negative', 'neutral']
        for sentiment in basic_sentiments:
            if sentiment in sentiments:
                count = sentiments[sentiment]
                response += f"  • {sentiment}: {count} segmentos\n"
        
        response += "\n🎭 **Emociones específicas:**\n"
        emotions = ['joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust']
        for emotion in emotions:
            if emotion in sentiments:
                count = sentiments[emotion]
                response += f"  • {emotion}: {count} segmentos\n"
        
        response += "\n📈 **Otros sentimientos detectados:**\n"
        other_sentiments = [s for s in sentiments.keys() 
                          if s not in basic_sentiments + emotions]
        for sentiment in sorted(other_sentiments):
            count = sentiments[sentiment]
            response += f"  • {sentiment}: {count} segmentos\n"
        
        response += f"\n📊 **Total de categorías:** {len(sentiments)}\n"
        total_segments = sum(sentiments.values())
        response += f"📈 **Total de segmentos con sentimiento:** {total_segments}\n"
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_get_capabilities(self, args: dict) -> list[types.TextContent]:
        """Handle getting system capabilities"""
        response = "🔧 Capacidades del Sistema de Búsqueda de Audio\n"
        response += "=" * 50 + "\n\n"
        
        # Dataset info
        response += f"📊 **Dataset:** {len(self.client.df)} segmentos de audio cargados\n"
        response += f"📁 **Archivos:** {self.client.df['source_file'].nunique()} archivos únicos\n\n"
        
        # Search capabilities
        response += "🔍 **Capacidades de búsqueda:**\n"
        response += "  ✅ Búsqueda semántica de texto\n"
        response += "  ✅ Búsqueda por palabras clave de audio\n"
        response += "  ✅ Búsqueda híbrida (texto + audio)\n"
        
        if self.client.sentiment_enabled:
            response += "  ✅ Búsqueda por sentimientos\n"
            response += "  ✅ Análisis de sentimientos por tema\n"
        else:
            response += "  ❌ Búsqueda por sentimientos (no disponible)\n"
        
        # Models info
        response += "\n🧠 **Modelos cargados:**\n"
        response += "  ✅ Embeddings de texto (Sentence Transformers)\n"
        if self.client.hybrid_search_enabled:
            response += "  ✅ Embeddings de audio (YAMNet)\n"
        else:
            response += "  ❌ Embeddings de audio (no disponible)\n"
        
        # Index info
        response += "\n📚 **Índices vectoriales:**\n"
        if hasattr(self.client, 'index_manager') and self.client.index_manager:
            response += "  ✅ Índice de texto\n"
            response += "  ✅ Índice de audio\n"
        else:
            response += "  ❌ Índices no cargados\n"
        
        # Available tools
        response += "\n🛠️ **Herramientas MCP disponibles:**\n"
        tools = [
            "semantic_search", "audio_search", "sentiment_search", 
            "hybrid_search", "mood_search", "browse_dataset",
            "dataset_stats", "find_text", "get_similar", 
            "analyze_sentiment", "list_sentiments", "get_capabilities",
            "check_status", "play_audio_segment", "process_youtube_url"
        ]
        for tool in tools:
            response += f"  • {tool}\n"
        
        return [types.TextContent(type="text", text=response)]
    
    async def _handle_check_status(self, args: dict) -> list[types.TextContent]:
        """Handle status check"""
        if not self.client:
            if hasattr(self, 'dataset_dir'):
                response = "⏳ **Estado del MCP Server: INICIALIZÁNDOSE**\n\n"
                response += "🔄 El servidor está cargando los siguientes componentes:\n"
                response += "  • 🧠 Modelos de embeddings de texto (Sentence Transformers)\n"
                response += "  • 🎵 Modelo de audio YAMNet (TensorFlow)\n"
                response += "  • 📊 Dataset de audio (301 segmentos)\n"
                response += "  • 🔍 Índices vectoriales FAISS\n\n"
                response += "⏱️ **Tiempo estimado:** 30-60 segundos\n"
                response += "💡 **Consejo:** Ejecuta este comando de nuevo en unos segundos para verificar el estado."
            else:
                response = "❌ **Estado del MCP Server: ERROR**\n\n"
                response += "El servidor no se ha inicializado correctamente.\n"
                response += "Por favor, verifica que el dataset esté disponible."
        else:
            df = self.client.df
            response = "✅ **Estado del MCP Server: LISTO**\n\n"
            response += "🎉 Todos los componentes están cargados y funcionando:\n"
            response += f"  • 📊 Dataset: {len(df)} segmentos cargados\n"
            response += f"  • 📁 Archivos: {df['source_file'].nunique()} archivos únicos\n"
            response += "  • 🧠 Embeddings de texto: ✅ Listos\n"
            
            if self.client.hybrid_search_enabled:
                response += "  • 🎵 Embeddings de audio: ✅ Listos\n"
            else:
                response += "  • 🎵 Embeddings de audio: ❌ No disponibles\n"
            
            if self.client.sentiment_enabled:
                response += "  • 🎭 Análisis de sentimientos: ✅ Listo\n"
            else:
                response += "  • 🎭 Análisis de sentimientos: ❌ No disponible\n"
            
            response += "\n🚀 **El servidor está listo para recibir consultas de búsqueda.**"
        
        return [types.TextContent(type="text", text=response)]
    
    def _get_audio_player_command(self):
        """Get the appropriate audio player command for the current OS"""
        system = platform.system()
        
        if system == "Darwin":  # macOS
            # Try ffplay first (from ffmpeg), then fallback to afplay
            if shutil.which("ffplay"):
                return "ffplay"
            elif shutil.which("afplay"):
                return "afplay"
            else:
                return None
        elif system == "Windows":
            # Use Windows Media Player command line
            if shutil.which("wmplayer"):
                return "wmplayer"
            elif shutil.which("ffplay"):
                return "ffplay"
            else:
                return None
        elif system == "Linux":
            # Try various Linux audio players
            for player in ["ffplay", "cvlc", "aplay", "paplay", "mplayer"]:
                if shutil.which(player):
                    return player
            return None
        else:
            return None
    
    def _build_audio_command(self, player, audio_file, start_time, end_time):
        """Build the command to play audio segment based on the player"""
        duration = end_time - start_time
        
        if player == "ffplay":
            return [
                "ffplay", 
                "-ss", str(start_time),
                "-t", str(duration),
                "-autoexit",
                "-nodisp",  # No video display
                audio_file
            ]
        elif player == "afplay":
            # afplay doesn't support time ranges directly, so we'll play the whole file
            # and mention the time range in the response
            return ["afplay", audio_file]
        elif player == "wmplayer":
            return ["wmplayer", audio_file]
        elif player == "cvlc":
            return [
                "cvlc", 
                "--play-and-exit",
                "--start-time", str(start_time),
                "--stop-time", str(end_time),
                audio_file
            ]
        elif player in ["aplay", "paplay"]:
            # These are for raw audio, might not work with all formats
            return [player, audio_file]
        elif player == "mplayer":
            return [
                "mplayer",
                "-ss", str(start_time),
                "-endpos", str(duration),
                audio_file
            ]
        else:
            return None
    
    async def _handle_play_audio_segment(self, args: dict) -> list[types.TextContent]:
        """Handle audio segment playback"""
        source_file = args["source_file"]
        start_time = args["start_time"]
        end_time = args["end_time"]
        segment_index = args.get("segment_index")
        
        try:
            # Extract just the filename if a path was provided
            source_filename = Path(source_file).name
            # Get the audio player command
            player = self._get_audio_player_command()
            if not player:
                return [types.TextContent(
                    type="text",
                    text="❌ No audio player found on this system. Please install ffmpeg (ffplay) or another supported audio player."
                )]
            
            # Find the audio file in the dataset
            audio_file_path = None
            
            # Look for the audio file in common locations
            possible_paths = [
                self.dataset_dir / "converted" / source_filename,
                self.dataset_dir / "audio" / source_filename,
                self.dataset_dir.parent / "data" / source_filename,
                self.dataset_dir.parent / "temp_audio" / source_filename,
            ]
            
            # Also try different extensions
            for base_path in possible_paths:
                if base_path.exists():
                    audio_file_path = base_path
                    break
                
                # Try different extensions
                for ext in [".wav", ".mp3", ".opus", ".m4a", ".flac"]:
                    alt_path = base_path.with_suffix(ext)
                    if alt_path.exists():
                        audio_file_path = alt_path
                        break
                
                if audio_file_path:
                    break
            
            if not audio_file_path:
                return [types.TextContent(
                    type="text",
                    text=f"❌ Audio file not found: {source_filename}\n"
                         f"Original: {source_file}\n"
                         f"Searched in: {', '.join(str(p.parent) for p in possible_paths)}\n"
                         f"Make sure the audio files are available in the dataset directory."
                )]
            
            # Build the playback command
            command = self._build_audio_command(player, str(audio_file_path), start_time, end_time)
            if not command:
                return [types.TextContent(
                    type="text",
                    text=f"❌ Could not build playback command for player: {player}"
                )]
            
            # Execute the command asynchronously
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Don't wait for completion if it's a GUI player
            if player in ["wmplayer", "afplay"]:
                response = f"🎵 **Reproduciendo segmento de audio**\n\n"
            else:
                # Wait for completion for command-line players
                stdout, stderr = await process.communicate()
                response = f"🎵 **Segmento de audio reproducido**\n\n"
            
            response += f"📄 **Archivo:** {source_filename}\n"
            response += f"⏰ **Tiempo:** {start_time:.1f}s - {end_time:.1f}s ({end_time-start_time:.1f}s de duración)\n"
            response += f"🔧 **Reproductor:** {player}\n"
            response += f"📂 **Ubicación:** {audio_file_path.parent.name}/{audio_file_path.name}\n"
            
            if segment_index is not None:
                response += f"📊 **Índice del segmento:** {segment_index}\n"
            
            if player == "afplay":
                response += f"\n⚠️ **Nota:** afplay reproduce el archivo completo. El segmento específico es de {start_time:.1f}s a {end_time:.1f}s.\n"
            
            response += f"\n✅ Comando ejecutado: `{' '.join(command)}`"
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ Error al reproducir el segmento de audio: {str(e)}\n"
                     f"Archivo: {source_filename}\n"
                     f"Original: {source_file}\n"
                     f"Tiempo: {start_time:.1f}s - {end_time:.1f}s"
            )]
    
    async def _handle_process_youtube_url(self, args: dict) -> list[types.TextContent]:
        """Handle YouTube URL processing"""
        youtube_url = args["youtube_url"]
        custom_title = args.get("title", "")
        
        try:
            # Change to the project root directory
            project_root = self.dataset_dir.parent if self.dataset_dir else Path("../")
            original_cwd = os.getcwd()
            
            response = f"🎬 **Procesando video de YouTube**\n"
            response += f"🔗 **URL:** {youtube_url}\n\n"
            
            # Step 1: Clean dataset
            response += "🧹 **Paso 1:** Limpiando dataset anterior...\n"
            clean_process = await asyncio.create_subprocess_exec(
                "./clean_dataset.sh",
                cwd=str(project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            clean_stdout, clean_stderr = await clean_process.communicate()
            
            if clean_process.returncode != 0:
                return [types.TextContent(
                    type="text",
                    text=f"❌ Error limpiando dataset: {clean_stderr.decode()}"
                )]
            
            response += "✅ Dataset limpiado\n\n"
            
            # Step 2: Download audio with yt-dlp
            response += "⬇️ **Paso 2:** Descargando audio de YouTube...\n"
            
            # Build yt-dlp command
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            
            # Build filename from custom title or use yt-dlp default
            if custom_title:
                # Sanitize title for filename
                safe_title = "".join(c for c in custom_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_title = safe_title.replace(' ', '_')
                output_template = f"{safe_title}.%(ext)s"
            else:
                output_template = "%(title)s.%(id)s.%(ext)s"
            
            ytdlp_command = [
                "yt-dlp",
                "-x",  # Extract audio only
                "--audio-format", "opus",  # Prefer opus format
                "--audio-quality", "0",    # Best quality
                "-o", str(data_dir / output_template),
                youtube_url
            ]
            
            ytdlp_process = await asyncio.create_subprocess_exec(
                *ytdlp_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            ytdlp_stdout, ytdlp_stderr = await ytdlp_process.communicate()
            
            if ytdlp_process.returncode != 0:
                return [types.TextContent(
                    type="text",
                    text=f"❌ Error descargando audio de YouTube:\n{ytdlp_stderr.decode()}\n\nComando: {' '.join(ytdlp_command)}"
                )]
            
            # Get the downloaded filename
            download_output = ytdlp_stdout.decode() + ytdlp_stderr.decode()
            response += f"✅ Audio descargado exitosamente\n"
            response += f"📁 Ubicación: data/\n\n"
            
            # Step 3: Build corpus dataset
            response += "🏗️ **Paso 3:** Procesando audio y construyendo dataset...\n"
            
            build_process = await asyncio.create_subprocess_exec(
                "./build_corpus_dataset.sh",
                cwd=str(project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            build_stdout, build_stderr = await build_process.communicate()
            
            if build_process.returncode != 0:
                return [types.TextContent(
                    type="text",
                    text=f"❌ Error construyendo dataset:\n{build_stderr.decode()}\n\nStdout:\n{build_stdout.decode()}"
                )]
            
            # Parse the build output for summary
            build_output = build_stdout.decode()
            response += "✅ Dataset construido exitosamente\n\n"
            
            # Step 4: Reload the dataset in MCP server
            response += "🔄 **Paso 4:** Recargando dataset en MCP server...\n"
            
            try:
                # Reinitialize the client with the new dataset
                await asyncio.get_event_loop().run_in_executor(
                    None, self.client._load_dataset
                )
                response += "✅ Dataset recargado en MCP server\n\n"
                
                # Get new dataset stats
                df = self.client.df
                response += "📊 **Resumen del nuevo dataset:**\n"
                response += f"  📈 Total de segmentos: {len(df)}\n"
                response += f"  📁 Archivos únicos: {df['source_file'].nunique()}\n"
                
                # Duration stats
                total_duration = (df['end_time'] - df['start_time']).sum()
                response += f"  ⏱️ Duración total: {total_duration:.1f} segundos ({total_duration/60:.1f} minutos)\n"
                
                # New files added
                new_files = df['source_file'].unique()
                response += f"  📄 Archivos procesados:\n"
                for file in new_files:
                    file_segments = len(df[df['source_file'] == file])
                    response += f"    • {file}: {file_segments} segmentos\n"
                
                # Sample of transcribed content
                response += f"\n📝 **Muestra del contenido transcrito:**\n"
                sample_texts = df['text'].head(3)
                for i, text in enumerate(sample_texts, 1):
                    preview = text[:100] + "..." if len(text) > 100 else text
                    response += f"  {i}. {preview}\n"
                
                response += f"\n🎉 **¡Proceso completado exitosamente!**\n"
                response += f"El contenido de YouTube ha sido procesado y está listo para búsquedas."
                
            except Exception as e:
                response += f"❌ Error recargando dataset: {str(e)}\n"
                response += f"El audio fue procesado pero necesitas reiniciar el MCP server manualmente."
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ Error procesando YouTube URL:\n{str(e)}\n\nURL: {youtube_url}"
            )]
    
    async def initialize_client(self, dataset_dir: str):
        """Initialize the audio search client"""
        try:
            self.dataset_dir = Path(dataset_dir)
            self.client = AudioDatasetClient(dataset_dir, logger=self.logger)
            await asyncio.get_event_loop().run_in_executor(
                None, self.client._load_dataset
            )
            # Only print status if running in terminal
            if hasattr(sys, 'stdout') and sys.stdout.isatty():
                self.logger.info(f"✅ MCP Server initialized with dataset from: {dataset_dir}")
            return True
        except Exception as e:
            # Always log errors
            self.logger.error(f"❌ Failed to initialize MCP server: {e}")
            return False
    
    def run(self, dataset_dir: str = "../dataset"):
        """Run the MCP server"""
        async def main():
            # Store dataset directory for lazy initialization
            self.dataset_dir = Path(dataset_dir)
            
            # Run the server immediately without waiting for client initialization
            from mcp.server.stdio import stdio_server
            async with stdio_server() as (read_stream, write_stream):
                # Start client initialization in background
                asyncio.create_task(self._lazy_initialize_client())
                
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="audio-search",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={}
                        )
                    )
                )
        
        asyncio.run(main())
    
    async def _lazy_initialize_client(self):
        """Initialize the client in the background"""
        try:
            if not self.client:
                self.client = AudioDatasetClient(str(self.dataset_dir), logger=self.logger)
                await asyncio.get_event_loop().run_in_executor(
                    None, self.client._load_dataset
                )
                # Only print status if running in terminal
                if hasattr(sys, 'stdout') and sys.stdout.isatty():
                    self.logger.info(f"✅ MCP Server initialized with dataset from: {self.dataset_dir}")
        except Exception as e:
            # Always log errors to stderr
            self.logger.error(f"❌ Failed to initialize MCP server: {e}")
            self.client = None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Search MCP Server")
    parser.add_argument(
        "--dataset-dir", 
        default="../dataset",
        help="Path to the dataset directory"
    )
    
    args = parser.parse_args()
    
    server = AudioSearchMCPServer()
    server.run(args.dataset_dir)