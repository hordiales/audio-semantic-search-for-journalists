# Audio Search MCP Server

MCP (Model Context Protocol) server for audio semantic search functionality.

This server exposes audio dataset search capabilities as tools for Claude Desktop, including:

- Semantic text search
- Audio keyword search  
- Sentiment analysis
- Hybrid search (text + audio)
- Dataset exploration

## Usage

See `../MCP_SETUP.md` for complete installation and configuration instructions.

## Quick Start

With UV:
```bash
uv sync
uv run python start_uv.py --dataset-dir ../dataset
```

With Poetry:
```bash
poetry install
poetry run python server.py --dataset-dir ../dataset
```