# Mini Doc-FAQ Agent Documentation

## Overview

The Mini Doc-FAQ Agent is a RAG (Retrieval-Augmented Generation) application built with Streamlit. It uses two lightweight agents: an Indexer and an Answerer.

## Features

The application provides several key features:

1. **Automatic Document Processing**: The Indexer agent automatically detects and processes `.txt` and `.md` files placed in the `/data` folder.

2. **Intelligent Chunking**: Documents are split into coherent chunks using sentence-based splitting with overlap to preserve context.

3. **Vector Search**: The system uses FAISS (Facebook AI Similarity Search) for fast similarity search across document chunks.

4. **Citation Support**: Answers include inline citations like [1], [2] that reference the source chunks used to generate the response.

## How It Works

### Indexer Agent

The Indexer is responsible for:
- Scanning the `/data` folder for text and markdown files
- Chunking documents into manageable pieces (approximately 300-800 tokens)
- Generating embeddings using the configured LLM provider (OpenAI or Gemini)
- Building a FAISS index for efficient similarity search

### Answerer Agent

The Answerer handles:
- Embedding user queries
- Retrieving the top-k most relevant chunks (default: 4)
- Generating concise answers with 2-5 bullet points
- Including inline citations to source material

## Supported Providers

The application supports two LLM providers:

- **OpenAI**: Uses `gpt-4o-mini` for chat and `text-embedding-3-small` for embeddings
- **Gemini**: Uses `gemini-1.5-flash` for chat and `text-embedding-004` for embeddings

## Usage

1. Place your documents (`.txt` or `.md` files) in the `/data` folder
2. The Indexer will automatically process them on first run
3. Enter questions in the query box
4. Receive answers with citations to relevant document chunks

## Technical Details

The application uses cosine similarity for vector search. Documents are chunked with approximately 120 words of overlap between chunks to ensure context is preserved across boundaries.

The system is designed to be lightweight and fast, with caching to avoid rebuilding the index on every rerun. All processing happens in-memory for simplicity.

## Limitations

Current limitations include:
- Text and Markdown files only (no PDF or DOCX support)
- In-memory index (not persisted to disk)
- No authentication or user management
- Single-session caching only

## Future Enhancements

Potential improvements could include:
- PDF document support
- Persistent index storage
- Metadata filtering
- URL-based document loading
- Hybrid search combining vector and keyword search

