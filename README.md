# Mini Doc-FAQ Agent (RAG + two tiny agents)

A minimal, production-clean Streamlit RAG application with two lightweight agents: an **Indexer** that processes and embeds documents, and an **Answerer** that retrieves relevant chunks and generates answers with inline citations.

## Features

- 📄 Auto-detects `.txt` and `.md` files in `/data` folder
- 🔍 Sentence-based chunking with overlap for better context preservation
- 🗂️ In-memory FAISS vector index (cosine similarity)
- 🤖 Dual LLM provider support: OpenAI or Google Gemini
- 📝 Answers with inline citations like [1], [2]
- ⚡ Cached indexing for fast reruns
- 🎨 Clean, user-friendly Streamlit UI

## Quick Start

### Prerequisites

- Python 3.10+
- API key for your chosen provider

### Installation

**For Gemini (currently configured):**
```bash
pip install streamlit faiss-cpu google-generativeai python-dotenv
```

**For OpenAI (if switching):**
```bash
pip install streamlit faiss-cpu openai python-dotenv
```

### Environment Variables

**Recommended: Use `.env` file**

1. Copy the `.env` file (or create it) in the project root
2. Add your API key:
   ```env
   GOOGLE_API_KEY=your_actual_api_key_here
   ```
   Or for OpenAI:
   ```env
   OPENAI_API_KEY=your_actual_api_key_here
   ```

**Alternative: System Environment Variables**

**Windows (PowerShell):**
```powershell
setx GOOGLE_API_KEY "your-api-key-here"
# or
setx OPENAI_API_KEY "your-api-key-here"
```

**Unix/Mac:**
```bash
export GOOGLE_API_KEY="your-api-key-here"
# or
export OPENAI_API_KEY="your-api-key-here"
```

**Note:** After setting environment variables in Windows, you may need to restart your terminal/IDE.

### Configuration

Open `app.py` and set the provider at the top:

```python
PROVIDER = "gemini"  # or "openai"
```

### Running the App

1. Place your `.txt` or `.md` files in the `/data` folder
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. The Indexer will automatically process your documents on first run
4. Ask questions in the query box!

## Project Structure

```
mini-doc-faq/
├── app.py          # Main Streamlit application
├── .env            # API keys (create from .env.example or add manually)
├── data/           # Place your .txt and .md files here
└── README.md       # This file
```

## How It Works

1. **Indexer Agent**: 
   - Scans `/data` for `.txt` and `.md` files
   - Chunks documents by sentences with overlap
   - Generates embeddings using the configured provider
   - Builds a FAISS index for fast similarity search

2. **Answerer Agent**:
   - Embeds the user's query
   - Retrieves top-k most similar chunks (default: 4)
   - Generates a concise answer with 2-5 bullet points
   - Includes inline citations referencing the retrieved chunks

## Models Used

**OpenAI:**
- Chat: `gpt-4o-mini`
- Embeddings: `text-embedding-3-small`

**Gemini:**
- Chat: `gemini-2.0-flash`
- Embeddings: `text-embedding-004`

## Limitations & Future Enhancements

**Current Limitations:**
- No persistence: index is rebuilt on each session (cached within session)
- Text/Markdown only: no PDF, DOCX, or other formats
- No authentication or user management
- In-memory index only (not saved to disk)

**Optional Upgrades:**
- Add PDF support with `pypdf` or `pdfplumber`
- Persist index to disk using `faiss.write_index()` and `faiss.read_index()`
- Add metadata filtering (e.g., filter by source file)
- Support for URL-based document loading
- Multi-file upload via Streamlit file uploader
- Hybrid search (vector + keyword)

## Error Handling

The app gracefully handles:
- Missing API keys (shows setup instructions)
- Empty `/data` folder (shows helpful message)
- Empty queries (ignores button click)
- File read errors (warns and continues)

## License

This is a minimal example project. Feel free to modify and extend as needed.

