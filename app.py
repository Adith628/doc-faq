"""
Mini Doc-FAQ Agent (RAG + two tiny agents)
A minimal Streamlit RAG app with Indexer and Answerer agents.
"""

import os
import glob
import re
import textwrap
from typing import Optional

import numpy as np
import streamlit as st

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, continue without it
    pass

# Provider toggle: "openai" or "gemini"
PROVIDER = "gemini"

# Conditional imports based on provider
if PROVIDER == "openai":
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        st.error("Missing dependency: pip install openai")
        st.stop()
elif PROVIDER == "gemini":
    try:
        import google.generativeai as genai
    except ImportError:
        st.error("Missing dependency: pip install google-generativeai")
        st.stop()
else:
    st.error(f"Invalid PROVIDER: {PROVIDER}. Must be 'openai' or 'gemini'")
    st.stop()

try:
    import faiss
except ImportError:
    st.error("Missing dependency: pip install faiss-cpu")
    st.stop()

# Agent system prompts (as constants)
INDEXER_SYSTEM_PROMPT = (
    "You are the Indexer. Given raw documents, split them into coherent chunks "
    "(300–800 tokens), preserving headings and lists. Produce clean chunks ready "
    "for retrieval; do not summarize."
)

ANSWERER_SYSTEM_PROMPT = (
    "You are the Answerer. Answer strictly from retrieved chunks. If context is "
    "insufficient, say so. Use short bullets, and include inline citations like "
    "[1], [2] referring to the provided chunks."
)


def chunk_text(text: str, max_tokens: int = 600, overlap: int = 120) -> list[str]:
    """
    Split text into chunks by sentences with approximate token-based sizing.
    
    Args:
        text: Input text to chunk
        max_tokens: Approximate max words per chunk
        overlap: Approximate overlap in words between chunks
    
    Returns:
        List of text chunks
    """
    # Simple sentence splitting by common delimiters
    sentences = re.split(r'[.!?]\s+', text)
    print(f"[Indexer] Splitting text into sentences: {len(sentences)} sentences found")
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        
        words = sentence.split()
        word_count = len(words)
        
        if current_word_count + word_count > max_tokens and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text.strip())
            
            # Start new chunk with overlap: take sentences from end until we have ~overlap words
            overlap_sentences = []
            overlap_word_count = 0
            for sent in reversed(current_chunk):
                sent_words = len(sent.split())
                if overlap_word_count + sent_words <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_word_count += sent_words
                else:
                    break
            
            current_chunk = overlap_sentences + [sentence]
            current_word_count = overlap_word_count + word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text.strip())
    
    # Filter empty chunks
    filtered_chunks = [c for c in chunks if c]
    print(f"[Indexer] Created {len(filtered_chunks)} chunks from text")
    return filtered_chunks


def load_docs(folder: str = "data") -> tuple[list[str], list[dict]]:
    """
    Load and chunk all .txt and .md files from the specified folder.
    
    Args:
        folder: Path to folder containing documents
    
    Returns:
        Tuple of (chunks, metadata) where metadata is list of dicts with 'source' and 'chunk_id'
    """
    chunks = []
    metadata = []
    
    if not os.path.exists(folder):
        print(f"[Indexer] Folder '{folder}' does not exist")
        return chunks, metadata
    
    # Find all .txt and .md files
    txt_files = glob.glob(os.path.join(folder, "*.txt"))
    md_files = glob.glob(os.path.join(folder, "*.md"))
    all_files = txt_files + md_files
    print(f"[Indexer] Found {len(all_files)} files ({len(txt_files)} .txt, {len(md_files)} .md)")
    
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_chunks = chunk_text(content)
            filename = os.path.basename(file_path)
            print(f"[Indexer] Processed '{filename}': {len(file_chunks)} chunks created")
            
            for i, chunk in enumerate(file_chunks):
                chunks.append(chunk)
                metadata.append({
                    'source': filename,
                    'chunk_id': i
                })
        except Exception as e:
            print(f"[Indexer] ERROR reading {file_path}: {e}")
            st.warning(f"Error reading {file_path}: {e}")
            continue
    
    print(f"[Indexer] Total chunks loaded: {len(chunks)} from {len(set(m['source'] for m in metadata))} file(s)")
    return chunks, metadata


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """
    Generate embeddings for a list of texts using the configured provider.
    
    Args:
        texts: List of text strings to embed
    
    Returns:
        List of numpy arrays (float32) representing embeddings
    """
    print(f"[Indexer] Generating embeddings for {len(texts)} chunks using {PROVIDER}...")
    if PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        client = OpenAI(api_key=api_key)
        
        # Batch embeddings (OpenAI handles batching internally)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        
        embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        print(f"[Indexer] Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")
        return embeddings
    
    elif PROVIDER == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        
        # Gemini embeddings
        embeddings = []
        for i, text in enumerate(texts, 1):
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(np.array(result['embedding'], dtype=np.float32))
            if i % 10 == 0 or i == len(texts):
                print(f"[Indexer] Embedded {i}/{len(texts)} chunks...")
        
        print(f"[Indexer] Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")
        return embeddings
    
    else:
        raise ValueError(f"Unknown provider: {PROVIDER}")


def embed_query(query: str) -> np.ndarray:
    """
    Generate embedding for a single query string.
    
    Args:
        query: Query text
    
    Returns:
        Numpy array (float32) representing the query embedding
    """
    print(f"[Answerer] Embedding query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    if PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    elif PROVIDER == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        embedding = np.array(result['embedding'], dtype=np.float32)
        print(f"[Answerer] Query embedding generated (dimension: {len(embedding)})")
        return embedding
    
    else:
        raise ValueError(f"Unknown provider: {PROVIDER}")


def llm_chat(prompt: str) -> str:
    """
    Generate a response using the configured LLM provider.
    
    Args:
        prompt: Full prompt including system and user messages
    
    Returns:
        Generated text response
    """
    print(f"[Answerer] LLM call: provider={PROVIDER}, prompt_length={len(prompt)}")
    if PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ANSWERER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    
    elif PROVIDER == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        # Use gemini-2.0-flash (gemini-1.5-flash has been deprecated)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Combine system prompt with user prompt
        full_prompt = f"{ANSWERER_SYSTEM_PROMPT}\n\n{prompt}"
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        print(f"[Answerer] LLM response received ({len(response.text)} characters)")
        return response.text
    
    else:
        raise ValueError(f"Unknown provider: {PROVIDER}")


@st.cache_data
def build_index() -> tuple[Optional[faiss.Index], list[str], list[dict]]:
    """
    Build FAISS index from documents in /data folder.
    Cached to avoid rebuilding on every rerun.
    
    Returns:
        Tuple of (index, chunks, metadata). Index is None if no documents found.
    """
    print("[Indexer] ===== Starting index build =====")
    chunks, metadata = load_docs("data")
    
    if not chunks:
        print("[Indexer] No chunks found, returning None")
        return None, [], []
    
    # Generate embeddings
    with st.spinner("Generating embeddings..."):
        embeddings = embed_texts(chunks)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    print(f"[Indexer] Embeddings array shape: {embeddings_array.shape}")
    
    # Normalize for cosine similarity (L2 normalization)
    faiss.normalize_L2(embeddings_array)
    print("[Indexer] Normalized embeddings for cosine similarity")
    
    # Build FAISS index (IndexFlatIP for inner product = cosine on normalized vectors)
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    print(f"[Indexer] FAISS index built: {index.ntotal} vectors, dimension {dimension}")
    print("[Indexer] ===== Index build complete =====")
    
    return index, chunks, metadata


def retrieve(index: faiss.Index, texts: list[str], meta: list[dict], 
             query: str, k: int = 4) -> list[dict]:
    """
    Retrieve top-k chunks for a query.
    
    Args:
        index: FAISS index
        texts: List of chunk texts
        meta: List of metadata dicts
        query: Query string
        k: Number of results to return
    
    Returns:
        List of dicts with 'rank', 'score', 'text', 'meta' keys
    """
    print(f"[Answerer] ===== Starting retrieval for query =====")
    print(f"[Answerer] Query: '{query}'")
    print(f"[Answerer] Retrieving top {k} chunks from {len(texts)} total chunks")
    
    # Embed query
    query_embedding = embed_query(query)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Normalize query embedding
    faiss.normalize_L2(query_embedding)
    print("[Answerer] Query embedding normalized")
    
    # Search
    scores, indices = index.search(query_embedding, min(k, len(texts)))
    print(f"[Answerer] Search completed, found {len(scores[0])} results")
    
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx < len(texts):
            source = meta[idx].get('source', 'Unknown')
            print(f"[Answerer] Rank {rank}: score={score:.4f}, source={source}, chunk_id={meta[idx].get('chunk_id', 'N/A')}")
            results.append({
                'rank': rank,
                'score': float(score),
                'text': texts[idx],
                'meta': meta[idx]
            })
    
    print(f"[Answerer] ===== Retrieval complete: {len(results)} results =====")
    return results


def answer_with_citations(query: str, contexts: list[dict]) -> str:
    """
    Generate an answer with inline citations from retrieved contexts.
    
    Args:
        query: User question
        contexts: List of retrieved context dicts with 'text' and 'meta'
    
    Returns:
        Formatted answer string with citations
    """
    print(f"[Answerer] ===== Generating answer =====")
    print(f"[Answerer] Query: '{query}'")
    print(f"[Answerer] Using {len(contexts)} context chunks")
    
    if not contexts:
        print("[Answerer] No contexts provided, returning default message")
        return "I don't have enough information to answer this question."
    
    # Build context string with numbered citations
    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        source = ctx['meta'].get('source', 'Unknown')
        text = ctx['text']
        context_parts.append(f"[{i}] Source: {source}\n{text}")
        print(f"[Answerer] Context {i}: {source} (chunk {ctx['meta'].get('chunk_id', 'N/A')}, {len(text)} chars)")
    
    context_str = "\n\n".join(context_parts)
    total_context_length = sum(len(ctx['text']) for ctx in contexts)
    print(f"[Answerer] Total context length: {total_context_length} characters")
    
    # Compose prompt
    prompt = f"""Based on the following retrieved contexts, answer the question. 
Use 2-5 bullet points and a one-line summary. Include inline citations like [1], [2] 
referring to the numbered contexts. If the information is insufficient, say so.

Question: {query}

Retrieved contexts:
{context_str}

Answer:"""
    
    print(f"[Answerer] Calling LLM ({PROVIDER}) to generate answer...")
    try:
        answer = llm_chat(prompt)
        print(f"[Answerer] Answer generated ({len(answer)} characters)")
        print("[Answerer] ===== Answer generation complete =====")
        return answer
    except Exception as e:
        print(f"[Answerer] ERROR generating answer: {e}")
        return f"Error generating answer: {e}"


# Streamlit UI
st.set_page_config(
    page_title="Mini Doc-FAQ Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Mini Doc-FAQ Agent")
st.caption("RAG-powered Q&A with Indexer and Answerer agents")

with st.expander("📖 Instructions", expanded=False):
    st.markdown("""
    1. Place `.txt` or `.md` files in the `/data` folder
    2. The Indexer will automatically process and index them
    3. Ask questions about your documents
    4. The Answerer will retrieve relevant chunks and provide answers with citations
    """)

# Check API key
if PROVIDER == "openai":
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not set. Please set it in the `.env` file or as an environment variable.")
        st.info("Add to `.env` file: `OPENAI_API_KEY=your_key`\n\nOr set as environment variable:\nWindows: `setx OPENAI_API_KEY your_key`\nUnix/Mac: `export OPENAI_API_KEY=your_key`")
        st.stop()
elif PROVIDER == "gemini":
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ GOOGLE_API_KEY not set. Please set it in the `.env` file or as an environment variable.")
        st.info("Add to `.env` file: `GOOGLE_API_KEY=your_key`\n\nOr set as environment variable:\nWindows: `setx GOOGLE_API_KEY your_key`\nUnix/Mac: `export GOOGLE_API_KEY=your_key`")
        st.stop()

# Build index
index, chunks, metadata = build_index()

if index is None:
    st.error("📁 No documents found in `/data` folder.")
    st.info("Add `.txt` or `.md` files to the `/data` folder and refresh the page.")
    st.stop()

st.success(f"✅ Index ready: {len(chunks)} chunks from {len(set(m['source'] for m in metadata))} file(s)")

# Query interface
st.divider()
query = st.text_input("Ask a question about your docs:", placeholder="e.g., What is the main topic?")

if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            results = retrieve(index, chunks, metadata, query, k=4)
        
        if results:
            st.subheader("📚 Retrieved Chunks")
            for result in results:
                with st.container():
                    st.markdown(f"**Rank {result['rank']}** | Source: `{result['meta']['source']}` | Score: {result['score']:.4f}")
                    # Shorten text for display
                    shortened = textwrap.shorten(result['text'], width=600, placeholder="...")
                    st.code(shortened, language=None)
            
            st.divider()
            st.subheader("💡 Answer")
            
            with st.spinner("Generating answer..."):
                answer = answer_with_citations(query, results)
            
            st.markdown(answer)
        else:
            st.warning("No results found.")

