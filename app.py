"""
Mini Doc-FAQ Agent (RAG + two tiny agents)
A minimal Streamlit RAG app with Indexer and Answerer agents.
"""

import os
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


def process_uploaded_file(uploaded_file) -> tuple[list[str], list[dict]]:
    """
    Process an uploaded file and chunk it.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Tuple of (chunks, metadata) where metadata is list of dicts with 'source' and 'chunk_id'
    """
    if uploaded_file is None:
        print(f"[Indexer] No file uploaded")
        return [], []
    
    try:
        # Read file content
        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
            content = str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "text/markdown" or uploaded_file.name.endswith('.md'):
            content = str(uploaded_file.read(), "utf-8")
        else:
            # Try to read as text anyway
            content = str(uploaded_file.read(), "utf-8")
        
        filename = uploaded_file.name
        print(f"[Indexer] Processing uploaded file '{filename}' ({len(content)} characters)")
        
        chunks = []
        metadata = []
        
        # Chunk the text
        file_chunks = chunk_text(content)
        print(f"[Indexer] Processed '{filename}': {len(file_chunks)} chunks created")
        
        for i, chunk in enumerate(file_chunks):
            chunks.append(chunk)
            metadata.append({
                'source': filename,
                'chunk_id': i
            })
        
        print(f"[Indexer] Total chunks created: {len(chunks)}")
        return chunks, metadata
    
    except Exception as e:
        print(f"[Indexer] ERROR processing file: {e}")
        raise


def process_text_input(text: str, source_name: str = "User Input") -> tuple[list[str], list[dict]]:
    """
    Process user-provided text input and chunk it.
    
    Args:
        text: Input text to process
        source_name: Name to use as source in metadata
    
    Returns:
        Tuple of (chunks, metadata) where metadata is list of dicts with 'source' and 'chunk_id'
    """
    if not text or not text.strip():
        print(f"[Indexer] No text provided")
        return [], []
    
    print(f"[Indexer] Processing user input text ({len(text)} characters)")
    chunks = []
    metadata = []
    
    # Chunk the text
    file_chunks = chunk_text(text)
    print(f"[Indexer] Processed '{source_name}': {len(file_chunks)} chunks created")
    
    for i, chunk in enumerate(file_chunks):
        chunks.append(chunk)
        metadata.append({
            'source': source_name,
            'chunk_id': i
        })
    
    print(f"[Indexer] Total chunks created: {len(chunks)}")
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
def build_index(text_input: str = None, uploaded_file_content: tuple = None) -> tuple[Optional[faiss.Index], list[str], list[dict]]:
    """
    Build FAISS index from user-provided text input or uploaded file.
    Cached based on input to avoid rebuilding when input hasn't changed.
    
    Args:
        text_input: User-provided text to index (optional)
        uploaded_file_content: Tuple of (filename, content) from uploaded file (optional)
    
    Returns:
        Tuple of (index, chunks, metadata). Index is None if no documents found.
    """
    print("[Indexer] ===== Starting index build =====")
    
    # Prioritize uploaded file if both are provided
    if uploaded_file_content:
        filename, content = uploaded_file_content
        chunks, metadata = process_text_input(content, source_name=filename)
    elif text_input:
        chunks, metadata = process_text_input(text_input)
    else:
        print("[Indexer] No input provided")
        return None, [], []
    
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
    1. **Upload a file** (`.txt` or `.md`) or **paste text** in the text area
    2. Click **"Index Document"** to process and index the content
    3. Ask questions about your document
    4. The Answerer will retrieve relevant chunks and provide answers with citations
    
    💡 **Tip**: If both file and text are provided, the uploaded file takes priority.
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

# Initialize session state
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'index_built' not in st.session_state:
    st.session_state.index_built = False
if 'index' not in st.session_state:
    st.session_state.index = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'metadata' not in st.session_state:
    st.session_state.metadata = []

# Document input section
st.subheader("📄 Document Input")

# Create tabs for file upload and text input
tab1, tab2 = st.tabs(["📁 Upload File", "✍️ Paste Text"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload a document file",
        type=['txt', 'md'],
        help="Supported formats: .txt and .md files"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.info(f"📄 File loaded: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")

with tab2:
    text_input = st.text_area(
        "Enter your document text:",
        value=st.session_state.text_input,
        height=200,
        placeholder="Paste or type your document content here...",
        help="You can paste text from any source. The Indexer will automatically chunk and process it."
    )
    st.session_state.text_input = text_input

# Index button
st.divider()
col1, col2 = st.columns([1, 4])
with col1:
    index_button = st.button("Index Document", type="primary")

# Build index when button is clicked
if index_button:
    # Prioritize uploaded file if both are provided
    if st.session_state.uploaded_file is not None:
        try:
            # Read file content
            if st.session_state.uploaded_file.type == "text/plain" or st.session_state.uploaded_file.name.endswith('.txt'):
                file_content = str(st.session_state.uploaded_file.read(), "utf-8")
            else:
                file_content = str(st.session_state.uploaded_file.read(), "utf-8")
            
            # Reset file pointer for potential re-read
            st.session_state.uploaded_file.seek(0)
            
            with st.spinner("Building index from uploaded file..."):
                index, chunks, metadata = build_index(
                    uploaded_file_content=(st.session_state.uploaded_file.name, file_content)
                )
                if index is not None:
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.metadata = metadata
                    st.session_state.index_built = True
                    st.success(f"✅ Index ready: {len(chunks)} chunks created from '{st.session_state.uploaded_file.name}'")
                else:
                    st.error("Failed to build index. Please check your file.")
                    st.session_state.index_built = False
        except Exception as e:
            st.error(f"Error processing file: {e}")
            print(f"[Indexer] ERROR: {e}")
            st.session_state.index_built = False
    
    elif text_input and text_input.strip():
        with st.spinner("Building index from text input..."):
            index, chunks, metadata = build_index(text_input=text_input)
            if index is not None:
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.metadata = metadata
                st.session_state.index_built = True
                st.success(f"✅ Index ready: {len(chunks)} chunks created")
            else:
                st.error("Failed to build index. Please check your input.")
                st.session_state.index_built = False
    else:
        st.warning("Please upload a file or enter some text to index.")
        st.session_state.index_built = False

# Show status if index is already built
if st.session_state.index_built and st.session_state.index is not None:
    st.info(f"📊 Current index: {len(st.session_state.chunks)} chunks ready for querying")

# Query interface
st.divider()
st.subheader("❓ Ask Questions")

if not st.session_state.index_built or st.session_state.index is None:
    st.info("👆 Please index a document first before asking questions.")
else:
    query = st.text_input("Ask a question about your document:", placeholder="e.g., What is the main topic?")

    if st.button("Ask", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant chunks..."):
                results = retrieve(
                    st.session_state.index, 
                    st.session_state.chunks, 
                    st.session_state.metadata, 
                    query, 
                    k=4
                )
            
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

