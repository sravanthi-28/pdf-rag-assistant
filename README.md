# PDF Question-Answering Assistant (RAG)

Upload a PDF and ask questions about it. The app retrieves the 3 most relevant
chunks and generates an answer with an LLM, showing the source page and chunk.

## How it works
PDF → text per page → chunks → embeddings (Ollama, nomic-embed-text) → FAISS index

Question → embedding → top-3 chunks → LLM (Ollama or Claude) → answer + sources

## Setup
1. Install [Ollama](https://ollama.com) and pull the models:
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.2:3b
   ```
2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Create a `.env` file:
   ```
   LLM_PROVIDER=ollama
   ```
   For Claude instead, use `LLM_PROVIDER=claude` and add `ANTHROPIC_API_KEY=your_key`.
4. Run:
   ```
   streamlit run app.py
   ```

## Files
- `rag.py`: extraction, chunking, embeddings, FAISS, retrieval, LLM answer
- `app.py`: Streamlit interface

## Limitations
- Works only on PDFs with a text layer (scanned PDFs need OCR).
- Small local models may blend facts or be slow on CPU.