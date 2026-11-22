# PDF Question-Answering Assistant (RAG)

Upload a PDF and ask questions about it. The app retrieves the 3 most relevant
chunks, generates an answer with a local LLM, and shows the source file, page,
and chunk for every answer. Everything runs locally with Ollama, so no API key
is needed.

## How it works

Indexing (once per PDF):

PDF → text per page → chunks → embeddings (Ollama, nomic-embed-text) → FAISS index

Querying (every question):

Question → embedding → top-3 chunks from FAISS → LLM (Ollama, llama3.2:3b) → answer + sources

## Tech stack

| Part | Tool |
|---|---|
| Interface | Streamlit |
| PDF text extraction | pypdf |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (800 characters, 150 overlap) |
| Embeddings | Ollama, `nomic-embed-text` |
| Vector store | FAISS (`IndexFlatIP`, cosine similarity) |
| Answer LLM | Ollama, `llama3.2:3b` |

## Setup (Windows)

1. Install [Ollama](https://ollama.com) and download the two models:
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.2:3b
   ```
2. Create a virtual environment and install the dependencies:
   ```
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. (Optional) Create a `.env` file to choose a different chat model:
   ```
   OLLAMA_CHAT_MODEL=llama3.2:3b
   ```
   Without this file the app uses `llama3.2:3b`.
4. Make sure Ollama is running, then start the app:
   ```
   streamlit run app.py
   ```
   It opens at http://localhost:8501.

## Usage

1. Upload one PDF. The app extracts the text, splits it into chunks, and builds the index.
2. Type a question in the chat box.
3. Read the streamed answer, then open the **Sources** expanders to see the
   file name, page, chunk ID, similarity score, and the chunk text.
4. If the best similarity score is low, the app warns that the document may not
   cover the question.

To test the pipeline without the web interface:

```
python rag.py yourfile.pdf
```

## Project files

- `rag.py`: extraction, chunking, embeddings, FAISS index, retrieval, LLM answer
- `app.py`: Streamlit interface
- `requirements.txt`: Python dependencies

## Limitations

- Works only on PDFs with a text layer. Scanned PDFs and image-only slides need OCR.
- Tables, charts, and diagrams are not interpreted.
- The small local model can blend facts or phrase answers loosely, and it is
  slow on a CPU-only laptop. A larger Ollama model (for example `llama3.1:8b`)
  improves accuracy.
- One document at a time, and the index is kept in memory (rebuilt on each upload).