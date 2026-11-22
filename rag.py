"""
rag.py - core logic for the PDF question-answering assistant.
"""
import os

import faiss
import numpy as np
import ollama
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

# ---------------------------------------------------------------- settings
EMBED_MODEL = "nomic-embed-text"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude")    # "claude" or "ollama"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3
SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context from a document. "
    "If the context contains information relevant to the question, use it to answer, "
    "even if it only partly answers the question. "
    "For comparison questions, describe each item using what the context says about it. "
    "Only if the context has nothing relevant, say you couldn't find it in the document. "
    "Do not use outside knowledge. "
    "Do not mention page numbers, excerpts, or file names; "
    "the app shows the sources separately. "
    "Be concise."
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)


# ------------------------------------------------- 1. extract text per page
def extract_pages(file):
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\uf0b7", "-").strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


# --------------------------------------------------------- 2. make chunks
def make_chunks(pages, filename):
    chunks = []
    for p in pages:
        for j, piece in enumerate(splitter.split_text(p["text"])):
            chunks.append({
                "text": piece,
                "page": p["page"],
                "chunk_id": f"p{p['page']}-c{j}",
                "source": filename,
            })
    return chunks


# --------------------------------------------------------- 3. embeddings
def embed(texts, prefix=""):
    out = []
    for i in range(0, len(texts), 32):
        batch = [prefix + t for t in texts[i:i + 32]]
        resp = ollama.embed(model=EMBED_MODEL, input=batch)
        out.extend(resp["embeddings"])
    vecs = np.array(out, dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


# --------------------------------------------------------- 4. FAISS index
def build_index(chunks):
    vectors = embed([c["text"] for c in chunks], prefix="search_document: ")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


# ------------------------------------------------------- 5. retrieve top-k
def retrieve(question, index, chunks, k=TOP_K):
    q = embed([question], prefix="search_query: ")
    scores, ids = index.search(q, k)
    return [
        {**chunks[i], "score": float(s)}
        for s, i in zip(scores[0], ids[0])
        if i != -1
    ]


# ------------------------------------------------------ 6. generate answer
def _build_user_message(question, retrieved):
    context = "\n\n".join(
        f"Excerpt {n}:\n{c['text']}" for n, c in enumerate(retrieved, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"
def _answer_claude(user_message):
    import anthropic

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not found. Add it to the .env file.")
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return msg.content[0].text


def _answer_ollama(user_message):
    resp = ollama.chat(
        model=OLLAMA_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={"temperature": 0.1, "num_ctx": 2048, "num_predict": 300},
        keep_alive="30m",
    )
def answer(question, retrieved):
    user_message = _build_user_message(question, retrieved)
    if LLM_PROVIDER == "ollama":
        return _answer_ollama(user_message)
    return _answer_claude(user_message)


def answer_stream(question, retrieved):
    """Yields the answer piece by piece (used by the Streamlit app)."""
    user_message = _build_user_message(question, retrieved)
    if LLM_PROVIDER == "ollama":
        stream = ollama.chat(
            model=OLLAMA_CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": 0.1, "num_ctx": 2048, "num_predict": 300},
            keep_alive="30m",
            stream=True,
        )
        for part in stream:
            yield part["message"]["content"]
    else:
        yield _answer_claude(user_message)

# ------------------------------------------- quick test: python rag.py file.pdf
if __name__ == "__main__":
    import sys
    import time

    path = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"
    pages = extract_pages(path)
    chunks = make_chunks(pages, os.path.basename(path))
    print(f"{len(pages)} pages with text, {len(chunks)} chunks")
    if not chunks:
        sys.exit("No extractable text (scanned PDF?).")

    index = build_index(chunks)
    print("Index built. Type a question (empty line to quit).")
    while True:
        q = input("\nQuestion: ").strip()
        if not q:
            break
        t0 = time.time()
        hits = retrieve(q, index, chunks)
        t1 = time.time()
        for h in hits:
            print(f"  [{h['chunk_id']}] page {h['page']} score {h['score']:.2f}: {h['text'][:100]!r}")
        ans = answer(q, hits)
        t2 = time.time()
        print("\nAnswer:", ans)
        print(f"\n(retrieval {t1 - t0:.1f}s, generation {t2 - t1:.1f}s)")