"""
app.py - Streamlit UI for the PDF question-answering assistant.
Run with:  streamlit run app.py
"""
import hashlib
import io

import streamlit as st

import rag

LOW_SCORE = 0.55

st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📄")
st.title("📄 PDF Question-Answering Assistant")
st.caption(
    f"Embeddings: Ollama ({rag.EMBED_MODEL}) · Vector store: FAISS · "
    f"LLM: Ollama ({rag.OLLAMA_CHAT_MODEL})"
)
@st.cache_resource(show_spinner=False)
def _warm_up():
    rag.warm_up()
    return True


_warm_up()

def show_sources(sources):
    st.subheader("Sources")
    for c in sources:
        label = f"{c['source']} · page {c['page']} · {c['chunk_id']} (score {c['score']:.2f})"
        with st.expander(label):
            st.write(c["text"])


# ------------------------------------------------------------- 1. upload
uploaded = st.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=False)

if uploaded is None:
    st.info("Upload a PDF to get started.")
    st.stop()

# ------------------------------------- 2-5. index the file (once per file)
file_bytes = uploaded.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()

if st.session_state.get("file_hash") != file_hash:
    st.session_state.pop("index", None)
    st.session_state.messages = []

    with st.spinner("Reading PDF, chunking, and building the index..."):
        try:
            pages = rag.extract_pages(io.BytesIO(file_bytes))
            if not pages:
                st.error("No extractable text found. This may be a scanned PDF (it needs OCR).")
                st.stop()
            chunks = rag.make_chunks(pages, uploaded.name)
            index = rag.build_index(chunks)
        except Exception as e:
            st.error(f"Could not process the PDF: {e}")
            st.info("If this mentions a connection error, make sure Ollama is running.")
            st.stop()

    st.session_state.update(
        file_hash=file_hash, index=index, chunks=chunks, n_pages=len(pages)
    )

st.success(
    f"**{uploaded.name}** is ready: {st.session_state.n_pages} pages with text, "
    f"{len(st.session_state.chunks)} chunks."
)

# ------------------------------------------------------ chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

# ------------------------------------------ 6-10. ask, retrieve, answer
question = st.chat_input("Ask a question about the document")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document..."):
                retrieved = rag.retrieve(
                    question, st.session_state.index, st.session_state.chunks
                )
            reply = st.write_stream(rag.answer_stream(question, retrieved))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

        if retrieved and retrieved[0]["score"] < LOW_SCORE:
            st.warning("Low similarity scores: the document may not cover this question.")
        show_sources(retrieved)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": retrieved}
    )