import streamlit as st

from rag import RAGIndex

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄")
st.title("📄 Chat with your documents")

if "rag" not in st.session_state:
    st.session_state.rag = None
    st.session_state.history = []

with st.sidebar:
    st.header("1. Upload documents")
    uploads = st.file_uploader("PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Process documents", disabled=not uploads):
        with st.spinner("Chunking and embedding..."):
            rag = RAGIndex()
            rag.build([(u.name, u.read()) for u in uploads])
            st.session_state.rag = rag
            st.session_state.history = []
        st.success(f"Indexed {len(rag.chunks)} chunks.")

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

question = st.chat_input("Ask a question about your documents")
if question:
    if st.session_state.rag is None:
        st.warning("Upload and process documents first (sidebar).")
    else:
        st.chat_message("user").write(question)
        with st.spinner("Thinking..."):
            answer, hits = st.session_state.rag.answer(question)
        st.chat_message("assistant").write(answer)
        with st.expander("Sources used"):
            for src, text, score in hits:
                st.caption(f"{src} (similarity {score:.2f})")
                st.write(text)
        st.session_state.history += [("user", question), ("assistant", answer)]
