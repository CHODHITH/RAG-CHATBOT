import os
import streamlit as st
from document_loader import load_pdf_documents
from rag_ppipipeline import RAGPipeline

st.set_page_config(
    page_title="Domain-Specific RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()

rag = get_rag_pipeline()

st.title("📚 Domain-Specific RAG Chatbot for PDF Question Answering")
st.markdown("""
Upload corporate policy documents, research notes, or manuals, and ask questions in natural language. 
The system retrieves relevant passages and generates grounded answers with source citations.
""")

# Sidebar for Document Management
st.sidebar.header("📁 Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Documents", 
    type=["pdf"], 
    accept_multiple_files=True
)

if "processed" not in st.session_state:
    st.session_state.processed = False

docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
os.makedirs(docs_dir, exist_ok=True)

if st.sidebar.button("Process Documents"):
    if uploaded_files:
        pdf_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(docs_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_paths.append(file_path)
        
        with st.spinner("Extracting text, chunking, and building vector embeddings..."):
            extracted = load_pdf_documents(pdf_paths)
            if extracted:
                rag.initialize_pipeline(extracted)
                st.session_state.processed = True
                st.sidebar.success(f"Successfully processed {len(extracted)} pages!")
            else:
                st.sidebar.error("No extractable text found in the uploaded PDFs.")
    else:
        # Check if default sample policy exists
        sample_pdf = os.path.join(docs_dir, "sample_policy.pdf")
        if os.path.exists(sample_pdf):
            with st.spinner("Processing default sample policy document..."):
                extracted = load_pdf_documents([sample_pdf])
                rag.initialize_pipeline(extracted)
                st.session_state.processed = True
                st.sidebar.success("Successfully processed default sample policy!")
        else:
            st.sidebar.warning("Please upload at least one PDF file.")

# Default sample loader button
if not st.session_state.processed:
    sample_pdf = os.path.join(docs_dir, "sample_policy.pdf")
    if os.path.exists(sample_pdf):
        if st.sidebar.button("Load Default Sample Policy"):
            extracted = load_pdf_documents([sample_pdf])
            rag.initialize_pipeline(extracted)
            st.session_state.processed = True
            st.sidebar.success("Loaded default sample policy (sample_policy.pdf).")

# Clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Source Citations"):
                for src in message["sources"]:
                    st.markdown(f"- **Document:** `{src['source']}` | **Page:** `{src['page']}`")

prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    if not st.session_state.processed:
        st.warning("Please process documents first using the sidebar button or load the default sample policy.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating grounded answer..."):
                answer, sources = rag.answer_question(prompt)
                st.markdown(answer)
                if sources:
                    with st.expander("View Source Citations"):
                        for src in sources:
                            st.markdown(f"- **Document:** `{src['source']}` | **Page:** `{src['page']}` (Relevance Score: {src.get('score', 0):.4f})")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
