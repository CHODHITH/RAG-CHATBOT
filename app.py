import os
import streamlit as st
from document_loader import load_pdf_documents
from rag_pipeline import RAGPipeline

st.set_page_config(page_title='Domain-Specific RAG Chatbot', page_icon='📚', layout='wide')

@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()

rag = get_rag_pipeline()

st.title('📚 Domain-Specific RAG Chatbot')
st.write('Upload PDF documents and ask questions based on their content.')

if 'processed' not in st.session_state:
    st.session_state.processed = False

if 'messages' not in st.session_state:
    st.session_state.messages = []

docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'documents')
os.makedirs(docs_dir, exist_ok=True)

st.sidebar.header('Document Upload')

uploaded_files = st.sidebar.file_uploader(
    'Upload PDF files',
    type='pdf',
    accept_multiple_files=True
)

if st.sidebar.button('Process Documents'):
    pdf_paths = []

    if uploaded_files:
        for file in uploaded_files:
            path = os.path.join(docs_dir, file.name)
            with open(path, 'wb') as f:
                f.write(file.getbuffer())
            pdf_paths.append(path)
    else:
        sample_pdf = os.path.join(docs_dir, 'sample.pdf')
        if os.path.exists(sample_pdf):
            pdf_paths.append(sample_pdf)

    if pdf_paths:
        with st.spinner('Processing documents...'):
            documents = load_pdf_documents(pdf_paths)
            rag.initialize_pipeline(documents)
            st.session_state.processed = True
            st.sidebar.success('Documents processed successfully')
    else:
        st.sidebar.warning('Please upload a PDF file')

if st.sidebar.button('Clear Chat'):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('sources'):
            with st.expander('Sources'):
                for src in message['sources']:
                    st.write(f"{src['source']} - Page {src['page']}")
prompt = st.chat_input('Ask a question')

if prompt:
    if not st.session_state.processed:
        st.warning('Please process documents first')
    else:
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt
        })

        with st.chat_message('user'):
            st.markdown(prompt)

        with st.chat_message('assistant'):
            with st.spinner('Generating answer...'):
                answer, sources = rag.answer_question(prompt)
                st.markdown(answer)

                if sources:
                    with st.expander('Sources'):
                        for src in sources:
                            st.write(f"{src['source']} - Page {src['page']}")
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': answer,
                    'sources': sources
                })
