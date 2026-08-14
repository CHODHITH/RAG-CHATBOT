import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

class VectorStoreManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len
        )

    def create_vector_store(self, extracted_docs):
        """
        Takes extracted documents (list of dicts with text, source, page)
        splits them into chunks, and creates a FAISS vector store.
        """
        langchain_docs = []
        for doc in extracted_docs:
            chunks = self.text_splitter.split_text(doc["text"])
            for chunk in chunks:
                langchain_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": doc["source"], "page": doc["page"]}
                    )
                )
        
        if not langchain_docs:
            return None

        self.vector_store = FAISS.from_documents(langchain_docs, self.embeddings)
        return self.vector_store

    def save_local(self, folder_path=None):
        if folder_path is None:
            folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_store", "saved_index")
        if self.vector_store:
            os.makedirs(folder_path, exist_ok=True)
            self.vector_store.save_local(folder_path)

    def load_local(self, folder_path=None):
        if folder_path is None:
            folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_store", "saved_index")
        if os.path.exists(folder_path):
            self.vector_store = FAISS.load_local(
                folder_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            return self.vector_store
        return None

    def similarity_search(self, query, k=4):
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search_with_score(query, k=k)
