# src/retriever.py

import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer


DATA_DIR = os.path.join("data", "legal_documents")
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.index")
DOCS_FILE = os.path.join(DATA_DIR, "documents.pkl")
MODEL_NAME_RETRIEVER = 'sentence-transformers/all-MiniLM-L6-v2'

class Retriever:
    def __init__(
            self, 
            index_path : str = INDEX_FILE, 
            docs_path : str = DOCS_FILE, 
            model_name : str = MODEL_NAME_RETRIEVER
        ):
        """
        Initialize the Retriever by loading the FAISS index, document chunks, and the embedding model.
        """
        assert isinstance(index_path, str), "Index path must be a string"
        assert isinstance(docs_path, str), "Docs path must be a string"
        assert isinstance(model_name, str), "Model name must be a string"
        
        assert os.path.exists(index_path), f"Index file not found at {index_path}"
        assert os.path.exists(docs_path), f"Docs file not found at {docs_path}"
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load document chunks and their corresponding case IDs
        with open(docs_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.doc_ids = data['doc_ids']
        
        # Load embedding model
        self.model = SentenceTransformer(model_name)
    
    def retrieve(
            self,
            query : str,
            top_k : int = 5
        ):
        """
        Retrieve the top_k most relevant document chunks for a given query.
        """
        assert isinstance(query, str), "Query must be a string"
        assert isinstance(top_k, int), "top_k must be an integer"

        # Generate embedding for the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Perform similarity search using FAISS
        _, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve the corresponding document chunks and case IDs
        retrieved_chunks = [self.chunks[idx] for idx in indices[0]]
        retrieved_doc_ids = [self.doc_ids[idx] for idx in indices[0]]
        
        return retrieved_chunks, retrieved_doc_ids
