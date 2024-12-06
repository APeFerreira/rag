# src/rag_system.py

from retriever import Retriever
from retriever import (
    DATA_DIR,
    INDEX_FILE,
    DOCS_FILE
)
import os
from generator import Generator
import pandas as pd

PREPROCESSED_CSV_FILE = os.path.join(DATA_DIR, "preprocessed_dataframe.csv") 

class RAGSystem:
    def __init__(
            self, 
            index_path : str = INDEX_FILE,
            docs_path : str = DOCS_FILE,
            preprocessed_csv_path : str = PREPROCESSED_CSV_FILE
        ):
        """
        Initialize the RAG system by loading the retriever and generator components.
        """
        assert isinstance(preprocessed_csv_path, str), "Preprocessed CSV path must be a string"
        assert os.path.exists(preprocessed_csv_path), f"Preprocessed CSV file not found at {preprocessed_csv_path}"

        self.retriever = Retriever(index_path=index_path, docs_path=docs_path)
        self.generator = Generator()
        
        # Load the preprocessed dataframe to map case IDs to original data
        self.df = pd.read_csv(preprocessed_csv_path)
    
    def get_context_from_chunks(
            self,
            doc_ids
        ):
        """
        Retrieve the full case texts corresponding to the provided case IDs.
        """
        # Get the unique case IDs
        unique_case_ids = list(set(doc_ids))
        
        # Fetch the corresponding cleaned titles and texts
        cases = self.df[self.df['case_id'].isin(unique_case_ids)]
        context_list = cases['cleaned_text'].tolist()
        
        # Combine the contexts
        combined_context = ' '.join(context_list)
        return combined_context
    
    def answer_question(self, question, top_k=5):
        """
        Generate an answer to the question using retrieved context.
        """
        # Retrieve relevant chunks and their case IDs
        retrieved_chunks, retrieved_doc_ids = self.retriever.retrieve(question, top_k=top_k)
        
        # For simplicity, combine the retrieved chunks as context
        context = ' '.join(retrieved_chunks)
        
        # Alternatively, retrieve full case texts if needed
        # context = self.get_context_from_chunks(retrieved_doc_ids)
        
        # Generate the answer
        answer = self.generator.generate_answer(question, context)
        return answer
