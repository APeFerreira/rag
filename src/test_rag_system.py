# src/test_rag_system.py

import os
import sys
from rag_system import RAGSystem
import pandas as pd

def load_rag_system(
):
    """
    Initialize and return the RAGSystem instance.
    """
    try:
        rag = RAGSystem()
        return rag
    except Exception as e:
        print(f"Error initializing RAGSystem: {e}")
        sys.exit(1)

def test_rag(rag, question, top_k=5):
    """
    Generate an answer using the RAG system and display the results.
    Optionally, display the retrieved documents.
    """
    try:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}\n")
        
        # Get the answer from the RAG system
        answer = rag.answer_question(question, top_k=top_k)
        
        # Optionally, retrieve the documents for display
        retrieved_chunks, retrieved_doc_ids = rag.retriever.retrieve(question, top_k=top_k)
        
        # Fetch the case titles from the dataframe
        case_titles = rag.df.set_index('case_id').loc[retrieved_doc_ids]['case_title'].tolist()
        
        print(f"Generated Answer:\n{answer}\n")
        print(f"Retrieved Documents:")
        for i, (doc_id, title, chunk) in enumerate(zip(retrieved_doc_ids, case_titles, retrieved_chunks), 1):
            print(f"\nResult {i}:")
            print(f"Case ID: {doc_id}")
            print(f"Case Title: {title}")
            print(f"Chunk: {chunk}\n")
            print(f"{'-'*80}\n")
    
    except Exception as e:
        print(f"Error during RAG system test: {e}")

def main():
    """
    Main function to execute the RAG system tests.
    """    
    # Initialize RAG System
    print("Loading RAG System...")
    rag = load_rag_system()
    print("RAG System loaded successfully.\n")
    
    # Define sample queries (You can customize these based on your dataset)
    sample_queries = [
        "What criteria are used to assess apparent bias in judicial decisions as outlined in Johnson v Johnson and Sydney Refractive Surgery Centre Pty Ltd v Federal Commissioner of Taxation?",
        "Under what circumstances can a court issue cost orders without proceeding to trial, according to Australian Securities Commission v Aust-Home Investments Limited?",
        "When are indemnity costs awarded instead of party and party costs in court proceedings, based on Alpine Hardwood Pty Ltd v Hardys Pty Ltd?"
    ]
    
    # Execute tests
    for query in sample_queries:
        test_rag(rag, query, top_k=5)

if __name__ == "__main__":
    main()
