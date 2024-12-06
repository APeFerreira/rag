# from src.retriever import Retriever

# def test_retriever():
#     retriever = Retriever()
#     query = "What is the legal age to purchase alcohol in the UK?"
#     top_k = 5
#     chunks, doc_ids = retriever.retrieve(query, top_k)
    
#     assert len(chunks) == top_k
#     assert len(doc_ids) == top_k
#     assert all(isinstance(chunk, str) for chunk in chunks)
#     assert all(isinstance(doc_id, str) for doc_id in doc_ids)
    
#     print("Retriever tests pass")

# src/test_retriever.py

import sys
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from src.retriever import Retriever
import pandas as pd

def load_retriever():
    """
    Initialize and return the Retriever instance.
    """
    retriever = Retriever()
    return retriever

def load_preprocessed_dataframe(preprocessed_csv_path):
    """
    Load the preprocessed dataframe to map case IDs to original data.
    """
    df = pd.read_csv(preprocessed_csv_path)
    return df

def test_query(retriever, df, query, top_k=5):
    """
    Retrieve and display the top_k documents for a given query.
    """
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")
    
    retrieved_chunks, retrieved_doc_ids = retriever.retrieve(query, top_k=top_k)
    
    for i, (chunk, doc_id) in enumerate(zip(retrieved_chunks, retrieved_doc_ids), 1):
        # Fetch the case title from the dataframe
        case_title = df.loc[df['case_id'] == doc_id, 'case_title'].values
        case_title = case_title[0] if len(case_title) > 0 else "N/A"
        
        print(f"Result {i}:")
        print(f"Case ID: {doc_id}")
        print(f"Case Title: {case_title}")
        print(f"Chunk: {chunk}\n")
        print(f"{'-'*80}\n")

def main():
    """
    Main function to execute the retriever tests.
    """
    # Define paths
    preprocessed_csv_path = 'data/legal_documents/preprocessed_dataframe.csv'
       
    # Load Retriever
    print("Loading Retriever...")
    retriever = load_retriever()
    print("Retriever loaded successfully.")
    
    # Load preprocessed dataframe
    print("Loading preprocessed dataframe...")
    df = load_preprocessed_dataframe(preprocessed_csv_path)
    print("Dataframe loaded successfully.")
    
    # Define sample queries
    sample_queries = [
        "Under what circumstances can a court issue cost orders without proceeding to trial?",# according to Australian Securities Commission v Aust-Home Investments Limited?",
        "When are indemnity costs awarded instead of party and party costs in court proceedings?", #based on Alpine Hardwood Pty Ltd v Hardys Pty Ltd?
        "What criteria are used to assess apparent bias in judicial decisions?", #as outlined in Johnson v Johnson and Sydney Refractive Surgery Centre Pty Ltd v Federal Commissioner of Taxation?
        "How does intellectual property law protect inventions?",
        "What is the process for filing a lawsuit in civil court?"
    ]
    
    # Test each query
    for query in sample_queries:
        test_query(retriever, df, query, top_k=5)

if __name__ == "__main__":
    main()
