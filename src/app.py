# src/app.py
#streamlit run src/app.py

import streamlit as st
from rag_system import RAGSystem

def main():
    st.title("Legal Question Answering System")
    st.write("Ask any legal question, and the system will provide an answer based on relevant legal cases.")

    question = st.text_input("Enter your legal question:")

    if st.button("Get Answer"):
        if question:
            rag = RAGSystem()
            with st.spinner('Fetching answer...'):
                answer = rag.answer_question(question, top_k=5)
            st.success("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
