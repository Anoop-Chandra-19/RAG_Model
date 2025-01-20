import streamlit as st
import requests

# Backend API URL
API_URL = "http://localhost:8000"

# Streamlit UI
st.title("Chat with a Retrieval-Augmented Generation (RAG) Model")
st.subheader("Ask your question:")

# User input
query = st.text_input("Enter your question here:")

if st.button("Submit"):
    if query.strip():
        # Retrieve relevant documents
        with st.spinner("Retrieving relevant documents..."):
            retrieve_response = requests.post(f"{API_URL}/retrieve", json={"query": query, "top_k": 5})
            if retrieve_response.status_code == 200:
                retrieved_chunks = retrieve_response.json()["results"]
                st.write("### Retrieved Documents:")
                for idx, chunk in enumerate(retrieved_chunks):
                    st.write(f"**Chunk {idx+1}:** {chunk['chunk']}")
            else:
                st.error("Error retrieving documents from the backend.")

        # Generate an answer
        with st.spinner("Generating response..."):
            generate_response = requests.post(f"{API_URL}/generate", json={"query": query})
            if generate_response.status_code == 200:
                response = generate_response.json()["response"]
                st.write("### Generated Answer:")
                st.success(response)
            else:
                st.error("Error generating answer from the backend.")
    else:
        st.error("Please enter a valid query.")
