from fastapi import FastAPI
from models.retrieval import Retriever
from models.generation import Generator

app = FastAPI()

# Initialize the Retriever and Generator
retriever = Retriever(
    embeddings_path="data/embeddings_local.npy", 
    docs_path="data/preprocessed_docs_local.json"
)
generator = Generator(model_name="gpt2")

@app.post("/retrieve")
def retrieve_docs(query: str, top_k: int = 5):
    results = retriever.retrieve(query, k=top_k)
    return {"results": results}

@app.post("/generate")
def generate_answer(query: str):
    retrieved_chunks = retriever.retrieve(query)
    response = generator.generate_response(query, retrieved_chunks)
    return {"response": response}
