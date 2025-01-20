import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import json

class Retriever:
    def __init__(self, embeddings_path, docs_path, model_name="all-MiniLM-L6-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings = torch.tensor(np.load(embeddings_path)).to(self.device)
        with open(docs_path, "r") as f:
            self.docs = json.load(f)
        self.model = SentenceTransformer(model_name).to(self.device)
    
    def retrieve(self, query, k=5):
        # Encode the query into an embedding
        query_embedding = self.model.encode([query], convert_to_tensor=True, device=self.device)
        
        # Calculate cosine similarity
        similarities = torch.nn.functional.cosine_similarity(query_embedding, self.embeddings)
        
        # Get top-k indices
        top_k_indices = torch.topk(similarities, k=k).indices.cpu().numpy()
        
        # Retrieve top-k chunks
        results = [self.docs[i] for i in top_k_indices]
        return results
