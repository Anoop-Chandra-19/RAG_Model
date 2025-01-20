import openai

class Generator:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_response(self, query, retrieved_chunks):
        # Combine retrieved chunks into a single context
        context = "\n".join([chunk["chunk"] for chunk in retrieved_chunks])
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        
        # Call the OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200
        )
        return response["choices"][0]["text"]
