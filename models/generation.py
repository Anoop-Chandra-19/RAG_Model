from transformers import AutoModelForCausalLM, AutoTokenizer

class Generator:
    def __init__(self, model_name = "gpt2"):
         # Load the Hugging Face model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, query, retrieved_chunks, max_length=200):
        # Combine retrieved chunks into a single context
        context = "\n".join([chunk["chunk"] for chunk in retrieved_chunks])
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        
        # Encode the input and generate a response
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1)
        
        # Decode the output to text
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response["choices"][0]["text"]
