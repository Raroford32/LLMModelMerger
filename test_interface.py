from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TestInterface:
    def __init__(self):
        self.model_path = "merged_model"
        self.model = None
        self.tokenizer = None

    def load_model(self):
        print("Loading merged model...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        print("Model loaded successfully.")

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def run_test_interface(self):
        self.load_model()
        print("Welcome to the merged model test interface.")
        print("Type 'exit' to quit.")
        
        while True:
            prompt = input("Enter your prompt: ")
            if prompt.lower() == 'exit':
                break
            response = self.generate_response(prompt)
            print("Model response:", response)
            print()

        print("Thank you for testing the merged model!")
