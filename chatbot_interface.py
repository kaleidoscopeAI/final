from transformers import AutoModelForCausalLM, AutoTokenizer

class ChatbotInterface:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("./local_model")
        self.model = AutoModelForCausalLM.from_pretrained("./local_model")

    def process_input(self, user_input: str) -> str:
        inputs = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors="pt")
        response = self.model.generate(inputs, max_length=1000)
        return self.tokenizer.decode(response[0], skip_special_tokens=True)
