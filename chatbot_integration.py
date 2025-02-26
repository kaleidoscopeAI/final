from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Chatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history = None

    def interact(self, user_input):
        """Generate response based on user input."""
        new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors="pt")
        bot_input_ids = torch.cat([self.chat_history, new_user_input_ids], dim=-1) if self.chat_history is not None else new_user_input_ids
        self.chat_history = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(self.chat_history[:, bot_input_ids.shape[-1]:], skip_special_tokens=True)
        return response
