import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib
import chat_function

tokenizer = AutoTokenizer.from_pretrained("/models/stablelm-tuned-alpha-3b")
model = AutoModelForCausalLM.from_pretrained("/models/stablelm-tuned-alpha-3b")
model.half().cuda()

print("Welcome to the StableLM Tuned (Alpha version) chat!")

try:
    while True:
        user_input = input()
        try:
            importlib.reload(chat_function)
            response = chat_function.chat(user_input, tokenizer, model)
            print(response)
        except Exception as e:
            print(f"Error in chat_function.py: {e}")
except KeyboardInterrupt:
    print("\nExiting the chat. Goodbye!")
