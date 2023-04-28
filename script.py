import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

tokenizer = AutoTokenizer.from_pretrained("/models/stablelm-tuned-alpha-3b")
model = AutoModelForCausalLM.from_pretrained("/models/stablelm-tuned-alpha-3b")
model.half().cuda()

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = set([50278, 50279, 50277, 1, 0])
        return input_ids[0][-1] in stop_ids

def system_prompt():
    with open("system.txt", "r") as f:
        return f.read()

def chat(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    tokens = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.7,
        do_sample=True,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

print("Welcome to the StableLM Tuned (Alpha version) chat!")

try:
    while True:
        user_input = input("You: ")
        prompt = f"{system_prompt()}{user_input}"
        response = chat(prompt)
        print("StableLM:", response)
except KeyboardInterrupt:
    print("\nExiting the chat. Goodbye!")
