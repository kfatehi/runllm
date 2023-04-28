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

def read_conversation():
    try:
        with open("conversation.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def write_conversation(conversation):
    with open("conversation.txt", "w") as f:
        f.write(conversation)

def chat(prompt):
    conversation = read_conversation()
    inputs = tokenizer(conversation + prompt, return_tensors="pt").to("cuda")
    tokens = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.7,
        do_sample=True,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )
    response = tokenizer.decode(tokens[0], skip_special_tokens=True)
    truncated_conversation = conversation[-1000:]  # Adjust the truncation value as needed
    write_conversation(truncated_conversation + f"You: {prompt}\nStableLM: {response}\n")
    return response

print("Welcome to the StableLM Tuned (Alpha version) chat!")

try:
    while True:
        user_input = input("You: ")
        prompt = f"{system_prompt()}{user_input}"
        response = chat(prompt)
        print("StableLM:", response)
except KeyboardInterrupt:
    print("\nExiting the chat. Goodbye!")
