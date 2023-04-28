print("Reloaded Chat function")
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = set([50278, 50279, 50277, 1, 0])
        return input_ids[0][-1] in stop_ids

def system_prompt():
    with open("system.txt", "r") as f:
        return f.read()

def read_conversation():
    try:
        with open("conversation.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def write_conversation(conversation):
    with open("conversation.json", "w") as f:
        json.dump(conversation, f, indent=2)

def truncate_conversation(conversation, user_input, tokenizer):
    tokenized_user_input = tokenizer(user_input, return_tensors="pt")
    
    while True:
        conversation_text = "\n".join([msg["content"] for msg in conversation])
        tokenized_conversation = tokenizer(conversation_text, return_tensors="pt")

        if len(tokenized_conversation["input_ids"][0]) + len(tokenized_user_input["input_ids"][0]) <= 4096:
            break

        conversation.pop(0)

    return conversation

def chat(user_prompt, tokenizer, model):
    conversation = read_conversation()
    if len(conversation) == 0:
        conversation.append({"role": "system", "content": system_prompt()})
    truncated_conversation = truncate_conversation(conversation, user_prompt, tokenizer)
    input_text = f"{system_prompt()}\n{''.join([msg['content'] for msg in truncated_conversation])}\n{user_prompt}\n\n"
    print(input_text)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    tokens = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.7,
        do_sample=True,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )
    response = tokenizer.decode(tokens[0], skip_special_tokens=True)
    truncated_conversation.append({"role": "user", "content": user_prompt})
    truncated_conversation.append({"role": "assistant", "content": response})
    write_conversation(truncated_conversation)
    return "\n\nStableLM:\n"+response