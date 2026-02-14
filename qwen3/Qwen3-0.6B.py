from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = r"C:\Users\John\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

messages = [

]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
