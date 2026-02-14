from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = r"C:\Users\John\.cache\huggingface\hub\models--google--gemma-3-270m-it\snapshots\ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
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
