from peft import PeftModel
from transformers import AutoModelForCausalLM

model_path = r"C:\Users\John\.cache\huggingface\hub\models--google--gemma-3-270m-it\snapshots\ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
model = AutoModelForCausalLM.from_pretrained(model_path)

model = PeftModel.from_pretrained(
    model, "trainer_output/checkpoint-30")

model = model.merge_and_unload()
model.save_pretrained("trainer_output/checkpoint-30")
