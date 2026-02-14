from peft import PeftModel
from transformers import AutoModelForCausalLM

model_path = r"C:\Users\John\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
model = AutoModelForCausalLM.from_pretrained(model_path)

model = PeftModel.from_pretrained(
    model, "trainer_output/checkpoint-30")

model = model.merge_and_unload()
model.save_pretrained("trainer_output/checkpoint-30")
