from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r"C:\Users\John\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

model = PeftModel.from_pretrained(
    model, "trainer_output/Qwen3-0.6B-zhuxi-job-1")

model = model.merge_and_unload()
model.save_pretrained("trainer_output/Qwen3-0.6B-zhuxi-job-1")
