from trl import SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model=r"C:\Users\John\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca",
    train_dataset=load_dataset("json", data_files="Qwen3-0.6B-zhuxi-job-1.json", split="train"))
trainer.train()
