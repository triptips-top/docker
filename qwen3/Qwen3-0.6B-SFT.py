from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import LoraConfig

trainer = SFTTrainer(
    model=r"C:\Users\John\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca",
    train_dataset=load_dataset("json",
                               data_files="Qwen3-0.6B-zhuxi-job.json",
                               split="train"),
    args=SFTConfig(learning_rate=2.0e-4,
                   per_device_train_batch_size=1,
                   gradient_accumulation_steps=8),
    peft_config=LoraConfig(r=32,
                           lora_alpha=16,
                           lora_dropout=0.05,
                           bias="none",
                           task_type="CAUSAL_LM")
)
trainer.train()
