from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import LoraConfig

trainer = SFTTrainer(
    model=r"C:\Users\John\.cache\huggingface\hub\models--google--gemma-3-270m-it\snapshots\ac82b4e820549b854eebf28ce6dedaf9fdfa17b3",
    train_dataset=load_dataset("json",
                               data_files="gemma-3-270m-it-zhuxi-job.json",
                               split="train"),
    args=SFTConfig(learning_rate=2.0e-4),
    peft_config=LoraConfig(r=32,
                           lora_alpha=16,
                           lora_dropout=0.05,
                           bias="none",
                           task_type="CAUSAL_LM")
)
trainer.train()
