from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset

trainer = SFTTrainer(
    model=r"C:\Users\John\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca",
    train_dataset=load_dataset("json",
                               data_files="Qwen3-0.6B-zhuxi-job-AI.json",
                               split="train"),
    eval_dataset=load_dataset("json",
                              data_files="Qwen3-0.6B-zhuxi-job.json",
                              split="train"),
    args=SFTConfig(learning_rate=2.0e-5,
                   num_train_epochs=3,
                   per_device_train_batch_size=2,
                   gradient_accumulation_steps=4,
                   eval_strategy="steps",
                   eval_on_start=True,
                   logging_steps=5,
                   eval_steps=5,
                   save_steps=5),
    peft_config=LoraConfig(r=16,
                           lora_alpha=32,
                           lora_dropout=0.05,
                           bias="none",
                           task_type="CAUSAL_LM",
                           target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
)
trainer.train(resume_from_checkpoint=False)
