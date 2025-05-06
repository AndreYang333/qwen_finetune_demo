# model/trainer.py
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from model.dataset import SupervisedDataset
import os



def load_model_and_tokenizer(model_name, lora, lora_cfg):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)

    # ✅ 修复 pad_token 问题
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ tokenizer.pad_token 设置为 eos_token")

    if lora:
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(**lora_cfg)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer




def train(cfg):
    model, tokenizer = load_model_and_tokenizer(
        cfg['model_name_or_path'],
        cfg['lora'],
        cfg.get('lora_config', {})
    )

    dataset = SupervisedDataset(cfg['data']['train_file'], tokenizer, cfg['max_seq_length'])

    training_args = TrainingArguments(
        output_dir=cfg['output_dir'],
        per_device_train_batch_size=cfg['train_batch_size'],
        per_device_eval_batch_size=cfg['eval_batch_size'],
        num_train_epochs=cfg['num_train_epochs'],
        learning_rate=cfg['learning_rate'],
        logging_steps=cfg['logging_steps'],
        save_steps=cfg['save_steps'],
        evaluation_strategy="steps" if cfg['data'].get('val_file') else "no",
        eval_steps=cfg['eval_steps'],
        warmup_ratio=cfg['warmup_ratio'],
        fp16=True,
        save_total_limit=2,
        report_to=cfg.get("report_to", "none"),
        logging_dir=cfg.get("logging_dir", "./logs")
    )

    print(f"✅ TensorBoard logging to: {training_args.logging_dir}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(cfg['output_dir'])
