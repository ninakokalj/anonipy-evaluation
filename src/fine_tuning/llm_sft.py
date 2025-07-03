from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


# formatting prompts
def formatting_prompts_func(example):
        return [f"### Question: {q}\n ### Answer: {a}" for q, a in zip(example['instruction'], example['output'])]


def finetune_model(model_path: str, train_data_path: str, output_model_dir: str):
    """Fine-tunes a large language model using the provided training dataset and saves the fine-tuned model."""

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = load_dataset("json", data_files=train_data_path, split="train")
    split_point = int(len(dataset) * 0.8) 
    train_dataset = dataset.select(range(0, split_point))
    eval_dataset = dataset.select(range(split_point, len(dataset)))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(output_dir=output_model_dir),
        formatting_func=formatting_prompts_func,
        data_collator=collator
    )

    trainer.train()


def finetune_adapters(model_path: str, train_data_path: str, output_model_dir: str):
    """Fine-tunes adapter modules for a large language model on the provided dataset and saves the resulting adapters."""

    dataset = load_dataset("json", data_files=train_data_path, split="train")
    split_point = int(len(dataset) * 0.8) 
    train_dataset = dataset.select(range(0, split_point))
    eval_dataset = dataset.select(range(split_point, len(dataset)))

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],    # q_proj = query projection, v_proj = value, k_proj = key, o_proj = output
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(output_dir=output_model_dir, label_names = ["input_ids"]),
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.train()


"""
- gradient_accumulation_steps: how many batches to process before updating model weights
- num_train_epochs: how many times to go through the entire training dataset
- max_steps: maximum number of batches to process (-1 means go through the full dataset)
- learning_rate: how quickly the model learns (usually between 1e-5 and 1e-6)
"""


