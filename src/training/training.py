from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset


def finetune_model(data_path, model_dir):

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")

    dataset = load_dataset("json", data_files=data_path, split="train")

    def formatting_prompts_func(example):
        return [f"### Question: {q}\n ### Answer: {a}" for q, a in zip(example['instruction'], example['output'])]


    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=SFTConfig(output_dir=model_dir),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.train()

finetune_model("data/training/eng/mix_eng.json", "models/360M")