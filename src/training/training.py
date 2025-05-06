from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from peft import LoraConfig


def formatting_prompts_func(example):
        return [f"### Question: {q}\n ### Answer: {a}" for q, a in zip(example['instruction'], example['output'])]


def finetune_model(model_path: str, train_data_path: str, output_model_dir: str):

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
        #modules_to_save=["lm_head", "embed_token"],
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



#finetune_model("HuggingFaceTB/SmolLM2-360M-Instruct", "data/training/data_ITALIAN/DATASET_it_300.json", "models/360M/italian_300")
finetune_adapters("meta-llama/Llama-3.2-1B-Instruct", "data/training/data_GERMAN/DATASET_de_200.json", "models/Llama-3.2-1B/german_200")


# gradient_accumulation_steps kok batch sizov počakat preden posodobiš
# epoch kolkrat greš čez celoten dataset num_train_epochs
# max_step max batchov ki jih bo vidu //pusti -1 da gre čez celoten dataset
# learning_rate -> 10^-5 -> 10^-6

# dodala definirala learning_rate, dodala eval_dataset