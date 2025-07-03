import random
import warnings

from typing import List, Tuple

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from anonipy.anonymize.generators.interface import GeneratorInterface
from anonipy.definitions import Entity
from anonipy.utils.package import is_installed_with

from constants import ALL_LABELS


# =====================================
# Main class
# =====================================


class LLMLabelGenerator(GeneratorInterface):
    """The class representing the LLM label generator.

    Examples:
        >>> from anonipy.anonymize.generators import LLMLabelGenerator
        >>> generator = LLMLabelGenerator()
        >>> generator.generate(entity)

    Attributes:
        model (models.Transformers): The model used to generate the label substitutes.

    Methods:
        generate(entity, entity_prefix, temperature):
            Generate the label based on the entity.

    """

    def __init__(
        self,
        *args,
        model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        adapter_name: str | None = None,
        in_context: bool = False,
        use_gpu: bool = False,
        use_quant: bool = False,
        **kwargs,
    ):
        """Initializes the LLM label generator.

        Args:
            model_name: The name of the model to use.
            adapter_name: The name of the adapter to use.
            in_context: Whether to use in-context learning or not.
            use_gpu: Whether to use GPU or not.
            use_quant: Whether to use quantization or not.

        Examples:
            >>> from anonipy.anonymize.generators import LLMLabelGenerator
            >>> generator = LLMLabelGenerator()
            LLMLabelGenerator()

        """

        super().__init__(*args, **kwargs)

        if use_gpu and not torch.cuda.is_available():
            warnings.warn(
                "The use_gpu=True flag requires GPU/CUDA, but it is not available. Setting use_gpu=False."
            )
            use_gpu = False

        if use_quant and not is_installed_with(["quant", "all"]):
            warnings.warn(
                "The use_quant=True flag requires the 'quant' extra dependencies, but they are not installed. Setting use_quant=False."
            )
            use_quant = False

        self.model, self.tokenizer = self._prepare_model_and_tokenizer(
            model_name, use_gpu, use_quant
        )

        if adapter_name:
            # load adapter into base model
            self.model = PeftModel.from_pretrained(self.model, adapter_name)

        self.in_context = in_context

    def generate(
        self,
        entity: Entity,
        *args,
        add_entity_attrs: str = "",
        temperature: float = 1.0,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        """Generate the substitute for the entity based on it's attributes.

        Examples:
            >>> from anonipy.anonymize.generators import LLMLabelGenerator
            >>> generator = LLMLabelGenerator()
            >>> generator.generate(entity)
            label

        Args:
            entity: The entity to generate the label from.
            add_entity_attrs: Additional entity attribute description to add to the generation.
            temperature: The temperature to use for the generation.
            top_p: The top p to use for the generation.

        Returns:
            The generated entity label substitute.

        """

        # If using in-context learning
        if self.in_context and entity.label in ALL_LABELS:  # icl only for labels in ALL_LABELS
            message = build_icl_prompt(entity, add_entity_attrs)
        else:
            message = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for generating replacements for text entities.",
                },
                {
                    "role": "user",
                    "content": f"What is a random {add_entity_attrs} {entity.label} replacement for {entity.text}? Respond only with the replacement.",
                },
            ]

        return self._generate_response(message, temperature, top_p)

    # =================================
    # Private methods
    # =================================

    def _prepare_model_and_tokenizer(
        self, model_name: str, use_gpu: bool, use_quant: bool
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Prepares the model and tokenizer.

        Args:
            model_name: The name of the model to use.

        Returns:
            The huggingface model.
            The huggingface tokenizer.

        """

        # prepare the model
        device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        dtype = torch.float32 if device.type == "cpu" else torch.float16

        model = self._load_model(model_name, device, dtype, use_quant, use_gpu)
        tokenizer = self._load_tokenizer(model_name)

        return model, tokenizer

    def _load_model(
        self,
        model_name: str,
        device: torch.device,
        dtype: torch.dtype,
        use_quant: bool,
        use_gpu: bool,
    ) -> AutoModelForCausalLM:
        """Load the model with appropriate configuration.

        Args:
            model_name: The name of the model to use.
            device: The device to use for the model.
            dtype: The data type to use for the model.
            use_quant: Whether to use quantization or not.
            use_gpu: Whether to use GPU or not.

        Returns:
            The huggingface model.

        """
        if use_quant and use_gpu:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True, bnb_4bit_compute_dtype=dtype
            )
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=dtype,
                quantization_config=quant_config,
            )

        if use_quant:
            warnings.warn(
                "Quantization is only supported on GPU, but use_gpu=False. Loading model without quantization."
            )

        return AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, torch_dtype=dtype
        )

    def _load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """Load the tokenizer with appropriate configuration.

        Args:
            model_name: The name of the model to use.

        Returns:
            The huggingface tokenizer.
        """
        return AutoTokenizer.from_pretrained(
            model_name, padding_side="right", use_fast=False
        )

    def _generate_response(
        self, message: List[dict], temperature: float, top_p: float
    ) -> str:
        """Generate the response from the LLM.

        Args:
            message: The message to generate the response from.
            temperature: The temperature to use for the generation.
            top_p: The top p to use for the generation.

        Returns:
            The generated response.

        """

        # tokenize the message
        input_ids = self.tokenizer.apply_chat_template(
            message, tokenize=True, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)

        # create attention mask (1 for all tokens)
        attention_mask = torch.ones_like(input_ids)

        # set pad token id if not set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*the `logits` model output.*")

            # generate the response
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        # decode the response
        response = self.tokenizer.decode(
            output_ids[0][len(input_ids[0]) :], skip_special_tokens=True
        )
        return response


# ========================================
# Helper functions for in-context learning
# ========================================

# Paths to the CSV files containing entities to use in prompts (only English data here)
paths = {
    "person name": "data/training/helpers/en/person.csv",
    "person": "data/training/helpers/en/person.csv",
    "name": "data/training/helpers/en/person.csv",
    "date": "data/training/helpers/en/dates.csv",
    "address": "data/training/helpers/en/address.csv",
    "organization": "data/training/helpers/en/orgs.csv",
    "company": "data/training/helpers/en/orgs.csv",
    "email": "data/training/helpers/en/emails.csv",
    "username": "data/training/helpers/en/user.csv",
    "passport number": "data/training/helpers/passport_nums.csv",
    "tax identification number": "data/training/helpers/en/tax.csv"
}

def build_icl_prompt(entity: Entity, add_entity_attrs: str) -> List[dict]:
    """Builds the prompt for in-context learning with 3 examples of correct replacements."""

    label = entity.label
    text = entity.text
    csv_path = paths[label]

    df = pd.read_csv(csv_path)
    column = df.iloc[:, 0]
    options = [val for val in column if val != text]

    sampled = random.sample(options, 6)
    a, b, c, d, e, f = sampled

    return [ 
    {
        "role": "system",
        "content": "You are a helpful AI assistant for generating replacements for text entities.",
    },
    {
        "role": "user",
        "content": f"""Here are a few examples of replacements:

What is a random {add_entity_attrs} {label} replacement for {a}?
{b}

What is a random {add_entity_attrs} {label} replacement for {c}?
{d}

What is a random {add_entity_attrs} {label} replacement for {e}?
{f}

Now, what is a random {add_entity_attrs} {label} replacement for {text}?
""",
    },
]