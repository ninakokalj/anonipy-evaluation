import re

import ollama
from ollama import chat
from pydantic import BaseModel

from anonipy.definitions import Entity


# Class for structured output
class Replacement(BaseModel):
    replacement: str


# =====================================
# Main class
# =====================================


class OllamaLabelGenerator:
    """The class representing the Ollama LLM label generator.
    
    Methods:
        generate(entity, add_entity_attrs, structured_output):
            Generate the label based on the entity.
    """

    def __init__(self, model_name: str):
        """Initializes the Ollama LLM label generator.

        Args:
            model_name: The name of the model to use.
        """

        self.model = model_name
        self.structured_output_fails = 0

    def generate(self, entity: Entity, add_entity_attrs: str, structured_output: bool = False) -> str:
        """Generate the substitute for the entity based on it's attributes.
            
            Args:
                entity: The entity to generate the label from.
                add_entity_attrs: Additional entity attribute description to add to the generation.
                structured_output: Whether to use structured output.

        Returns:
            The generated entity label substitute.
        """

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant for generating replacements for text entities.",
            },
            {
                "role": "user",
                "content": f"What is a random {add_entity_attrs} {entity.label} replacement for {entity.text}? Respond only with the replacement.",
            }
        ]
        if structured_output:
            try:
                response = chat(
                    messages=messages,
                    model=self.model,
                    format=Replacement.model_json_schema(),
                )
                print(Replacement.model_validate_json(response.message.content).replacement)
                return Replacement.model_validate_json(response.message.content).replacement
            except Exception as e:
                self.structured_output_fails += 1
                print(f"Structured output failed: {str(e)}")
                

        response = chat(
            messages=messages,
            model=self.model
        )
        response = response.message.content

        # Clean up for deepseek models
        cleaned_content = re.sub(r"<think>.*?</think>\n?", "", response, flags=re.DOTALL)
        print(cleaned_content.strip())
        return cleaned_content.strip()

    def print_logs(self):
        print(f"Structured output failed {self.structured_output_fails} times.")