from ollama import chat
import ollama
from pydantic import BaseModel
from anonipy.definitions import Entity

class Replacement(BaseModel):
    replacement: str

class OllamaInterface:
    def __init__(self, model_name: str):
        
        self.model = model_name
        ollama.pull(model_name)

    def generate(self, entity: Entity, add_entity_attrs: str) -> str:

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

        response = chat(
            messages=messages,
            model=self.model,
            format=Replacement.model_json_schema(),
        )

        return Replacement.model_validate_json(response.message.content).replacement
    

