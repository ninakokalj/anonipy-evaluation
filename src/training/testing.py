from anonipy.anonymize.generators import LLMLabelGenerator
from anonipy.definitions import Entity




# llm_generator_small = LLMLabelGenerator(model_name = "models/135M/checkpoint-84", use_gpu = True, use_quant = False)
llm_generator_big = LLMLabelGenerator(model_name = "models/360M/checkpoint-1062", use_gpu = True, use_quant = False)

ent = Entity(
    "Jane Doe",
    "person",
    0,
    1
)

# generated_text_small = llm_generator_small.generate(entity=ent, add_entity_attrs="English", temperature=0.7)
generated_text_big = llm_generator_big.generate(entity=ent, add_entity_attrs="English", temperature=0.7)

# print("135M: " + generated_text_small)
print("360M: " + generated_text_big)