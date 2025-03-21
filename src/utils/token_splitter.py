# koda za ner in tokenized_text



LANGUAGE_MAPPING = {
    "Dutch": ("nl", "Dutch"),
    "Slovene": ("sl", "Slovenian"),
    "Italian": ("it", "Italian"),
    "Greek": ("el", "Greek"),
    "French": ("fr", "French"),
    "English": ("en", "English"),
    "German": ("de", "German"),
}

import importlib


class SpaCyTokenSplitter:
    def __init__(self, lang=None):
        try:
            import spacy  # noqa
        except ModuleNotFoundError as error:
            raise error.__class__("Please install spacy with: `pip install spacy`")
        if lang is None:
            lang = ("en", "English")  # Default to English if no language is specified
        self.nlp = self._prepare_pipeline(lang)

    def __call__(self, text):
        doc = self.nlp(text)
        for token in doc:
            yield token.text, token.idx, token.idx + len(token.text)

    def _prepare_pipeline(self, lang: tuple):
        """Prepare the spacy pipeline.

        Prepares the pipeline for processing the text in the corresponding
        provided language.

        Returns:
            The spacy text processing and extraction pipeline.

        """
        # load the appropriate parser for the language
        module_lang, class_lang = lang[0].lower(), lang[1].lower().title()
        language_module = importlib.import_module(f"spacy.lang.{module_lang}")
        language_class = getattr(language_module, class_lang)
        # initialize the language parser
        nlp = language_class()
        nlp.add_pipe("sentencizer")
        return nlp
    
def find_sub_list(sub_list: list, main_list: list) -> list:
    sub_list_indices = []
    sll = len(sub_list)
    for idx in (i for i, e in enumerate(main_list) if e == sub_list[0]):
        if main_list[idx : idx + sll] == sub_list:  
            sub_list_indices.append((idx, idx + sll - 1))

    return sub_list_indices

def remove_duplicates(lst):
    unique_tuples = set(tuple(sublist) for sublist in lst)
    result = sorted([list(t) for t in unique_tuples], key=lambda x: (x[0], x[1]))
    return result

def create_new_gliner_example(example: dict, new_entities: list) -> dict:
    
    tokenizer = SpaCyTokenSplitter(
        lang=LANGUAGE_MAPPING[example["language"]]
    )

    ttext = [t[0] for t in tokenizer(example["text"])]
    ner = []
    for entity in new_entities: # new_entities = [{"text": text, "label": label}]
        entity_text = [t[0] for t in tokenizer(entity["text"])]   
        entity_indices = find_sub_list(entity_text, ttext) 
        for entity_index in entity_indices:
            ner.append([entity_index[0], entity_index[1], entity["label"]])

    return {
        "text": example["text"],
        "language": example["language"],
        # "domain": example["domain"],
        "tokenized_text": ttext,
        "ner": remove_duplicates(ner),
    }