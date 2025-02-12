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

    def _prepare_pipeline(self, lang):
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
    
def find_sub_list(sub_list, main_list):
    sub_list_indices = []
    sll = len(sub_list)
    for idx in (i for i, e in enumerate(main_list) if e == sub_list[0]):
        if main_list[idx : idx + sll] == sub_list:  #problem?
            sub_list_indices.append((idx, idx + sll - 1))

    return sub_list_indices

def create_gliner_example(example):
    
    tokenizer = SpaCyTokenSplitter(
        lang=LANGUAGE_MAPPING[example["language"]]
    )


    # print(example["language"], LANGUAGE_MAPPING[example["language"]])
    ttext = [t[0] for t in tokenizer(example["text"])]
    ner = []
    for entity in example["entities"]:
        entity_text = [t[0] for t in tokenizer(entity["entity"])]   
        entity_types = entity["types"]
        entity_indices = find_sub_list(entity_text, ttext)  # ta ne najde
        for entity_index in entity_indices:
            ner.append([entity_index[0], entity_index[1], entity_types[0]])

    return {
        "text": example["text"],
        "language": example["language"],
        "domain": example["domain"],
        "entities": example["entities"],
        "tokenized_text": ttext,
        "gliner_entities": ner,
    }