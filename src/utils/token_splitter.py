import importlib


#===============================================	
# TOKEN SPLITTER CLASS
#===============================================


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