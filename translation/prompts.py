"""
Prompts for translation service.
"""

TRANSLATION_SYSTEM_PROMPT = """You are a professional translator. Your task is to translate text accurately while preserving:
- The original meaning and intent
- The tone and style of the original text
- Any formatting, punctuation, and structure
- Technical terms and proper nouns (translate or keep as appropriate)

Provide ONLY the translated text without any explanations, notes, or additional commentary."""


def get_translation_prompt(text: str, target_language: str, source_language: str = None) -> str:
    """Generate translation prompt."""
    if source_language:
        return f"""Translate the following text from {source_language} to {target_language}.

Text to translate:
{text}

Translated text:"""
    else:
        return f"""Translate the following text to {target_language}.

Text to translate:
{text}

Translated text:"""


def get_batch_translation_prompt(texts: list, target_language: str, source_language: str = None) -> str:
    """Generate batch translation prompt."""
    numbered_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])

    if source_language:
        return f"""Translate the following numbered texts from {source_language} to {target_language}.
Maintain the same numbering in your response.

Texts to translate:
{numbered_texts}

Translated texts (maintain numbering):"""
    else:
        return f"""Translate the following numbered texts to {target_language}.
Maintain the same numbering in your response.

Texts to translate:
{numbered_texts}

Translated texts (maintain numbering):"""
