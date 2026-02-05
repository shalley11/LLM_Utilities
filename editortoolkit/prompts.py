"""
Prompts for Editor Toolkit service.
"""


EDITOR_SYSTEM_PROMPT = """You are a professional text editor. Your task is to edit text accurately while:
- Preserving the original meaning and intent
- Using ONLY information present in the text
- NOT adding new information or assumptions
- Maintaining factual accuracy

Provide ONLY the edited text without any explanations, notes, or additional commentary."""


def get_rephrase_prompt(text: str) -> str:
    """Generate prompt for rephrasing text."""
    return f"""You are a rewriting system.

RULES:
- Use ONLY the information present in the text
- Do NOT change the meaning
- Do NOT add new information

TASK:
Rephrase the text to improve clarity and readability while preserving meaning.
Remove any repetitions and redundancy to make the text more natural and fluent.

TEXT:
{text}

OUTPUT:
Rephrased Text:
"""


def get_professional_prompt(text: str) -> str:
    """Generate prompt for professional tone rewriting."""
    return f"""You are a professional writing assistant.

RULES:
- Use ONLY the information present in the text
- Do NOT change the meaning
- Do NOT add new information

TASK:
Rewrite the text in a formal, professional tone suitable for business communication.
Use precise vocabulary and proper structure.

TEXT:
{text}

OUTPUT:
Professional Text:
"""


def get_proofread_prompt(text: str, focus: str = "general") -> str:
    """Generate prompt for proofreading text."""
    focus_instructions = {
        "general": "Fix all grammar, spelling, punctuation, and clarity issues.",
        "grammar": "Focus on fixing grammatical errors only.",
        "punctuation": "Focus on fixing punctuation errors only.",
        "clarity": "Focus on improving clarity and readability."
    }

    instruction = focus_instructions.get(focus, focus_instructions["general"])

    return f"""You are a professional proofreader.

RULES:
- Preserve the original meaning
- Do NOT add new content
- Do NOT remove important information
- Make minimal changes necessary

TASK:
{instruction}

TEXT:
{text}

OUTPUT:
Proofread Text:
"""


def get_concise_prompt(text: str) -> str:
    """Generate prompt for making text concise."""
    return f"""You are a text condensation system.

RULES:
- Use ONLY the information present in the text
- Do NOT remove essential meaning
- Do NOT add new information

TASK:
Shorten the text by removing unnecessary words and redundancy.
Keep only essential information while preserving meaning.

TEXT:
{text}

OUTPUT:
Concise Text:
"""


def get_editor_prompt(text: str, task: str) -> str:
    """Get the appropriate prompt for the editing task."""
    prompt_functions = {
        "rephrase": get_rephrase_prompt,
        "professional": get_professional_prompt,
        "proofread": get_proofread_prompt,
        "concise": get_concise_prompt,
    }

    if task not in prompt_functions:
        raise ValueError(f"Unsupported task: {task}. Supported tasks: {list(prompt_functions.keys())}")

    return prompt_functions[task](text)


def get_batch_editor_prompt(texts: list, task: str) -> str:
    """Generate batch editing prompt."""
    numbered_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])

    task_instructions = {
        "rephrase": "Rephrase each text to improve clarity and readability while preserving meaning. Remove any repetitions and redundancy.",
        "professional": "Rewrite each text in a formal, professional tone suitable for business communication.",
        "proofread": "Fix grammar, spelling, punctuation, and clarity issues in each text.",
        "concise": "Shorten each text by removing unnecessary words while preserving essential meaning.",
    }

    instruction = task_instructions.get(task, task_instructions["rephrase"])

    return f"""You are a professional text editor.

RULES:
- Use ONLY the information present in each text
- Do NOT change the meaning
- Do NOT add new information
- Maintain the same numbering in your response

TASK:
{instruction}

TEXTS TO EDIT:
{numbered_texts}

OUTPUT (maintain numbering):
"""


def get_refinement_prompt(current_result: str, user_feedback: str, task: str) -> str:
    """
    Generate a prompt for refining a previous result based on user feedback.

    Args:
        current_result: The current/last result to refine
        user_feedback: User's feedback/instructions for refinement
        task: Original task type (rephrase, professional, etc.)

    Returns:
        Refinement prompt
    """
    task_context = {
        "rephrase": "rephrased text",
        "professional": "professional text",
        "concise": "concise text",
        "proofread": "proofread text"
    }

    result_type = task_context.get(task, "text")

    return f"""You are a text refinement assistant.

CURRENT {result_type.upper()}:
{current_result}

USER FEEDBACK:
{user_feedback}

RULES:
- Modify the {result_type} according to the user's feedback
- Maintain the overall meaning and accuracy
- Only change what the user specifically requests
- Keep unchanged parts intact

TASK:
Refine the {result_type} based on the user's feedback.

OUTPUT:
Refined {result_type.title()}:
"""


def get_iterative_refinement_prompt(
    current_result: str,
    user_feedback: str,
    original_text: str = None
) -> str:
    """
    Generate a prompt for iterative refinement with optional original context.

    Args:
        current_result: Current result to refine
        user_feedback: User's refinement instructions
        original_text: Optional original input text for context

    Returns:
        Refinement prompt
    """
    original_context = ""
    if original_text:
        # Include truncated original for context
        truncated = original_text[:500] + "..." if len(original_text) > 500 else original_text
        original_context = f"""
ORIGINAL INPUT (for reference):
{truncated}

"""

    return f"""You are a text refinement assistant helping to improve content through iteration.
{original_context}
CURRENT VERSION:
{current_result}

USER'S REQUEST:
{user_feedback}

RULES:
- Apply the user's requested changes
- Preserve parts not mentioned in the feedback
- Maintain accuracy and coherence
- Do NOT add information not present in the current version

OUTPUT:
Refined Version:
"""
