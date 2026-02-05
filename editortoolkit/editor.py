"""
Core editing logic using LLM.
"""
import re
import logging
from typing import Optional, List, Tuple

from .llm_client import generate_text_with_logging
from .config import EDITOR_DEFAULT_MODEL, EDITOR_TEMPERATURE
from .prompts import (
    EDITOR_SYSTEM_PROMPT,
    get_editor_prompt,
    get_batch_editor_prompt,
    get_refinement_prompt,
    get_iterative_refinement_prompt,
)

logger = logging.getLogger(__name__)


class Editor:
    """Editor class using LLM for text editing tasks."""

    def __init__(self, model: str = None, temperature: float = None):
        """
        Initialize the editor.

        Args:
            model: Model to use for editing
            temperature: LLM temperature (lower = more deterministic)
        """
        self.model = model or EDITOR_DEFAULT_MODEL
        self.temperature = temperature if temperature is not None else EDITOR_TEMPERATURE

    async def close(self):
        """Close method for compatibility (no-op since we use module-level session)."""
        pass

    async def edit(self, text: str, task: str) -> str:
        """
        Edit text based on the specified task.

        Args:
            text: Text to edit
            task: Editing task (rephrase, professional, proofread, concise)

        Returns:
            Edited text
        """
        full_prompt = f"{EDITOR_SYSTEM_PROMPT}\n\n{get_editor_prompt(text, task)}"

        logger.info(f"[EDITOR] Editing {len(text)} chars | task={task}")

        response = await generate_text_with_logging(
            prompt=full_prompt,
            model=self.model,
            task=f"editor_{task}",
            temperature=self.temperature
        )

        edited_text = response.strip()

        logger.info(f"[EDITOR] Edit complete | task={task} | output_chars={len(edited_text)}")

        return edited_text

    async def edit_batch(self, texts: List[str], task: str) -> List[Tuple[str, str]]:
        """
        Edit multiple texts based on the specified task.

        Args:
            texts: List of texts to edit
            task: Editing task (rephrase, professional, proofread, concise)

        Returns:
            List of tuples (original_text, edited_text)
        """
        full_prompt = f"{EDITOR_SYSTEM_PROMPT}\n\n{get_batch_editor_prompt(texts, task)}"

        logger.info(f"[EDITOR] Batch editing {len(texts)} items | task={task}")

        response = await generate_text_with_logging(
            prompt=full_prompt,
            model=self.model,
            task=f"editor_batch_{task}",
            temperature=self.temperature
        )

        # Parse numbered response
        edits = self._parse_batch_response(response, len(texts))

        logger.info(f"[EDITOR] Batch edit complete | task={task} | items={len(edits)}")

        return list(zip(texts, edits))

    async def refine(
        self,
        current_result: str,
        user_feedback: str,
        task: str,
        original_text: Optional[str] = None
    ) -> str:
        """
        Refine a previous result based on user feedback.

        Args:
            current_result: Current result to refine
            user_feedback: User's feedback/instructions
            task: Original task type
            original_text: Optional original input text for context

        Returns:
            Refined text
        """
        if original_text:
            prompt = get_iterative_refinement_prompt(current_result, user_feedback, original_text)
        else:
            prompt = get_refinement_prompt(current_result, user_feedback, task)

        full_prompt = f"{EDITOR_SYSTEM_PROMPT}\n\n{prompt}"

        logger.info(f"[EDITOR] Refining result | task={task} | feedback_len={len(user_feedback)}")

        response = await generate_text_with_logging(
            prompt=full_prompt,
            model=self.model,
            task=f"editor_refine_{task}",
            temperature=self.temperature
        )

        refined_text = response.strip()

        logger.info(f"[EDITOR] Refinement complete | output_chars={len(refined_text)}")

        return refined_text

    def _parse_batch_response(self, response: str, expected_count: int) -> List[str]:
        """
        Parse numbered batch editing response.

        Args:
            response: LLM response with numbered edits
            expected_count: Expected number of edits

        Returns:
            List of edited texts
        """
        lines = response.strip().split('\n')
        edits = []

        # Try to extract numbered items
        pattern = r'^\d+\.\s*(.+)$'

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.match(pattern, line)
            if match:
                edits.append(match.group(1).strip())
            elif edits:
                # Continuation of previous line
                edits[-1] += ' ' + line

        # If parsing failed, try to split by double newlines
        if len(edits) != expected_count:
            parts = re.split(r'\n\s*\n', response.strip())
            if len(parts) == expected_count:
                edits = [p.strip() for p in parts]
            elif len(edits) < expected_count:
                # Pad with empty strings if needed
                edits.extend([''] * (expected_count - len(edits)))
            else:
                # Truncate if too many
                edits = edits[:expected_count]

        return edits


async def edit_text(
    text: str,
    task: str,
    model: Optional[str] = None
) -> dict:
    """
    Convenience function to edit text.

    Args:
        text: Text to edit
        task: Editing task
        model: Model to use (optional)

    Returns:
        Dictionary with editing results
    """
    editor = Editor(model=model)

    try:
        edited_text = await editor.edit(text=text, task=task)

        return {
            "original_text": text,
            "edited_text": edited_text,
            "task": task,
            "model": editor.model,
            "char_count": len(text),
            "word_count": len(text.split()),
            "status": "completed"
        }

    finally:
        await editor.close()
