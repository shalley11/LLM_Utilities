"""
Prompt templates for hierarchical summarization.
"""

# =========================
# Batch summarization prompt
# =========================
BATCH_SUMMARY_PROMPT = """
You are summarizing PART {batch_index} of {total_batches} of a larger document.

IMPORTANT RULES:
- Use ONLY the information present in the content
- Do NOT add assumptions, interpretations, or external knowledge
- If information is missing or unclear, state it explicitly
- Preserve factual accuracy (numbers, dates, metrics, names)
- PRESERVE all image placeholders exactly as they appear (e.g. [IMAGE_1], [IMAGE_2]). Include them inline in the summary where the image is relevant.

CONTENT:
{content}

TASK:
Generate a concise summary of approximately {word_count} words that:
- Captures the main ideas and key findings in this part
- Preserves important facts, figures, and results
- Maintains enough context to be merged with summaries from other parts
- Keeps all [IMAGE_X] placeholders in the output at contextually appropriate positions

OUTPUT:
Summary:
"""


# ==================================
# Summary type instructions
# ==================================
SUMMARY_TYPE_INSTRUCTIONS = {
    "brief": """
Generate a BRIEF final summary.

RULES:
- Focus on the single most important idea and outcome
- Keep it as short as possible while remaining complete
- Preserve critical conclusions or facts if present
- Do NOT add background, examples, or assumptions

STYLE:
- Extremely concise
- Clear and factual
""",

    "bullets": """
Generate a BULLET-WISE final summary.

RULES:
- Use bullets to cover all key topics and findings
- Include important facts, figures, or comparisons when relevant
- Avoid redundancy or overlapping bullets
- Exclude minor or repetitive details

STYLE:
- Information-dense bullets
- Start bullets with strong nouns or verbs
""",

    "detailed": """
Generate a DETAILED final summary.

RULES:
- Cover all major themes and insights across the content
- Integrate information into a coherent and logical narrative
- Preserve important data points, trends, and conclusions
- Avoid unnecessary repetition

STYLE:
- Clear structure and logical flow
- Neutral, professional tone
""",

    "executive": """
You are an experienced executive communications advisor writing for senior leadership.

RULES:
- Use ONLY the information provided in the content
- Do NOT add assumptions, interpretations, or external knowledge
- Focus on outcomes, implications, and decisions rather than operational details

TASK:
Create a concise, high-impact executive summary that:
- Highlights key outcomes, insights, and decisions
- Emphasizes business impact, risks, and opportunities
- Omits technical detail unless essential for understanding

STYLE:
- Professional, neutral, and confident tone
- Clear and structured
- No repetition, no filler
"""
}


# =========================
# Final combination prompt
# =========================
FINAL_COMBINE_PROMPT = """
You are combining summaries from different sections of a document into one coherent summary.

IMPORTANT RULES:
- Use ONLY the provided summaries
- Do NOT introduce new information or assumptions
- PRESERVE all image placeholders exactly as they appear (e.g. [IMAGE_1], [IMAGE_2]). Include them inline where the image is relevant.

Section Summaries:
{combined_content}

{instruction}

OUTPUT:
Final Summary:
"""


# =========================
# Direct summarization prompt
# =========================
DIRECT_SUMMARY_PROMPT = """
Analyze the following content and generate a summary.

IMPORTANT RULES:
- Use ONLY the information present in the content
- Do NOT add assumptions or external knowledge
- Preserve factual accuracy
- PRESERVE all image placeholders exactly as they appear (e.g. [IMAGE_1], [IMAGE_2]). Include them inline in the summary where the image is relevant.

Content:
{content}

{instruction}

OUTPUT:
Summary:
"""


def get_batch_summary_prompt(content: str, batch_index: int, total_batches: int, word_count: int) -> str:
    """Generate prompt for batch summarization."""
    return BATCH_SUMMARY_PROMPT.format(
        batch_index=batch_index + 1,
        total_batches=total_batches,
        content=content,
        word_count=word_count
    )


def get_final_combine_prompt(summaries: list, summary_type: str) -> str:
    """Generate prompt for combining summaries."""
    combined_content = "\n\n".join(
        f"[Section {i + 1}]: {summary}"
        for i, summary in enumerate(summaries)
    )

    instruction = SUMMARY_TYPE_INSTRUCTIONS.get(
        summary_type, SUMMARY_TYPE_INSTRUCTIONS["detailed"]
    )

    return FINAL_COMBINE_PROMPT.format(
        combined_content=combined_content,
        instruction=instruction
    )


def get_direct_summary_prompt(content: str, summary_type: str) -> str:
    """Generate prompt for direct summarization (small documents)."""
    instruction = SUMMARY_TYPE_INSTRUCTIONS.get(
        summary_type, SUMMARY_TYPE_INSTRUCTIONS["detailed"]
    )
    return DIRECT_SUMMARY_PROMPT.format(
        content=content,
        instruction=instruction
    )
