"""Prompt formatting and answer parsing compatible with WorldBench."""

from __future__ import annotations

import re
from collections.abc import Mapping

_ANSWER = re.compile(r"(?:^|\n)\s*ANSWER:\s*([A-D])\s*\Z", re.IGNORECASE)


def format_prompt(question: str, options: Mapping[str, str]) -> str:
    """Format one WorldBench multiple-choice question."""
    return (
        f"Question: {question}\n\n"
        "Options:\n"
        f"A: {options['A']}\n"
        f"B: {options['B']}\n"
        f"C: {options['C']}\n"
        f"D: {options['D']}\n\n"
        "Please answer the question and select the most appropriate answer from the given options.\n"
        "First provide a brief explanation of your reasoning, then state your answer.\n\n"
        "Your response must end with the answer line and contain nothing after it.\n"
        "Format your response as:\n"
        "Explanation: [Brief explanation of your reasoning]\n"
        "Answer: [LETTER]"
    )


def parse_answer(raw_text: object) -> str:
    """Extract an A-D answer from a model response."""
    if not isinstance(raw_text, str) or not raw_text:
        return ""

    text = re.sub(r"[*_`]", " ", raw_text.strip().upper())
    match = _ANSWER.search(text)
    return match.group(1).upper() if match else ""
