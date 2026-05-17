"""
Parses markdown conversation traces into structured Python objects.

Supports:
- User turns
- Assistant turns
- recommendations: null detection
- end_of_conversation detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ConversationTurn:
    turn_number: int
    user_message: str
    assistant_message: str
    recommendations_null: bool
    end_of_conversation: bool


TURN_PATTERN = re.compile(
    r"### Turn (\d+)(.*?)(?=### Turn \d+|\Z)",
    re.DOTALL,
)

USER_PATTERN = re.compile(
    r"\*\*User\*\*\s*> (.*?)(?=\*\*Agent\*\*)",
    re.DOTALL,
)

AGENT_PATTERN = re.compile(
    r"\*\*Agent\*\*\s*(.*?)(?=_`end_of_conversation`|_No recommendations|$)",
    re.DOTALL,
)

NULL_RECS_PATTERN = re.compile(
    r"recommendations:\s*null",
    re.IGNORECASE,
)

END_PATTERN = re.compile(
    r"`end_of_conversation`:\s*\*\*(true|false)\*\*",
    re.IGNORECASE,
)


def clean_block(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"\n\s*> ?", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def parse_trace(trace_path: str | Path) -> List[ConversationTurn]:
    path = Path(trace_path)

    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    raw = path.read_text(encoding="utf-8")

    turns: List[ConversationTurn] = []

    for match in TURN_PATTERN.finditer(raw):
        turn_number = int(match.group(1))
        chunk = match.group(2)

        user_match = USER_PATTERN.search(chunk)
        agent_match = AGENT_PATTERN.search(chunk)
        end_match = END_PATTERN.search(chunk)

        if not user_match:
            continue

        user_message = clean_block(user_match.group(1))

        assistant_message = (
            clean_block(agent_match.group(1))
            if agent_match
            else ""
        )

        recommendations_null = bool(
            NULL_RECS_PATTERN.search(chunk)
        )

        end_of_conversation = (
            end_match.group(1).lower() == "true"
            if end_match
            else False
        )

        turns.append(
            ConversationTurn(
                turn_number=turn_number,
                user_message=user_message,
                assistant_message=assistant_message,
                recommendations_null=recommendations_null,
                end_of_conversation=end_of_conversation,
            )
        )

    return turns