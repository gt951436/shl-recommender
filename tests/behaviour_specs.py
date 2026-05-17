"""
Behavioral assertions for conversational recommendation flows.
"""

from __future__ import annotations

from typing import Any


CONFIRMATION_SIGNALS = [
    "perfect",
    "confirmed",
    "that's good",
    "that works",
    "lock it in",
    "locking it in",
    "keep it as-is",
    "keep the shortlist",
    "final list",
]

LEGAL_SIGNALS = [
    "legally required",
    "compliance requirement",
    "regulatory obligation",
    "does this satisfy",
]

CLARIFICATION_SIGNALS = [
    "what level",
    "which language",
    "which fits",
    "selection or development",
    "one more question",
]


def is_clarification_turn(reply: str) -> bool:
    r = reply.lower()

    return (
        "?" in r
        or any(sig in r for sig in CLARIFICATION_SIGNALS)
    )


def is_confirmation_turn(user_message: str) -> bool:
    u = user_message.lower()

    return any(sig in u for sig in CONFIRMATION_SIGNALS)


def is_legal_refusal(reply: str) -> bool:
    r = reply.lower()

    refusal_terms = [
        "outside what i can advise",
        "legal",
        "compliance team",
        "counsel",
        "cannot interpret",
    ]

    return any(term in r for term in refusal_terms)


def assert_behavior(
    expected_turn,
    actual_response: dict[str, Any],
) -> list[str]:
    """
    Returns list of assertion failures.
    Empty list = PASS.
    """

    failures = []

    reply = actual_response.get("reply", "")
    recommendations = actual_response.get("recommendations")
    eoc = actual_response.get("end_of_conversation")

    # ─────────────────────────────────────────────
    # Null recommendation expectations
    # ─────────────────────────────────────────────

    if expected_turn.recommendations_null:
        if recommendations is not None:
            failures.append(
                "Expected recommendations=null"
            )

    # ─────────────────────────────────────────────
    # Clarification expectation
    # ─────────────────────────────────────────────

    if expected_turn.recommendations_null:
        if not is_clarification_turn(reply) and not is_legal_refusal(reply):
            failures.append(
                "Expected clarification/refusal behavior"
            )

    # ─────────────────────────────────────────────
    # End of conversation
    # ─────────────────────────────────────────────

    if expected_turn.end_of_conversation:
        if eoc is not True:
            failures.append(
                "Expected end_of_conversation=true"
            )

    return failures