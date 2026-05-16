"""
Pydantic models that define the exact API contract.

The assignment schema is NON-NEGOTIABLE — these models enforce it.
Any deviation will break the automated evaluator.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    """A single turn in the conversation history."""
    role: str = Field(
        ...,
        description="Either 'user' or 'assistant'",
        pattern="^(user|assistant)$",
    )
    content: str = Field(
        ...,
        description="The text content of this turn",
        min_length=1,
    )


class ChatRequest(BaseModel):
    """
    The full request body for POST /chat.

    The entire conversation history is sent on every call (stateless API).
    The last message must always be from the user.
    """
    messages: List[Message] = Field(
        ...,
        description="Full conversation history, oldest first. Last item must be role='user'.",
        min_length=1,
    )

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, messages: List[Message]) -> List[Message]:
        if messages[-1].role != "user":
            raise ValueError("The last message in the conversation must be from the 'user'.")
        return messages


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class Recommendation(BaseModel):
    """
    A single SHL assessment in the shortlist.

    All three fields are required. 'url' MUST come from the scraped catalog.
    """
    name: str = Field(..., description="Exact product name from the SHL catalog")
    url: str = Field(..., description="Official SHL catalog URL for this product")
    test_type: str = Field(
        ...,
        description=(
            "Short type code(s). Examples: 'K' (Knowledge), 'P' (Personality), "
            "'A' (Ability), 'S' (Simulation), 'B' (Biodata/SJT), "
            "'C' (Competencies), 'D' (Development)"
        ),
    )


class ChatResponse(BaseModel):
    """
    The exact response schema returned by POST /chat.

    Rules enforced here:
      - reply: always present (never null/empty)
      - recommendations: null OR a list of 1–10 items (never empty list)
      - end_of_conversation: boolean, true only when task is complete
    """
    reply: str = Field(
        ...,
        description="The agent's conversational response text",
        min_length=1,
    )
    recommendations: Optional[List[Recommendation]] = Field(
        default=None,
        description=(
            "null when still clarifying or refusing. "
            "Array of 1–10 items when a shortlist is committed."
        ),
    )
    end_of_conversation: bool = Field(
        default=False,
        description="true only when the agent considers the task complete.",
    )

    @field_validator("recommendations")
    @classmethod
    def recommendations_size(
        cls, recs: Optional[List[Recommendation]]
    ) -> Optional[List[Recommendation]]:
        if recs is not None:
            if len(recs) == 0:
                # Convert empty list to null — we never return an empty array
                return None
            if len(recs) > 10:
                raise ValueError("Cannot return more than 10 recommendations.")
        return recs


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL CATALOG MODELS
# ─────────────────────────────────────────────────────────────────────────────

class CatalogProduct(BaseModel):
    """
    Internal representation of a single SHL assessment from the catalog.
    Used during ingestion, indexing, and retrieval.
    Not exposed in the API.
    """
    name: str
    url: str
    test_type_codes: List[str] = Field(
        description="List of type letter codes, e.g. ['K'], ['P'], ['K', 'S']"
    )
    test_type_labels: List[str] = Field(
        description="Human-readable type labels, e.g. ['Knowledge & Skills']"
    )
    duration: Optional[str] = Field(
        default=None,
        description="Duration string, e.g. '25 minutes', 'Untimed', or None if unknown"
    )
    languages: List[str] = Field(
        default_factory=list,
        description="List of supported language names"
    )
    description: Optional[str] = Field(
        default=None,
        description="Product description text if available"
    )
    is_individual_test: bool = Field(
        default=True,
        description="True = Individual Test Solution (in scope). False = Job Solution (out of scope)."
    )

    def to_recommendation(self) -> Recommendation:
        """Convert to API Recommendation format."""
        return Recommendation(
            name=self.name,
            url=self.url,
            test_type=",".join(self.test_type_codes),
        )

    def to_embedding_text(self) -> str:
        """
        Produce the rich text string that will be embedded into the FAISS index.

        This text is what semantic search runs against, so it must include:
        - All names and synonyms (e.g. 'OPQ32r' AND 'Occupational Personality Questionnaire')
        - The type labels in plain English
        - Duration (helps filter "quick" vs "thorough" requests)
        - Languages (critical for bilingual/language-specific queries)
        - Description (if available)

        We also manually add domain keywords based on test type to boost retrieval:
        - Personality tests → leadership, behaviour, workplace style
        - Ability tests → cognitive, reasoning, numerical, verbal
        - Simulation → hands-on, practical, realistic
        etc.
        """
        parts = [f"Product Name: {self.name}"]

        type_str = ", ".join(self.test_type_labels) if self.test_type_labels else "Unknown"
        parts.append(f"Assessment Type: {type_str}")

        # Add domain keywords based on type codes to improve retrieval
        keyword_map = {
            "P": "personality behaviour behavioral traits workplace style leadership character",
            "A": "cognitive ability aptitude reasoning numerical verbal inductive deductive intelligence IQ",
            "K": "knowledge skills technical proficiency domain expertise test quiz",
            "S": "simulation practical hands-on realistic job preview interactive exercise",
            "B": "situational judgement biodata SJT scenarios decision making work context",
            "C": "competencies competency framework workplace skills assessment",
            "D": "development developmental 360 feedback growth learning",
            "E": "exercises assessment centre role play in-tray case study",
        }
        domain_keywords = []
        for code in self.test_type_codes:
            if code in keyword_map:
                domain_keywords.append(keyword_map[code])
        if domain_keywords:
            parts.append(f"Domain Keywords: {' '.join(domain_keywords)}")

        if self.duration:
            parts.append(f"Duration: {self.duration}")

        if self.languages:
            # Limit to first 10 languages + count to keep embedding text manageable
            lang_preview = ", ".join(self.languages[:10])
            extra = len(self.languages) - 10
            if extra > 0:
                lang_preview += f" (+{extra} more)"
            parts.append(f"Languages: {lang_preview}")

        if self.description:
            parts.append(f"Description: {self.description}")

        parts.append(f"URL: {self.url}")

        return "\n".join(parts)

    def to_metadata_dict(self) -> dict:
        """Serialize to plain dict for JSON storage alongside FAISS index."""
        return {
            "name": self.name,
            "url": self.url,
            "test_type_codes": self.test_type_codes,
            "test_type_labels": self.test_type_labels,
            "duration": self.duration,
            "languages": self.languages,
            "description": self.description,
        }

    @classmethod
    def from_metadata_dict(cls, d: dict) -> "CatalogProduct":
        """Deserialize from the JSON metadata file."""
        return cls(
            name=d["name"],
            url=d["url"],
            test_type_codes=d.get("test_type_codes", []),
            test_type_labels=d.get("test_type_labels", []),
            duration=d.get("duration"),
            languages=d.get("languages", []),
            description=d.get("description"),
        )