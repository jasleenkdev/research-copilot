"""Output schemas.

CONCEPT: Why Pydantic for LLM output
An LLM returns text, but downstream code needs data with guaranteed fields and
types (later: graph state, the Critic agent, an API response). A Pydantic model
is that contract. PydanticOutputParser uses it twice:
  1. It renders the schema as format instructions for the prompt.
  2. It validates the model's reply against the schema and raises
     OutputParserException if the reply doesn't match.

Field descriptions end up inside those format instructions, so they are part of
the prompt. Write them for the model.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ResearchAnswer(BaseModel):
    """A structured answer to a research question."""

    question: str = Field(
        description="The question being answered, restated in one sentence."
    )
    summary: str = Field(description="A direct answer in 2-4 sentences.")
    key_points: list[str] = Field(
        description="The 3-6 most important supporting points."
    )
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "Title and URL of each source actually used from the research notes. "
            "Empty if the notes contain no sources. Never invent sources."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="How well-established the answer is."
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="What remains uncertain or would be worth researching next.",
    )
