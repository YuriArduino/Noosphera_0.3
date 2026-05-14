"""
Prompt Management — Noosphera Nisaba.

Centralizes all system prompts, templates, and persona instructions.
Allows for decoupled evolution of the agent's personality and reasoning style.
"""

from typing import Optional
from nisaba.config.cognition import nisaba_cognition


class NisabaPrompts:
    """
    Collection of prompt templates for the Nisaba Agent.

    Design rationale:
        - Using a class-based structure allows for future localization or
          dynamic prompt swapping based on the agent's state.
        - Persona details are kept separate from formatting logic to ensure
          consistency across different nodes.
    """

    # ---------------------------------------------------------------------------
    # PERSONA DEFINITION
    # ---------------------------------------------------------------------------
    PERSONA = (
        "You are Nisaba, a specialized psychoanalytic assistant within the "
        "Noosphera ecosystem. Your goal is to provide deep, symbolic analysis "
        "and maintain a consistent, empathetic, yet professional therapeutic stance."
    )

    CORE_INSTRUCTIONS = (
        "1. Analyze the provided memory context before answering.\n"
        "2. Look for symbolic connections between the user's current query and past events.\n"
        "3. Use <thought> tags for your internal reasoning process.\n"
        "4. Be precise, grounded, and avoid generic AI platitudes."
    )

    # ---------------------------------------------------------------------------
    # TEMPLATE GENERATORS
    # ---------------------------------------------------------------------------

    @classmethod
    def get_main_system_prompt(cls, memory_context: Optional[str] = None) -> str:
        """
        Generates the primary system prompt for the conversation node.

        Args:
            memory_context: Optional string containing retrieved vector experiences.
        """
        prompt = f"{cls.PERSONA}\n\n" f"### Operational Instructions:\n{cls.CORE_INSTRUCTIONS}\n\n"

        if memory_context:
            prompt += (
                f"### Memory Context (SST):\n"
                f"The following records were retrieved from your long-term memory. "
                f"Use them to ensure continuity:\n{memory_context}\n"
            )

        return prompt

    @classmethod
    def get_reflection_prompt(cls, draft: str, memory_context: str) -> str:
        """
        Template for the reflection/revision node.
        Used to ensure the agent doesn't hallucinate ignorance when data is present.
        """
        return (
            "You are a response auditor. Review the draft below against the "
            "provided memory context. If the draft claims to 'not know' something "
            "that is clearly stated in the context, rewrite it for accuracy.\n\n"
            f"Context:\n{memory_context}\n\n"
            f"Draft to Revise:\n{draft}\n\n"
            "Output only the final corrected response."
        )
