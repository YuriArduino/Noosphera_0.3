"""
LLM Configuration Overrides — Nisaba Agent.

Defines the specific cognitive parameters, reasoning modes, and
specializations for the Nisaba Agent.
"""

from typing import List
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from agents.shared.config.llm import SharedLLMSettings


class NisabaLLMSettings(SharedLLMSettings):
    """
    Nisaba-specific LLM parameters.

    Refines the LLM behavior for psychoanalytic orchestration and
    high-context cognitive tasks.

    Features:
        - Independent temperature control for creative reasoning.
        - Specialized reasoning modes (hybrid/symbolic).
        - Domain-specific expertise injection.

    Use cases:
        - Adjusting CHAT_TEMPERATURE to 0.5 for balanced, consistent analysis.
        - Defining 'hybrid' reasoning to combine vector retrieval with logic.

    Design rationale:
        - Inheriting from SharedLLMSettings ensures that if the global
          AGENT_LLM_BASE_URL changes in the root .env, Nisaba follows automatically.
    """

    # ---------------------------------------------------------------------------
    # COGNITIVE PARAMETERS
    # ---------------------------------------------------------------------------

    CHAT_TEMPERATURE: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Control for randomness: 0.0 is deterministic, 1.0+ is creative.",
    )

    AGENT_REASONING_MODE: str = Field(
        default="hybrid",
        description="The logic framework used by the agent (e.g., hybrid, reactive).",
    )

    AGENT_SPECIALIZATION: List[str] = Field(
        default_factory=lambda: ["semantic analysis", "symbolic interpretation"],
        description="Areas of expertise injected into the system prompt.",
    )

    # Apply Nisaba prefix to all inherited and new fields
    model_config = SettingsConfigDict(env_prefix="NISABA_", frozen=True)


# Global instance for Nisaba
nisaba_llm_settings = NisabaLLMSettings()
llm_settings = nisaba_llm_settings

__all__ = ["NisabaLLMSettings", "nisaba_llm_settings", "llm_settings"]
