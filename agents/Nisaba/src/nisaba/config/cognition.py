"""
Cognitive Processing Configuration — Nisaba Agent.

Defines the parameters for reasoning windows, reflection cycles, and
Human-in-the-Loop (HITL) triggers specific to Nisaba's logic.
"""

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from agents.shared.config.base import SharedBaseSettings
from agents.shared.config.memory import memory_settings


class NisabaCognitionSettings(SharedBaseSettings):
    """
    Nisaba-specific Reasoning and Reflection settings.

    This class defines how Nisaba processes information over time,
    independent of the underlying storage hardware.

    Features:
        - HITL Thresholding for confidence-based intervention.
        - Reflection model selection for background consolidation.
        - Windowed memory management to optimize context window usage.

    Use cases:
        - Triggering a manual review when confidence drops below 50.0.
        - Using a small model (Nemotron) for background reflection to save costs.

    Design rationale:
        - Separation of 'Cognition' from 'Memory' allows the agent to change
          its reasoning style without affecting the database connection.
    """

    # ---------------------------------------------------------------------------
    # HITL & CONFIDENCE
    # ---------------------------------------------------------------------------

    HITL_THRESHOLD: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Confidence score below which human intervention is required.",
    )

    # ---------------------------------------------------------------------------
    # REASONING WINDOWS
    # ---------------------------------------------------------------------------

    MEMORY_WINDOW_SIZE: int = Field(
        default=10,
        description="Number of recent turns to maintain in active reasoning context.",
    )

    # ---------------------------------------------------------------------------
    # REFLECTION LAYER
    # ---------------------------------------------------------------------------

    REFLECTION_ENABLED: bool = Field(
        default=True,
        description="Enables background processing of trajectories into insights.",
    )

    REFLECTION_MODEL: str = Field(
        default="nvidia/nemotron-3-nano-4b",
        description="The specific model used for the reflection/summarization task.",
    )

    # Apply Nisaba prefix to all cognition fields
    model_config = SettingsConfigDict(env_prefix="NISABA_", frozen=True)


# Global instance for Nisaba's cognitive logic
nisaba_cognition = NisabaCognitionSettings()

__all__ = ["NisabaCognitionSettings", "nisaba_cognition", "memory_settings"]
