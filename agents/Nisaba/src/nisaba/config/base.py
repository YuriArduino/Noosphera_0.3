"""
Base Configuration Overrides — Nisaba Agent.

Refines the shared infrastructure specifically for the Nisaba Agent,
applying service-specific prefixes and isolation rules.
"""

from pydantic_settings import SettingsConfigDict

from agents.shared.config.base import SharedBaseSettings


class NisabaBaseSettings(SharedBaseSettings):
    """
    Nisaba-specific configuration base.

    Inherits global SST logic from SharedBaseSettings while isolating
    environment variables using the 'NISABA_' prefix.

    Features:
        - Inherits global .env discovery from the shared infrastructure.
        - Enforces service isolation via environment prefixes.

    Design rationale:
        - Using a prefix allows multiple agents (Nisaba, Thoth, etc.) to share
          the same global .env file without variable name collisions.
        - Overriding model_config here maintains the "DRY" (Don't Repeat Yourself)
          principle by reusing the parent's encoding and validation logic.

    Use cases:
        - Defining specific database URLs for Nisaba (NISABA_DATABASE_URL).
        - Configuring agent-specific LLM parameters and thresholds.
    """

    model_config = SettingsConfigDict(env_prefix="NISABA_")
