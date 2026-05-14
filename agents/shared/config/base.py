from pathlib import Path
from typing import Optional, Union

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class SharedBaseSettings(BaseSettings):
    """
    Global Noosphera infrastructure settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )


class PathMixin:

    @staticmethod
    def resolve_path(
        path: Union[str, Path],
        base: Optional[Path] = None,
    ) -> Path:

        p = Path(path) if isinstance(path, str) else path

        if base and not p.is_absolute():
            p = base / p

        p.mkdir(parents=True, exist_ok=True)

        return p
