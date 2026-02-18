from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    newsapi_key: str


def load_config() -> AppConfig:
    load_dotenv()
    key = os.getenv("NEWSAPI_KEY", "").strip()
    if not key:
        raise ValueError("Missing NEWSAPI_KEY in environment or .env file.")
    return AppConfig(newsapi_key=key)


def assert_research_acknowledged(ack_research_only: bool) -> None:
    if not ack_research_only:
        raise ValueError(
            "You must acknowledge research-only usage with --ack-research-only. "
            "This tool does not provide investment advice."
        )
