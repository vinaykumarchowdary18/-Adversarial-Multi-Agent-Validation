"""
core/config.py — load and validate all env vars at startup.
Fails loud so you know immediately if a key is missing.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # Keys
    gemini_api_key: str
    groq_api_key: str
    openrouter_api_key: str
    github_token: str
    tavily_api_key: str

    # Models
    gemini_model: str
    groq_model: str
    openrouter_model: str
    github_model: str

    # Debate tuning
    max_debate_rounds: int
    min_consensus_score: float
    log_level: str
    outputs_dir: str


def load_config() -> Config:
    missing = []
    required = [
        "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
        "GITHUB_TOKEN", "TAVILY_API_KEY",
    ]
    for key in required:
        if not os.getenv(key):
            missing.append(key)
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example → .env and fill in your keys."
        )

    return Config(
        gemini_api_key=os.environ["GEMINI_API_KEY"],
        groq_api_key=os.environ["GROQ_API_KEY"],
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
        github_token=os.environ["GITHUB_TOKEN"],
        tavily_api_key=os.environ["TAVILY_API_KEY"],
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free"),
        github_model=os.getenv("GITHUB_MODEL", "openai/gpt-4o-mini"),
        max_debate_rounds=int(os.getenv("MAX_DEBATE_ROUNDS", "2")),
        min_consensus_score=float(os.getenv("MIN_CONSENSUS_SCORE", "0.75")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        outputs_dir=os.getenv("OUTPUTS_DIR", "./outputs"),
    )
