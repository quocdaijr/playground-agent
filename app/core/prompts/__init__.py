from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template from a Markdown file by name (without extension)."""
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt '{name}' not found at {path}")
    return path.read_text(encoding="utf-8").strip()


# Default system prompt — fallback and backward-compat module-level constant
SYSTEM_PROMPT = load_prompt("system")


def load_system_prompt(provider: str) -> str:
    """
    Load the system prompt for the given LLM provider.

    Looks for ``system_{provider}.md`` first (e.g. ``system_anthropic.md``);
    falls back to the default ``system.md`` if not found.
    """
    try:
        return load_prompt(f"system_{provider.lower()}")
    except FileNotFoundError:
        return SYSTEM_PROMPT
