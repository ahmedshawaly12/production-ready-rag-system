import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailPattern:
    name: str
    pattern: re.Pattern[str]
    replacement: str


class GuardrailsService:
    """Detect and mask common sensitive data in text."""

    def __init__(self) -> None:
        self.patterns = [
            GuardrailPattern(
                name="email",
                pattern=re.compile(
                    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
                ),
                replacement="[EMAIL]",
            ),
            GuardrailPattern(
                name="credit_card",
                pattern=re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
                replacement="[POSSIBLE_CARD_NUMBER]",
            ),
            GuardrailPattern(
                name="phone",
                pattern=re.compile(
                    r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?"
                    r"(?:\(?\d{2,4}\)?[\s.-]?)"
                    r"\d{3,4}[\s.-]?\d{3,4}(?!\w)"
                ),
                replacement="[PHONE]",
            ),
            GuardrailPattern(
                name="api_key",
                pattern=re.compile(
                    r"\b(?:sk-[A-Za-z0-9_-]{16,}|"
                    r"AKIA[A-Z0-9]{16}|"
                    r"gh[pousr]_[A-Za-z0-9]{20,})\b"
                ),
                replacement="[API_KEY]",
            ),
            GuardrailPattern(
                name="secret_assignment",
                pattern=re.compile(
                    r"""(?i)\b(api[_-]?key|access[_-]?token|"""
                    r"""refresh[_-]?token|client[_-]?secret|"""
                    r"""password|passwd|secret)\b"""
                    r"""(\s*[:=]\s*)"""
                    r"""("[^"]*"|'[^']*'|[^\s,;]+)"""
                ),
                replacement=None,  # Constructed below per match.
            ),
        ]

    def mask_text(self, text: str | None) -> str | None:
        """Mask detected sensitive values without mutating the input."""
        if text is None or not text:
            return text

        for rule in self.patterns:
            if rule.name == "secret_assignment":
                text = rule.pattern.sub(
                    lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
                    text,
                )
            else:
                text = rule.pattern.sub(rule.replacement, text)

        return text

    def mask_input(self, text: str) -> str:
        return self.mask_text(text) or ""

    def mask_output(self, text: str) -> str:
        return self.mask_text(text) or ""
