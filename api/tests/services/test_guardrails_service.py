import re

import pytest

from api.services.guardrails_service import GuardrailsService


@pytest.fixture
def guard():
    return GuardrailsService()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("contact john.doe@example.com now", "contact [EMAIL] now"),
        ("call me on 555-123-4567", "call me on [PHONE]"),
        ("key sk-abcdefghijklmnop1234", "key [API_KEY]"),
        ("aws AKIAIOSFODNN7EXAMPLE", "aws [API_KEY]"),
        ("token ghp_abcdefghijklmnopqrstuvwxyz0123", "token [API_KEY]"),
        ("password = hunter2", "password = [REDACTED]"),
        ("api_key: 'abc123'", "api_key: [REDACTED]"),
        ("john@example.com password=abc", "[EMAIL] password=[REDACTED]"),
    ],
)
def test_sensitive_values_are_masked(guard, text, expected):
    assert guard.mask_text(text) == expected


@pytest.mark.parametrize(
    "text",
    ["What are good sources of vitamin C?", "Is 2000 calories a day enough?"],
)
def test_normal_questions_are_unchanged(guard, text):
    assert guard.mask_input(text) == text


def test_none_and_empty_inputs(guard):
    assert guard.mask_text(None) is None
    assert guard.mask_text("") == ""
    assert guard.mask_input("") == ""
    assert guard.mask_output("") == ""


def test_credit_card_is_fully_masked(guard):
    # Expected to FAIL today: the phone rule runs first and eats most of the number,
    # leaving the last group of digits behind. Fix: put "credit_card" before "phone".
    masked = guard.mask_text("my card is 4111 1111 1111 1111")
    assert not re.search(r"\d{4}", masked)
