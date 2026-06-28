"""Unit tests for the language parameter on the personality compose service.

These tests cover:

- :class:`backend.models.ComposeRequest` rejects invalid language codes.
- :func:`backend.services.personality.compose_as_profile` actually appends the
  resolved language name to the system prompt — for every language code the
  frontend can send.
- The default ``language="en"`` is honored when the caller omits the body.

Why this exists
---------------
The compose endpoint takes the user's selected language and appends
``"Output language: {name}"`` to the system prompt so the local Qwen3 LLM
generates the utterance in the correct tongue. ``name`` is looked up in
``backend.backends.LANGUAGE_CODE_TO_NAME``, which historically only
contained 10 Qwen3 codes. The ``ComposeRequest.pattern`` matches a larger
24-code set (matching ``VoiceProfileCreate.language``), so a user with a
Hebrew / Arabic / Hindi / Korean / Persian / Turkish / etc. profile would
otherwise get a system prompt containing the literal string
``"Output language: None"`` and the LLM would answer in English (or
default) instead of the selected language.

These tests pin the contract: every code in ``ComposeRequest.pattern``
must resolve to a non-empty language name in the system prompt.
"""

import asyncio
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend import models
from backend.backends import LANGUAGE_CODE_TO_NAME
from backend.services import personality

# ---------------------------------------------------------------------------
# Extract the supported language codes straight from ComposeRequest.pattern
# so this test stays in sync if the set ever changes.
# ---------------------------------------------------------------------------

_FIELD_INFO = models.ComposeRequest.model_fields["language"]
# pydantic v2 keeps the regex on the metadata entries; older v1 put it on the Field itself.
_COMPOSE_PATTERN_RE = getattr(_FIELD_INFO, "pattern", None) or next(
    (m.pattern for m in (_FIELD_INFO.metadata or []) if hasattr(m, "pattern")),
    None,
)
# Pattern looks like "^(zh|en|...|cs)$" — strip the anchors and parens.
if _COMPOSE_PATTERN_RE:
    _INNER = _COMPOSE_PATTERN_RE.strip("^$").lstrip("(").rstrip(")")
    SUPPORTED_LANGUAGE_CODES = sorted(_INNER.split("|"))
else:
    SUPPORTED_LANGUAGE_CODES = []


# ---------------------------------------------------------------------------
# ComposeRequest validation
# ---------------------------------------------------------------------------


class TestComposeRequestPattern:
    def test_default_language_is_english(self):
        req = models.ComposeRequest()
        assert req.language == "en"

    def test_accepts_every_supported_code(self):
        for code in SUPPORTED_LANGUAGE_CODES:
            req = models.ComposeRequest(language=code)
            assert req.language == code, f"pattern rejected valid code {code!r}"

    def test_rejects_unknown_code(self):
        with pytest.raises(ValidationError):
            models.ComposeRequest(language="klingon")

    def test_rejects_empty_string(self):
        with pytest.raises(ValidationError):
            models.ComposeRequest(language="")


# ---------------------------------------------------------------------------
# compose_as_profile — language actually lands in the system prompt
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Stand-in for the Qwen3 backend that records what was asked of it."""

    def __init__(self):
        self.last_system_prompt: str | None = None
        self.model_size = "1.7B"

    async def generate(self, *, prompt, system, max_tokens, temperature, model_size):
        self.last_system_prompt = system
        return "stub output"


@pytest.fixture
def fake_llm_backend(monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(personality.llm_service, "get_llm_model", lambda: backend)
    return backend


class TestComposeLanguageInterpolation:
    def test_every_supported_code_resolves_to_a_real_language_name(self, fake_llm_backend):
        """Regression test for the LANGUAGE_CODE_TO_NAME gap.

        Prior to the fix, 14 of the 24 supported codes returned ``None`` from
        ``LANGUAGE_CODE_TO_NAME.get(...)`` and the system prompt contained
        ``"Output language: None"`` — silently defeating the whole feature for
        any user with a non-Qwen engine profile.
        """
        missing = [c for c in SUPPORTED_LANGUAGE_CODES if not LANGUAGE_CODE_TO_NAME.get(c)]
        assert not missing, (
            f"ComposeRequest accepts {missing} but LANGUAGE_CODE_TO_NAME has no entry for them; "
            f"the system prompt would say 'Output language: None'. "
            f"Either narrow ComposeRequest.pattern or extend LANGUAGE_CODE_TO_NAME."
        )

    @pytest.mark.parametrize("code", SUPPORTED_LANGUAGE_CODES)
    def test_system_prompt_contains_resolved_language_name(self, fake_llm_backend, code):
        """Every supported code produces a system prompt naming that language."""
        asyncio.run(
            personality.compose_as_profile(
                personality="a grumpy dockworker",
                data=models.ComposeRequest(language=code),
            )
        )
        name = LANGUAGE_CODE_TO_NAME.get(code)
        assert name is not None, f"language {code!r} has no name in LANGUAGE_CODE_TO_NAME"
        prompt = fake_llm_backend.last_system_prompt or ""
        assert f"Output language: {name}" in prompt, (
            f"system prompt for language {code!r} did not contain the resolved name {name!r}; got: {prompt!r}"
        )
        # And never the broken literal — defense in depth.
        assert "Output language: None" not in prompt

    def test_default_language_uses_english(self, fake_llm_backend):
        """Caller omitting the request body still gets English output."""
        asyncio.run(personality.compose_as_profile(personality="a quiet librarian"))
        prompt = fake_llm_backend.last_system_prompt or ""
        assert "Output language: english" in prompt
        assert "Output language: None" not in prompt

    def test_missing_personality_raises_value_error(self, fake_llm_backend):
        """Existing contract: empty personality string -> 400 from the route."""
        with pytest.raises(ValueError, match="no personality"):
            asyncio.run(personality.compose_as_profile(personality=None))

    def test_compose_prompt_is_not_silently_truncated(self, fake_llm_backend):
        """Compose output is generated with the compose-mode temperature and trigger turn."""
        asyncio.run(
            personality.compose_as_profile(
                personality="a cheerful baker",
                data=models.ComposeRequest(language="de"),
            )
        )
        # Compose mode is "hot" (0.9) for variety. The trigger user turn is "Speak."
        prompt = fake_llm_backend.last_system_prompt or ""
        assert "Produce one short utterance" in prompt
        # Language sentence lives at the end so it lands as the final instruction.
        assert prompt.rstrip().endswith("Output language: german")
