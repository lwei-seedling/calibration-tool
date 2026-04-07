"""Tests for the OpenAI Codex review plugin.

All tests use mocked API calls — no real OpenAI requests are made.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from calibration.plugins.openai_codex import (
    CodexReviewer,
    ReviewFinding,
    ReviewResult,
    _strip_fences,
    _supports_json_mode,
)


# ---------------------------------------------------------------------------
# ReviewFinding model tests
# ---------------------------------------------------------------------------

class TestReviewFinding:
    def test_valid_finding(self):
        f = ReviewFinding(
            severity="critical",
            file="calibration/utils/irr.py",
            line_hint="42",
            category="math",
            description="Off-by-one in IRR bracket.",
        )
        assert f.severity == "critical"
        assert f.file == "calibration/utils/irr.py"
        assert f.line_hint == "42"

    def test_line_hint_optional(self):
        f = ReviewFinding(severity="info", file="x.py", category="other", description="ok")
        assert f.line_hint is None

    def test_invalid_severity_raises(self):
        with pytest.raises(ValidationError):
            ReviewFinding(severity="blocker", file="x.py", category="math", description="x")

    def test_invalid_category_raises(self):
        with pytest.raises(ValidationError):
            ReviewFinding(severity="info", file="x.py", category="typo", description="x")

    def test_all_valid_severities(self):
        for sev in ("critical", "warning", "info"):
            f = ReviewFinding(severity=sev, file="x.py", category="other", description="d")
            assert f.severity == sev

    def test_all_valid_categories(self):
        for cat in ("math", "edge_case", "numerical_stability", "logic", "other"):
            f = ReviewFinding(severity="info", file="x.py", category=cat, description="d")
            assert f.category == cat


# ---------------------------------------------------------------------------
# ReviewResult dataclass tests
# ---------------------------------------------------------------------------

class TestReviewResult:
    def test_default_fields(self):
        r = ReviewResult()
        assert r.findings == []
        assert r.files_reviewed == []
        assert r.model_used == ""
        assert r.total_tokens_used == 0
        assert r.errors == []

    def test_to_dict_structure(self):
        r = ReviewResult(
            findings=[ReviewFinding(severity="info", file="a.py", category="math", description="x")],
            files_reviewed=["a.py"],
            model_used="gpt-4o",
            total_tokens_used=100,
        )
        d = r.to_dict()
        assert "findings" in d
        assert "files_reviewed" in d
        assert d["model_used"] == "gpt-4o"
        assert d["total_tokens_used"] == 100
        assert isinstance(d["findings"][0], dict)
        assert d["findings"][0]["severity"] == "info"


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_strip_fences_json(self):
        assert _strip_fences("```json\n[]\n```") == "[]"

    def test_strip_fences_plain(self):
        assert _strip_fences("```\n[]\n```") == "[]"

    def test_strip_fences_no_fences(self):
        assert _strip_fences("[]") == "[]"

    def test_strip_fences_strips_whitespace(self):
        assert _strip_fences("  []  ") == "[]"

    def test_supports_json_mode_gpt4o(self):
        assert _supports_json_mode("gpt-4o") is True

    def test_supports_json_mode_turbo(self):
        assert _supports_json_mode("gpt-4-turbo") is True

    def test_supports_json_mode_plain_gpt4(self):
        assert _supports_json_mode("gpt-4") is False


# ---------------------------------------------------------------------------
# CodexReviewer initialisation tests
# ---------------------------------------------------------------------------

class TestCodexReviewerInit:
    def test_default_files_populated(self):
        r = CodexReviewer()
        assert len(r.files) == 7
        assert "calibration/project/simulation.py" in r.files

    def test_custom_files_override(self):
        custom = ["calibration/utils/irr.py"]
        r = CodexReviewer(files=custom)
        assert r.files == custom

    def test_default_model(self):
        r = CodexReviewer()
        assert r.model == "gpt-4o"

    def test_custom_model(self):
        r = CodexReviewer(model="gpt-4-turbo")
        assert r.model == "gpt-4-turbo"

    def test_base_path_resolves(self):
        r = CodexReviewer()
        # base_path should point to the repo root (parent of calibration/)
        assert (r.base_path / "calibration").is_dir()

    def test_custom_base_path(self, tmp_path):
        r = CodexReviewer(base_path=tmp_path)
        assert r.base_path == tmp_path


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------

class TestCodexReviewerPrompt:
    def test_prompt_contains_filename(self):
        r = CodexReviewer()
        p = r._build_prompt("calibration/utils/irr.py", "def foo(): pass")
        assert "calibration/utils/irr.py" in p

    def test_prompt_contains_source(self):
        r = CodexReviewer()
        p = r._build_prompt("x.py", "MY_UNIQUE_CODE_123")
        assert "MY_UNIQUE_CODE_123" in p

    def test_prompt_mentions_json(self):
        from calibration.plugins.openai_codex import SYSTEM_PROMPT
        assert "JSON" in SYSTEM_PROMPT

    def test_prompt_mentions_severity_levels(self):
        from calibration.plugins.openai_codex import SYSTEM_PROMPT
        assert "critical" in SYSTEM_PROMPT
        assert "warning" in SYSTEM_PROMPT
        assert "info" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------

class TestParseResponse:
    def setup_method(self):
        self.reviewer = CodexReviewer()

    def test_parses_valid_array(self):
        payload = json.dumps([
            {"severity": "warning", "line_hint": "10", "category": "math",
             "description": "Missing Itô correction."},
        ])
        findings = self.reviewer._parse_response("irr.py", payload)
        assert len(findings) == 1
        assert findings[0].severity == "warning"
        assert findings[0].file == "irr.py"

    def test_handles_empty_array(self):
        findings = self.reviewer._parse_response("irr.py", "[]")
        assert findings == []

    def test_handles_markdown_fence(self):
        payload = "```json\n[]\n```"
        findings = self.reviewer._parse_response("irr.py", payload)
        assert findings == []

    def test_handles_dict_wrapper(self):
        # gpt-4o json_object mode sometimes wraps in a top-level dict
        payload = json.dumps({"findings": [
            {"severity": "info", "line_hint": None, "category": "other",
             "description": "Note."},
        ]})
        findings = self.reviewer._parse_response("irr.py", payload)
        assert len(findings) == 1

    def test_handles_malformed_json(self):
        with pytest.raises(ValueError, match="JSON parse error"):
            self.reviewer._parse_response("irr.py", "not json {{{")

    def test_skips_malformed_items(self):
        payload = json.dumps([
            {"severity": "INVALID_SEV", "category": "math", "description": "x"},
            {"severity": "info", "line_hint": None, "category": "other", "description": "ok"},
        ])
        findings = self.reviewer._parse_response("irr.py", payload)
        # The invalid item is silently skipped; valid item is kept
        assert len(findings) == 1
        assert findings[0].severity == "info"

    def test_missing_description_defaults_empty(self):
        # description key absent → _parse_response supplies "" via .get() default
        payload = json.dumps([{"severity": "warning", "category": "math"}])
        findings = self.reviewer._parse_response("irr.py", payload)
        assert len(findings) == 1
        assert findings[0].description == ""


# ---------------------------------------------------------------------------
# review() integration tests (mocked OpenAI client)
# ---------------------------------------------------------------------------

def _make_mock_client(response_text: str, tokens: int = 50):
    """Build a mock openai.OpenAI client that returns a fixed response."""
    mock_usage = MagicMock()
    mock_usage.total_tokens = tokens

    mock_choice = MagicMock()
    mock_choice.message.content = response_text

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


class TestCodexReviewerReview:
    def test_review_happy_path(self, tmp_path):
        # Create a minimal fake source file
        fake_file = tmp_path / "calibration" / "utils" / "irr.py"
        fake_file.parent.mkdir(parents=True)
        fake_file.write_text("def dummy(): pass")

        response = json.dumps([
            {"severity": "warning", "line_hint": "1", "category": "math",
             "description": "Dummy finding."},
        ])
        mock_client = _make_mock_client(response, tokens=100)

        with patch("calibration.plugins.openai_codex._get_openai_client", return_value=mock_client):
            reviewer = CodexReviewer(
                files=["calibration/utils/irr.py"],
                api_key="sk-fake",
                base_path=tmp_path,
            )
            result = reviewer.review()

        assert len(result.findings) == 1
        assert result.findings[0].severity == "warning"
        assert result.total_tokens_used == 100
        assert "calibration/utils/irr.py" in result.files_reviewed
        assert result.errors == []

    def test_review_missing_api_key_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        reviewer = CodexReviewer(files=[], api_key=None, base_path=tmp_path)
        with pytest.raises(RuntimeError, match="API key"):
            reviewer.review()

    def test_review_missing_openai_package(self, tmp_path, monkeypatch):
        # Simulate openai not being installed: _get_openai_client raises RuntimeError.
        # review() propagates this immediately (client is created before the file loop).
        def _raise_import(*_args, **_kwargs):
            raise RuntimeError("The 'openai' package is not installed.")

        with patch("calibration.plugins.openai_codex._get_openai_client", side_effect=_raise_import):
            reviewer = CodexReviewer(files=[], api_key="sk-fake", base_path=tmp_path)
            with pytest.raises(RuntimeError, match="openai"):
                reviewer.review()

    def test_review_api_failure_accumulates_error(self, tmp_path):
        fake_file = tmp_path / "calibration" / "utils" / "irr.py"
        fake_file.parent.mkdir(parents=True)
        fake_file.write_text("def dummy(): pass")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API timeout")

        with patch("calibration.plugins.openai_codex._get_openai_client", return_value=mock_client):
            reviewer = CodexReviewer(
                files=["calibration/utils/irr.py"],
                api_key="sk-fake",
                base_path=tmp_path,
            )
            result = reviewer.review()

        assert result.findings == []
        assert len(result.errors) == 1
        assert "API timeout" in result.errors[0]

    def test_review_missing_file_accumulates_error(self, tmp_path):
        with patch("calibration.plugins.openai_codex._get_openai_client"):
            reviewer = CodexReviewer(
                files=["does/not/exist.py"],
                api_key="sk-fake",
                base_path=tmp_path,
            )
            result = reviewer.review()

        assert result.findings == []
        assert any("not found" in e.lower() for e in result.errors)

    def test_review_multiple_files(self, tmp_path):
        for name in ("irr.py", "stats.py"):
            f = tmp_path / "calibration" / "utils" / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("def dummy(): pass")

        response = json.dumps([
            {"severity": "info", "line_hint": None, "category": "other", "description": "ok"},
        ])
        mock_client = _make_mock_client(response, tokens=30)

        with patch("calibration.plugins.openai_codex._get_openai_client", return_value=mock_client):
            reviewer = CodexReviewer(
                files=["calibration/utils/irr.py", "calibration/utils/stats.py"],
                api_key="sk-fake",
                base_path=tmp_path,
            )
            result = reviewer.review()

        assert len(result.files_reviewed) == 2
        assert len(result.findings) == 2  # one per file
        assert result.total_tokens_used == 60  # 30 per call × 2

    def test_review_result_to_dict_roundtrip(self, tmp_path):
        fake_file = tmp_path / "calibration" / "utils" / "irr.py"
        fake_file.parent.mkdir(parents=True)
        fake_file.write_text("def dummy(): pass")

        response = json.dumps([
            {"severity": "critical", "line_hint": "5", "category": "math",
             "description": "Divide by zero."},
        ])
        mock_client = _make_mock_client(response)

        with patch("calibration.plugins.openai_codex._get_openai_client", return_value=mock_client):
            reviewer = CodexReviewer(
                files=["calibration/utils/irr.py"],
                api_key="sk-fake",
                base_path=tmp_path,
            )
            result = reviewer.review()

        d = result.to_dict()
        assert d["findings"][0]["severity"] == "critical"
        assert d["findings"][0]["category"] == "math"
        # Ensure it is JSON-serialisable
        json.dumps(d)
