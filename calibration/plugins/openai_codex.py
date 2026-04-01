"""OpenAI Codex / GPT-4 code review plugin.

Uses the OpenAI API to review the calibration tool's core financial source
files and identify logic errors, mathematical mistakes, edge cases, and
numerical stability issues — providing a second model's perspective.

Install the optional dependency::

    pip install 'calibration-tool[codex]'

Then set your API key::

    export OPENAI_API_KEY=sk-...

Quick usage::

    from calibration.plugins.openai_codex import CodexReviewer
    result = CodexReviewer().review()
    for f in result.findings:
        print(f.severity, f.file, f.description)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert Python quant developer specialising in financial mathematics, "
    "Monte Carlo methods, and numerical analysis. You are reviewing production code for "
    "a blended-finance capital allocation tool.\n\n"
    "Your response MUST be a valid JSON array (may be empty []) where each element has "
    "exactly these keys:\n"
    '  "severity": "critical" | "warning" | "info"\n'
    '  "line_hint": "<line number or range string, or null>"\n'
    '  "category": "math" | "edge_case" | "numerical_stability" | "logic" | "other"\n'
    '  "description": "<concise explanation of the issue>"\n\n'
    "Do NOT output any prose, markdown, or code fences — only the raw JSON array."
)

USER_PROMPT_TEMPLATE = """\
Review the following Python source file for correctness issues: {filename}

Focus EXCLUSIVELY on:
1. Mathematical correctness (GBM drift/variance correction, NPV formulas, IRR \
bracket logic, waterfall arithmetic, LP formulation, etc.)
2. Numerical stability (overflow, underflow, division-by-zero, catastrophic \
cancellation, ill-conditioned matrices)
3. Edge cases that could cause silent wrong results (empty arrays, zero \
denominators, off-by-one, single-scenario degenerate paths, etc.)
4. Logic errors in the waterfall, calibration, or optimisation formulations

Do NOT flag: style issues, naming conventions, PEP 8, missing docstrings, \
type hints, or performance concerns unless they cause correctness problems.

Source code:
```python
{source}
```
"""


class ReviewFinding(BaseModel):
    """A single finding from the AI code review."""

    severity: Literal["critical", "warning", "info"] = Field(
        ..., description="Severity level of the finding."
    )
    file: str = Field(..., description="Source file path that was reviewed.")
    line_hint: str | None = Field(
        None, description="Line number or range hint, if provided by the model."
    )
    category: Literal["math", "edge_case", "numerical_stability", "logic", "other"] = Field(
        ..., description="Category of the issue."
    )
    description: str = Field(..., description="Human-readable description of the issue.")


@dataclass
class ReviewResult:
    """Aggregated output from a plugin review run."""

    findings: list[ReviewFinding] = field(default_factory=list)
    files_reviewed: list[str] = field(default_factory=list)
    model_used: str = ""
    total_tokens_used: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "findings": [f.model_dump() for f in self.findings],
            "files_reviewed": self.files_reviewed,
            "model_used": self.model_used,
            "total_tokens_used": self.total_tokens_used,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Models that support response_format={"type": "json_object"}
_JSON_MODE_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-turbo-preview"}


def _supports_json_mode(model: str) -> bool:
    return any(m in model for m in _JSON_MODE_MODELS)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if the model wrapped the JSON in them."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _get_openai_client(api_key: str):  # type: ignore[return]
    """Lazy-import and instantiate the OpenAI client.

    This is the single location where ``import openai`` lives so that:
    - Importing this module never fails even without the package installed.
    - Tests can mock this function easily.
    """
    try:
        import openai  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is not installed. "
            "Run: pip install 'calibration-tool[codex]'"
        ) from exc
    return openai.OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CodexReviewer:
    """Reviews calibration source files using the OpenAI API.

    Each file is sent in a separate API call so the model can focus and provide
    precise, file-level feedback. Errors from individual calls are accumulated
    in ``ReviewResult.errors`` rather than raising an exception.

    Parameters
    ----------
    model:
        OpenAI model to use. Defaults to ``"gpt-4o"``.
    files:
        List of file paths to review (relative to ``base_path``).
        Defaults to :attr:`DEFAULT_FILES`.
    api_key:
        OpenAI API key. Defaults to the ``OPENAI_API_KEY`` environment variable.
    base_path:
        Root directory from which relative file paths are resolved.
        Defaults to the repo root (three levels above this file).
    """

    DEFAULT_FILES: list[str] = [
        "calibration/project/simulation.py",
        "calibration/vehicle/calibration.py",
        "calibration/vehicle/capital_stack.py",
        "calibration/vehicle/risk_mitigants.py",
        "calibration/portfolio/optimizer.py",
        "calibration/utils/irr.py",
        "calibration/utils/stats.py",
    ]

    def __init__(
        self,
        model: str = "gpt-4o",
        files: list[str] | None = None,
        api_key: str | None = None,
        base_path: Path | None = None,
    ) -> None:
        self.model = model
        self.files = files if files is not None else list(self.DEFAULT_FILES)
        self.api_key = api_key
        self.base_path = base_path or Path(__file__).parent.parent.parent

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def review(self) -> ReviewResult:
        """Run the code review and return aggregated findings.

        Raises
        ------
        RuntimeError
            If the API key is missing or the ``openai`` package is not installed.
        """
        import os  # noqa: PLC0415

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key not set. "
                "Pass api_key= or export OPENAI_API_KEY=sk-..."
            )

        client = _get_openai_client(api_key)

        result = ReviewResult(model_used=self.model)

        for rel_path in self.files:
            source = self._load_file(Path(rel_path))
            if source is None:
                result.errors.append(f"File not found: {rel_path}")
                continue

            result.files_reviewed.append(rel_path)
            try:
                response_text, tokens = self._call_api(client, rel_path, source)
                result.total_tokens_used += tokens
                findings = self._parse_response(rel_path, response_text)
                result.findings.extend(findings)
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"{rel_path}: {exc}")

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_file(self, rel_path: Path) -> str | None:
        full = self.base_path / rel_path
        if not full.exists():
            return None
        return full.read_text(encoding="utf-8")

    def _build_prompt(self, filename: str, source: str) -> str:
        return USER_PROMPT_TEMPLATE.format(filename=filename, source=source)

    def _call_api(self, client, filename: str, source: str) -> tuple[str, int]:
        """Call the OpenAI API and return (response_text, tokens_used)."""
        prompt = self._build_prompt(filename, source)
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        if _supports_json_mode(self.model):
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens

    def _parse_response(self, filename: str, text: str) -> list[ReviewFinding]:
        """Parse the model's JSON response into ReviewFinding objects."""
        cleaned = _strip_fences(text)
        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON parse error: {exc}. Raw response: {cleaned[:200]}") from exc

        # The model may return a dict with a top-level key wrapping the array
        # (common when json_object mode is used with gpt-4o).
        if isinstance(raw, dict):
            # Find the first list value
            for v in raw.values():
                if isinstance(v, list):
                    raw = v
                    break
            else:
                raw = []

        if not isinstance(raw, list):
            raise ValueError(f"Expected JSON array, got {type(raw).__name__}")

        findings: list[ReviewFinding] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                findings.append(
                    ReviewFinding(
                        file=filename,
                        severity=item.get("severity", "info"),
                        line_hint=item.get("line_hint"),
                        category=item.get("category", "other"),
                        description=item.get("description", ""),
                    )
                )
            except Exception:  # noqa: BLE001 — skip malformed items silently
                pass

        return findings
