#!/usr/bin/env python3
"""CLI runner for the OpenAI Codex code-review plugin.

Reviews the calibration tool's core source files using the OpenAI API and
reports any logic errors, mathematical mistakes, edge cases, or numerical
stability issues found by a second model.

Usage::

    # Basic (text report, all default files):
    python run_codex_review.py

    # Select specific files and model:
    python run_codex_review.py \\
        --files calibration/vehicle/calibration.py calibration/utils/irr.py \\
        --model gpt-4o

    # JSON output (pipe-friendly):
    python run_codex_review.py --output-format json

    # Markdown report (e.g. for a PR comment):
    python run_codex_review.py --output-format markdown > review.md

Exit codes:
    0  Completed successfully (findings are normal, not errors).
    1  Missing API key or 'openai' package not installed.
    2  Every API call failed — no findings could be gathered.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
_SEVERITY_LABEL = {"critical": "CRITICAL", "warning": "WARNING ", "info": "INFO    "}


def _format_text(result) -> str:  # type: ignore[no-untyped-def]
    lines: list[str] = []
    criticals = [f for f in result.findings if f.severity == "critical"]
    warnings  = [f for f in result.findings if f.severity == "warning"]
    infos     = [f for f in result.findings if f.severity == "info"]

    for group in (criticals, warnings, infos):
        for finding in group:
            hint = f"  line {finding.line_hint}" if finding.line_hint else ""
            lines.append(
                f"[{_SEVERITY_LABEL[finding.severity]}] "
                f"{finding.file}{hint}\n"
                f"  ({finding.category}) {finding.description}"
            )

    if result.errors:
        lines.append("")
        lines.append("--- Errors during review ---")
        for err in result.errors:
            lines.append(f"  {err}")

    lines.append("")
    lines.append(
        f"Reviewed {len(result.files_reviewed)} file(s) — "
        f"{len(criticals)} critical, {len(warnings)} warning(s), {len(infos)} info."
    )
    lines.append(f"Model: {result.model_used}  |  Tokens used: {result.total_tokens_used:,}")
    return "\n".join(lines)


def _format_markdown(result) -> str:  # type: ignore[no-untyped-def]
    lines: list[str] = ["# AI Code Review Report", ""]
    lines.append(
        f"**Model:** `{result.model_used}`  |  "
        f"**Files reviewed:** {len(result.files_reviewed)}  |  "
        f"**Tokens:** {result.total_tokens_used:,}"
    )
    lines.append("")

    # Summary table
    criticals = sum(1 for f in result.findings if f.severity == "critical")
    warnings  = sum(1 for f in result.findings if f.severity == "warning")
    infos     = sum(1 for f in result.findings if f.severity == "info")
    lines += [
        "## Summary",
        "",
        f"| Severity | Count |",
        f"|----------|-------|",
        f"| Critical | {criticals} |",
        f"| Warning  | {warnings} |",
        f"| Info     | {infos} |",
        "",
    ]

    if result.findings:
        lines += [
            "## Findings",
            "",
            "| Severity | File | Line | Category | Description |",
            "|----------|------|------|----------|-------------|",
        ]
        for f in sorted(result.findings, key=lambda x: _SEVERITY_ORDER[x.severity]):
            hint = f.line_hint or "—"
            desc = f.description.replace("|", "\\|")
            lines.append(f"| {f.severity} | `{f.file}` | {hint} | {f.category} | {desc} |")
        lines.append("")

    if result.errors:
        lines += ["## Errors", ""]
        for err in result.errors:
            lines.append(f"- {err}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review calibration source files with the OpenAI API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
        help="Source files to review (relative to --base-path). "
             "Defaults to the 7 core calibration files.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        metavar="MODEL",
        help="OpenAI model to use (default: gpt-4o).",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "markdown"],
        default="text",
        dest="output_format",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--base-path",
        default=None,
        metavar="DIR",
        dest="base_path",
        help="Repo root directory (default: directory of this script).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        dest="api_key",
        help="OpenAI API key (default: OPENAI_API_KEY env var).",
    )
    args = parser.parse_args()

    base_path = Path(args.base_path) if args.base_path else Path(__file__).parent

    try:
        from calibration.plugins.openai_codex import CodexReviewer  # noqa: PLC0415
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    reviewer = CodexReviewer(
        model=args.model,
        files=args.files,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        base_path=base_path,
    )

    try:
        result = reviewer.review()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output_format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    elif args.output_format == "markdown":
        print(_format_markdown(result))
    else:
        print(_format_text(result))

    # Exit 2 if every file errored and we got no findings
    if result.errors and not result.files_reviewed:
        sys.exit(2)


if __name__ == "__main__":
    main()
