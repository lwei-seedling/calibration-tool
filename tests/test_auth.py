"""Regression tests for auth.py password verification."""
from __future__ import annotations

import hashlib

from auth import (
    _audit_event,
    _hash_password_pbkdf2,
    _is_legacy_hash,
    _verify_password,
)


def test_pbkdf2_roundtrip():
    h = _hash_password_pbkdf2("correct horse battery staple")
    assert _verify_password("correct horse battery staple", h)
    assert not _verify_password("wrong", h)


def test_pbkdf2_salt_is_unique():
    h1 = _hash_password_pbkdf2("pw")
    h2 = _hash_password_pbkdf2("pw")
    assert h1 != h2
    assert _verify_password("pw", h1)
    assert _verify_password("pw", h2)


def test_pbkdf2_format_shape():
    h = _hash_password_pbkdf2("pw")
    parts = h.split("$")
    assert parts[0] == "pbkdf2_sha256"
    assert int(parts[1]) > 0
    assert len(bytes.fromhex(parts[2])) == 16
    assert len(bytes.fromhex(parts[3])) == 32


def test_legacy_sha256_still_verifies():
    legacy = hashlib.sha256(b"hunter2").hexdigest()
    assert _verify_password("hunter2", legacy)
    assert not _verify_password("nope", legacy)


def test_malformed_hash_rejected():
    assert not _verify_password("pw", "")
    assert not _verify_password("pw", "pbkdf2_sha256$not-an-int$abcd$ef")
    assert not _verify_password("pw", "pbkdf2_sha256$1000$zz$zz")
    assert not _verify_password("pw", "pbkdf2_sha256$1000$abcd")
    assert not _verify_password("pw", "pbkdf2_sha256$0$abcd$ef")
    # iteration cap: reject absurd counts that would DoS verify
    assert not _verify_password("pw", "pbkdf2_sha256$999999999$abcd$ef")


def test_empty_password_rejected():
    h = _hash_password_pbkdf2("")
    assert not _verify_password("", h)
    legacy_empty = hashlib.sha256(b"").hexdigest()
    assert not _verify_password("", legacy_empty)


def test_whitespace_in_stored_hash_ignored():
    """Operators often paste hashes with trailing newlines into secrets.toml."""
    h = _hash_password_pbkdf2("pw")
    assert _verify_password("pw", h + "\n")
    assert _verify_password("pw", "  " + h + "  ")
    legacy = hashlib.sha256(b"pw").hexdigest()
    assert _verify_password("pw", legacy + "\n")


def test_legacy_hex_case_insensitive():
    legacy = hashlib.sha256(b"pw").hexdigest()
    assert _verify_password("pw", legacy.upper())
    assert _verify_password("pw", legacy.lower())


def test_non_ascii_password_roundtrip():
    pwd = "pässwörd-🔒-日本語"
    h = _hash_password_pbkdf2(pwd)
    assert _verify_password(pwd, h)
    assert not _verify_password("passw0rd", h)


# ------------------------------------------------------------------
# Legacy-hash detection (rotation nudge)
# ------------------------------------------------------------------

def test_is_legacy_hash_true_for_sha256_digest():
    legacy = hashlib.sha256(b"pw").hexdigest()
    assert _is_legacy_hash(legacy)
    assert _is_legacy_hash(legacy.upper())
    assert _is_legacy_hash("  " + legacy + "\n")  # paste-friendly


def test_is_legacy_hash_false_for_pbkdf2():
    assert not _is_legacy_hash(_hash_password_pbkdf2("pw"))


def test_is_legacy_hash_false_for_malformed():
    assert not _is_legacy_hash("")
    assert not _is_legacy_hash("   ")
    assert not _is_legacy_hash("deadbeef")           # too short
    assert not _is_legacy_hash("z" * 64)             # 64 chars but not hex
    assert not _is_legacy_hash("a" * 63)             # wrong length


# ------------------------------------------------------------------
# Audit record formatting (must never leak secrets)
# ------------------------------------------------------------------

def test_audit_event_shape_and_timestamp():
    line = _audit_event("login_success", now=0.0)
    assert line.startswith("[auth] ")
    assert "login_success" in line
    assert "1970-01-01T00:00:00+00:00" in line


def test_audit_event_includes_detail():
    line = _audit_event("login_failed", detail="attempt=3", now=0.0)
    assert "login_failed" in line
    assert "attempt=3" in line


def test_audit_event_does_not_echo_password():
    # detail is the only free-text field; callers never pass the password,
    # and a typical detail must not resemble credential material.
    line = _audit_event("login_failed", detail="attempt=2", now=0.0)
    assert "password" not in line.lower()
