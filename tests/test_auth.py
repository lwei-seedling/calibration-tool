"""Regression tests for auth.py password verification."""
from __future__ import annotations

import hashlib

from auth import _hash_password_pbkdf2, _verify_password


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
