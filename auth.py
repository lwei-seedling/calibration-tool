"""Authentication gate for the Streamlit app.

Usage in app.py:
    from auth import check_auth, logout

    def main():
        st.set_page_config(...)
        if not check_auth():
            st.stop()
            return
        # ... rest of app ...

Generate a password hash for secrets.toml:
    python auth.py                # interactive, no shell-history leak
    python auth.py YourPassword   # non-interactive (automation)

Supported hash formats in ``secrets.toml`` under ``[auth].password_hash``:

* **PBKDF2-SHA256** (recommended) -- ``pbkdf2_sha256$<iters>$<salt_hex>$<dk_hex>``.
  Salted, slow (390,000 iterations), resistant to offline brute-force.
* **Legacy SHA-256** -- 64-char hex digest. Still verifies so existing
  deployments keep working, but rotate to PBKDF2 at your next opportunity.
  A legacy hash triggers a one-time rotation warning after login.

Audit logging: login successes, failures, lockouts, and legacy-hash usage
are written to stderr as ``[auth] <utc-timestamp> <event> [detail]`` lines
(captured by Streamlit Cloud / container logs). Passwords are never logged.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import sys
import time
from datetime import datetime, timezone

_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 60

_PBKDF2_PREFIX = "pbkdf2_sha256"
_PBKDF2_ITERATIONS = 390_000
_PBKDF2_SALT_BYTES = 16
_PBKDF2_MAX_ITERATIONS = 10_000_000   # defense-in-depth cap against DoS hashes


def _hash_password_pbkdf2(password: str, *, salt: bytes | None = None,
                          iterations: int = _PBKDF2_ITERATIONS) -> str:
    """Return ``pbkdf2_sha256$iters$salt_hex$dk_hex`` for *password*."""
    if salt is None:
        salt = os.urandom(_PBKDF2_SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"{_PBKDF2_PREFIX}${iterations}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, expected_hash: str) -> bool:
    """Constant-time verify *password* against *expected_hash*.

    Accepts both the PBKDF2 format produced by ``_hash_password_pbkdf2`` and
    legacy single-round SHA-256 hex digests. Empty passwords are rejected
    unconditionally. ``expected_hash`` is stripped so copy-pasted TOML values
    with trailing whitespace/newlines still verify. Legacy SHA-256 digests are
    compared case-insensitively.
    """
    if not password or not expected_hash:
        return False
    expected_hash = expected_hash.strip()
    if not expected_hash:
        return False

    if expected_hash.startswith(_PBKDF2_PREFIX + "$"):
        parts = expected_hash.split("$")
        if len(parts) != 4:
            return False
        _, iter_s, salt_hex, dk_hex = parts
        try:
            iterations = int(iter_s)
            salt = bytes.fromhex(salt_hex)
            expected_dk = bytes.fromhex(dk_hex)
        except ValueError:
            return False
        if not (0 < iterations <= _PBKDF2_MAX_ITERATIONS):
            return False
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
        return hmac.compare_digest(candidate, expected_dk)

    # Legacy SHA-256 hex digest (backward compatibility).
    candidate = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(candidate, expected_hash.lower())


def _is_legacy_hash(expected_hash: str) -> bool:
    """True if *expected_hash* is a legacy bare SHA-256 hex digest (not PBKDF2).

    Used to nudge operators to rotate to the salted PBKDF2 format. A legacy
    hash is a 64-character hex string with no ``pbkdf2_sha256$`` prefix.
    """
    h = expected_hash.strip()
    if not h or h.startswith(_PBKDF2_PREFIX + "$"):
        return False
    if len(h) != 64:
        return False
    try:
        bytes.fromhex(h)
    except ValueError:
        return False
    return True


def _audit_event(event: str, *, detail: str = "", now: float | None = None) -> str:
    """Build a single-line audit record with a UTC timestamp.

    Never pass secrets (passwords, hashes) in *detail* -- audit records are
    written to stderr and captured by deployment logs. *detail* should carry
    only non-sensitive metadata such as attempt counts.
    """
    ts = datetime.fromtimestamp(
        now if now is not None else time.time(), tz=timezone.utc
    ).isoformat(timespec="seconds")
    line = f"[auth] {ts} {event}"
    if detail:
        line += f" {detail}"
    return line


def _audit_log(event: str, *, detail: str = "") -> None:
    """Emit an audit record to stderr (captured by Streamlit Cloud logs)."""
    print(_audit_event(event, detail=detail), file=sys.stderr, flush=True)


def _reset_auth_state() -> None:
    import streamlit as st

    for key in ("_authenticated", "_login_attempts", "_login_locked_until"):
        st.session_state.pop(key, None)


def check_auth() -> bool:
    """Show a login form and return ``True`` only when authenticated.

    Must be called **after** ``st.set_page_config()``.

    Behaviour by secrets configuration:

    * No ``[auth]`` section in secrets -> open access (dev mode).
    * ``[auth]`` exists but ``password_hash`` is missing/empty -> fail-closed
      (blocks access with an admin error).
    * ``[auth]`` with valid ``password_hash`` -> password gate.
    """
    import streamlit as st

    # Any failure to read a well-formed [auth] section -> open-access (dev) mode.
    # Streamlit raises StreamlitSecretNotFoundError (KeyError subclass) when
    # secrets.toml is absent; a malformed section can surface as AttributeError.
    try:
        auth_cfg = st.secrets["auth"]
    except (KeyError, FileNotFoundError, AttributeError):
        return True

    expected_hash = auth_cfg.get("password_hash", "")
    if not expected_hash:
        st.error("Auth is enabled but no password is configured. Contact admin.")
        return False

    if st.session_state.get("_authenticated"):
        if st.session_state.pop("_legacy_hash_warning", False):
            st.warning(
                "This deployment uses a legacy SHA-256 password hash. "
                "Rotate to PBKDF2 with `python auth.py` and update secrets.toml."
            )
        return True

    now = time.time()
    locked_until = st.session_state.get("_login_locked_until", 0.0)
    if locked_until and now < locked_until:
        remaining = int(locked_until - now) + 1
        st.title("⛔ Temporarily Locked")
        st.error(f"Too many failed attempts. Try again in {remaining}s.")
        return False
    if locked_until and now >= locked_until:
        st.session_state["_login_locked_until"] = 0.0
        st.session_state["_login_attempts"] = 0

    attempts = st.session_state.get("_login_attempts", 0)

    st.title("\U0001f512 Login Required")
    st.caption("Internal demo access")

    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")

    if submitted:
        if _verify_password(password, expected_hash):
            _audit_log("login_success")
            _reset_auth_state()
            st.session_state["_authenticated"] = True
            if _is_legacy_hash(expected_hash):
                _audit_log("legacy_hash_in_use", detail="rotate to pbkdf2_sha256")
                # Surfaced on the post-login rerun (see top of check_auth);
                # an st.warning here would be wiped by st.rerun().
                st.session_state["_legacy_hash_warning"] = True
            st.rerun()
        else:
            new_attempts = attempts + 1
            st.session_state["_login_attempts"] = new_attempts
            if new_attempts >= _MAX_ATTEMPTS:
                st.session_state["_login_locked_until"] = time.time() + _LOCKOUT_SECONDS
                _audit_log("lockout", detail=f"seconds={_LOCKOUT_SECONDS}")
                st.error(
                    f"Too many failed attempts. Locked for {_LOCKOUT_SECONDS}s."
                )
            else:
                _audit_log("login_failed", detail=f"attempt={new_attempts}")
                st.error("Incorrect password. Please try again.")

    return False


def logout() -> None:
    """Clear authentication state."""
    _reset_auth_state()


if __name__ == "__main__":
    import getpass
    import sys

    pwd = sys.argv[1] if len(sys.argv) > 1 else getpass.getpass("Password: ")
    print(_hash_password_pbkdf2(pwd))
