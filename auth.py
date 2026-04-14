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
    python auth.py YourPassword
"""
from __future__ import annotations

import hashlib
import hmac

_MAX_ATTEMPTS = 5


def _hash_password(password: str) -> str:
    """SHA-256 hex digest of *password*."""
    return hashlib.sha256(password.encode()).hexdigest()


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

    # ------------------------------------------------------------------
    # Dev / open-access mode: no [auth] section in secrets
    # ------------------------------------------------------------------
    try:
        auth_cfg = st.secrets["auth"]
    except (KeyError, FileNotFoundError):
        return True

    # ------------------------------------------------------------------
    # Fail-closed: [auth] exists but hash is missing / empty
    # ------------------------------------------------------------------
    expected_hash = auth_cfg.get("password_hash", "")
    if not expected_hash:
        st.error("Auth is enabled but no password is configured. Contact admin.")
        return False

    # ------------------------------------------------------------------
    # Already authenticated this session
    # ------------------------------------------------------------------
    if st.session_state.get("_authenticated"):
        return True

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    attempts = st.session_state.get("_login_attempts", 0)
    if attempts >= _MAX_ATTEMPTS:
        st.title("\u26d4 Access Blocked")
        st.error("Too many failed attempts. Please refresh the page to try again.")
        return False

    # ------------------------------------------------------------------
    # Login form
    # ------------------------------------------------------------------
    st.title("\U0001f512 Login Required")
    st.caption("Internal demo access")

    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")

    if submitted:
        if hmac.compare_digest(_hash_password(password), expected_hash):
            st.session_state["_authenticated"] = True
            st.session_state["_login_attempts"] = 0
            st.rerun()
        else:
            st.session_state["_login_attempts"] = attempts + 1
            st.error("Incorrect password. Please try again.")

    return False


def logout() -> None:
    """Clear authentication state."""
    import streamlit as st

    st.session_state.pop("_authenticated", None)
    st.session_state.pop("_login_attempts", None)


# ------------------------------------------------------------------
# CLI helper: python auth.py <password>
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    pwd = sys.argv[1] if len(sys.argv) > 1 else input("Password: ")
    print(_hash_password(pwd))
