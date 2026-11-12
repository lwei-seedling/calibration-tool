# Deployment and authentication

Optional password gate (`auth.py`) and Streamlit Community Cloud deployment. Not needed for local use.

[← Back to README](../README.md)

---

## Password gate

The Streamlit app ships with an optional password gate (`auth.py`). It has
three modes depending on how `.streamlit/secrets.toml` is configured:

| `secrets.toml` | Mode | Use case |
|---|---|---|
| No `[auth]` section | **Open access** (dev) | Local development, private networks |
| `[auth]` with empty hash | **Fail-closed** | Misconfiguration guard |
| `[auth]` with valid hash | **Password gate** | Private demos, Streamlit Cloud |

### Generate a password hash

```bash
python auth.py                 # interactive — uses getpass.getpass (no shell history)
python auth.py YourPassword    # non-interactive (automation/CI)
```

Output is a salted PBKDF2-SHA256 string:

```
pbkdf2_sha256$390000$3c80...e2f1$04aa...858c
```

### Configure `secrets.toml`

Create `.streamlit/secrets.toml` in the repo root (or paste into the
Streamlit Cloud → Settings → Secrets dashboard):

```toml
[auth]
password_hash = "pbkdf2_sha256$390000$3c80...e2f1$04aa...858c"
```

Restart the app; the login form appears at startup. File permissions on a
self-hosted deployment should be `chmod 600 .streamlit/secrets.toml`.

### Hash format

| Format | Purpose | Notes |
|---|---|---|
| `pbkdf2_sha256$<iters>$<salt_hex>$<dk_hex>` | **Recommended.** Salted PBKDF2-SHA256 (390,000 iters). | Produced by `python auth.py`. |
| 64-char hex digest | **Legacy** unsalted SHA-256. | Still verifies; rotate to PBKDF2. |

### Rate limiting

After 5 failed login attempts the session is locked for 60 seconds with a
visible countdown; the form becomes usable again automatically (no full
page refresh required). On successful login or explicit logout, the
attempt counter and lockout timestamp are cleared.

> **Known limitation — session-scoped lockout.** The 5-attempt/60-s window
> is tracked in Streamlit `session_state`, so an attacker can bypass it by
> opening a fresh incognito session. The lockout is a **UX rate limiter,
> not a hard security control**. For real brute-force defense, front the
> app with Cloudflare Turnstile, Streamlit Community Cloud's Google SSO
> (Settings → Sharing → *Only specific people*), or an IP-aware reverse
> proxy.

### Running the auth tests

```bash
pytest tests/test_auth.py -v
```

### Manual test plan

Before promoting a new deployment to users, walk through the checklist in
[`docs/AUTH_MANUAL_TEST_PLAN.md`](docs/AUTH_MANUAL_TEST_PLAN.md). It covers
open-access mode, fail-closed mode, successful/failed logins, lockout and
expiry, logout cleanup, PBKDF2↔legacy verification, and the session-scope
bypass acknowledgement.

---

---

## Streamlit Cloud Deployment

To deploy for non-technical users (URL-only, no Python install required):

**1. Ensure `requirements.txt` is present** (already in the repo):
```
pip install -r requirements.txt   # verified list of all UI dependencies
```

**2. Push to GitHub** and connect the repo at [share.streamlit.io](https://share.streamlit.io):
- Repository: `lwei-seedling/calibration-tool`
- Branch: `main`
- Main file path: `app.py`

**3. Configure secrets** in the Streamlit Cloud dashboard (Settings → Secrets):

```toml
[auth]
password_hash = "pbkdf2_sha256$390000$...$..."   # generate via python auth.py
```

Omit the `[auth]` section entirely for an open demo (not recommended for sensitive data).

**4. Share the deployed URL.** Users can upload CSVs, run calibration, explore results,
and download reports — no Python installation needed.

---
