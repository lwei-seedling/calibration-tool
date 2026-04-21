# Auth & Login — Manual Test Plan

Walk through this checklist before promoting a new deployment (local, shared
staging, or Streamlit Cloud) to end users. Everything here is black-box —
no source changes, only the Streamlit UI and `secrets.toml`.

**Estimated time:** 15 – 20 minutes on a clean environment.

---

## Pre-requisites

| Item | How to satisfy |
|---|---|
| Repo checked out, UI extras installed | `pip install -e ".[ui]"` |
| Streamlit on PATH | `streamlit --version` returns a version |
| A known plaintext password for the test | Choose a throwaway value, e.g. `Test-Login-2026` |
| A fresh browser profile | Use an incognito window or a dedicated Chrome/Firefox profile |

> **Tip.** Close any running Streamlit server between scenarios so
> `session_state` starts empty: `Ctrl+C` in the launching terminal.

---

## 0. Generate a PBKDF2 hash

**Goal:** confirm the CLI emits the expected hash format and does not leak
plaintext into shell history.

| Step | Expected result |
|---|---|
| Run `python auth.py` (no argument) | Prompts `Password:` with no echo as you type |
| Type a password and press Enter | Prints a single line starting with `pbkdf2_sha256$390000$` |
| `history | tail -5` (bash) | The password is **not** visible in history |
| Run `python auth.py Test-Login-2026` | Prints a PBKDF2 hash (different salt → different output on every run) |
| Run the same command twice | The two hashes **differ** (salt is random); both verify the same plaintext |

Save one hash from the interactive run into `HASH=` in your shell for the
next sections.

---

## 1. Open-access (dev) mode

**Goal:** confirm the app is fully unauthenticated when `[auth]` is absent.

| Step | Expected result |
|---|---|
| Ensure `.streamlit/secrets.toml` does **not** contain an `[auth]` section (delete or rename the file) | — |
| `streamlit run app.py` | App loads directly to the **Setup** page; no login form is shown |
| Sidebar shows the *Logout* button | Clicking *Logout* has no effect (idempotent; UI stays on Setup) |

Record result: ☐ Pass ☐ Fail

---

## 2. Fail-closed mode (misconfigured hash)

**Goal:** confirm an empty `password_hash` cannot be used as a backdoor.

Create `.streamlit/secrets.toml`:

```toml
[auth]
password_hash = ""
```

| Step | Expected result |
|---|---|
| Restart `streamlit run app.py` | Page shows error: *"Auth is enabled but no password is configured. Contact admin."* |
| No login form is rendered | — |
| The rest of the app is unreachable (`st.stop()` takes effect) | — |

Record result: ☐ Pass ☐ Fail

---

## 3. Password gate — happy path

**Goal:** confirm a valid password lets the user in.

Update `.streamlit/secrets.toml`:

```toml
[auth]
password_hash = "pbkdf2_sha256$390000$...$..."   # the $HASH you saved in step 0
```

| Step | Expected result |
|---|---|
| Restart `streamlit run app.py` | Login form appears with title *🔒 Login Required* |
| Submit the correct password | Form disappears; main Setup page renders; sidebar shows *Logout* |
| Reload the browser tab | Still authenticated (`_authenticated` persists in session state) |

Record result: ☐ Pass ☐ Fail

---

## 4. Password gate — invalid password

| Step | Expected result |
|---|---|
| Submit a wrong password once | Red error: *"Incorrect password. Please try again."* |
| Form remains usable | Password field is cleared; counter is now 1 |
| Submit 3 more wrong passwords (total 4 failures) | Same error each time; **not** yet locked |

Record result: ☐ Pass ☐ Fail

---

## 5. Lockout on the 5th consecutive failure

| Step | Expected result |
|---|---|
| Submit a 5th wrong password | Page flips to *"⛔ Temporarily Locked"* |
| Error reads *"Too many failed attempts. Try again in 60s."* (or similar, value varies ±1 s) | — |
| Attempting to interact with the password form | No form is rendered while locked |
| Wait 10 – 20 seconds, reload the tab | Still locked; countdown reflects remaining seconds |

Record result: ☐ Pass ☐ Fail

---

## 6. Lockout expires automatically (no full refresh required)

| Step | Expected result |
|---|---|
| Wait for the countdown to hit `0s` | — |
| Interact with the page (e.g. click anywhere that triggers a rerun) — **do not** manually reload | Login form re-renders; attempt counter is cleared |
| Submit the correct password | Logs in normally |

> If the form does **not** re-render after the countdown, that's a bug —
> file it as a regression against `auth.py::check_auth`.

Record result: ☐ Pass ☐ Fail

---

## 7. Logout clears all state

| Step | Expected result |
|---|---|
| From the authenticated app, click *Logout* in the sidebar | App returns to login form |
| Submit 3 wrong passwords | Attempt counter reads 3 (or the error text matches a fresh gate) |
| Click *Logout* equivalent path — refresh and log in with the correct password | Lets you in; counter is **reset**, not carried over |

Record result: ☐ Pass ☐ Fail

---

## 8. Legacy SHA-256 hash still verifies

**Goal:** confirm existing deployments won't break.

Generate a legacy digest:

```bash
python -c "import hashlib; print(hashlib.sha256(b'Test-Login-2026').hexdigest())"
```

Replace `password_hash` in `secrets.toml` with that 64-char hex string (no
prefix). Restart the server.

| Step | Expected result |
|---|---|
| Submit the correct password | Logs in |
| Submit a wrong password | Normal error path |

Record result: ☐ Pass ☐ Fail

---

## 9. Copy-paste robustness (whitespace in hash)

**Goal:** confirm operators who paste hashes with trailing whitespace or
newlines aren't locked out by their own typo.

Edit `secrets.toml` so the `password_hash` value has a trailing space
and/or newline before the closing quote:

```toml
[auth]
password_hash = "pbkdf2_sha256$390000$...$...   "
```

| Step | Expected result |
|---|---|
| Restart the server | No startup error |
| Submit the correct password | Logs in normally |

Record result: ☐ Pass ☐ Fail

---

## 10. Session-scope bypass — acknowledged limitation

**Goal:** document (not fix) that the 60-second lockout is session-local.

| Step | Expected result |
|---|---|
| Lock out session A (Browser profile 1) by submitting 5 wrong passwords | Session A shows "Temporarily Locked" |
| In Browser profile 2 (or a private/incognito window), open the same URL | A fresh login form is shown; **no lockout is enforced** |
| Submit any password (right or wrong) in profile 2 | Behaves as a fresh session |

> This is **expected behaviour** and is documented in `README.md` and
> `CLAUDE.md`. The lockout is a UX rate limiter, not a hard security
> control. If you need IP-based rate limiting, put Cloudflare Turnstile /
> Streamlit Cloud SSO / an authenticating reverse proxy in front of the
> app.

Record result: ☐ Pass (behaviour matches docs) ☐ Fail (unexpected
divergence)

---

## 11. Deployment checklist (Streamlit Cloud)

Before sharing the deployed URL externally:

- [ ] `[auth]` block is configured in the Streamlit Cloud *Secrets*
      dashboard (**not** committed to `git`)
- [ ] Plaintext password is shared via a password manager / 1Password link,
      not email or chat
- [ ] `secrets.toml` on any self-hosted box is `chmod 600` and owned by the
      Streamlit service user
- [ ] Legacy SHA-256 hashes have been rotated to PBKDF2 (optional but
      recommended)
- [ ] `pytest tests/test_auth.py -v` is green on the deployed branch

---

## Regression triage

If any scenario fails, capture:

1. The `secrets.toml` snippet (redact the hash).
2. Exact steps to reproduce.
3. The browser console / Streamlit server log output around the failure.
4. `git rev-parse HEAD` on the deployed branch.

File as a GitHub issue tagged `auth` and reference this document.
