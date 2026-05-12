# Security

This document describes how terminaleyes handles secrets — passwords
that the controller types into the target machine (login passwords,
website logins, sudo, etc.) — and the threat model those protections
assume.

> **Client platform support.** The convenience wrapper
> `scripts/te-secrets` and its "encrypted at rest, no plaintext on
> disk" guarantees only work on **macOS** today. They rely on the
> macOS Keychain (`security` CLI) to store the vault's master
> passphrase. On Linux clients you'd need to bring your own
> equivalent (libsecret / `keyring`); see
> [Linux & other clients](#linux--other-clients) below. Everything
> else (vault file format, redaction on the wire, Pi-side logging)
> is platform-independent.

---

## TL;DR

```bash
# one-time
scripts/te-secrets init                   # master passphrase → macOS Keychain
scripts/te-secrets add desktop            # encrypts value into vault.enc

# every use
scripts/te-secrets run desktop "unlock the screen"
```

- **Master passphrase** lives in macOS Keychain (AES-encrypted at
  rest by the OS, unlocked at login).
- **Entry values** live in `~/.config/terminaleyes/vault.enc`
  (AES-256-GCM, scrypt KDF N=2¹⁵, mode `0600`).
- **At runtime** the wrapper pulls the master from Keychain and
  injects it as `TERMINALEYES_VAULT_PASSPHRASE` **only into the
  spawned child process**. Never written to disk, never in your
  shell env, never on the network.

You never type the password value where I (or any other party
reading the terminal) can see it. The only string that appears in
your shell, your history, and your conversation with me is the
**entry name** (`desktop`), which is meaningless without the vault
file *and* the Keychain master.

---

## Layered storage

There are two distinct secrets at different layers.

### Layer 1 — the master passphrase

A single passphrase that unlocks the vault file. Stored encrypted at
rest in the macOS Keychain under the service `terminaleyes-vault`,
account = `$USER`. The Keychain itself is unlocked when you log into
macOS and is encrypted with a key derived from your macOS account
password (Apple's documentation, `kSecAttrAccessibleWhenUnlocked`).

`scripts/te-secrets init` (one-time) sets/replaces this entry via
`security add-generic-password -U`, which prompts via the OS dialog
— terminaleyes never sees the value.

`scripts/te-secrets exec / run / add / get / ls / rm` reads it via
`security find-generic-password -w` and exports it into the
spawned child process as `TERMINALEYES_VAULT_PASSPHRASE`. That
variable's lifetime is the lifetime of that child — it's never
serialised to a file, never put into your shell's environment,
never sent over a network socket.

### Layer 2 — the encrypted vault file

`~/.config/terminaleyes/vault.enc` is the actual name/value store.
File format and crypto (see `src/terminaleyes/agents/vault.py`):

| Property         | Value                                                                 |
|------------------|-----------------------------------------------------------------------|
| Cipher           | AES-256-GCM (`cryptography.hazmat`)                                   |
| KDF              | scrypt with `N=2**15`, `r=8`, `p=1`, key length = 32 bytes            |
| Salt             | 16 random bytes per-file, written into the header                     |
| Nonce            | 12 random bytes per encryption, written into the header               |
| File permissions | `0600` (owner read/write only — enforced on every write via `os.chmod`)|
| Write style      | Atomic (`tempfile.NamedTemporaryFile` + `os.replace`) so a crash mid-write can't truncate the file |
| File magic       | 4-byte header so future format bumps don't silently corrupt old files |
| Authentication   | AEAD — any byte flipped in the ciphertext, salt, nonce, or header makes decryption fail loudly rather than yielding garbage |

The plaintext payload is a JSON object `{ "name": "value", ... }`.
You can store anything you want under each key — a login password,
an API token, an SSH passphrase. The agent layer's `LoginAgent`
fetches one entry, types it via `keyboard.send_text(value,
secret=True)`, and the keyboard backend redacts the value from
local logs (Pi-side already only logs `length=N`).

---

## At-runtime data flow

```
$ scripts/te-secrets run desktop "unlock the screen"

  ┌─ te-secrets (bash) ─────────────────────────────────────────────┐
  │  security find-generic-password -s terminaleyes-vault -w        │
  │       │                                                         │
  │       ▼                                                         │
  │  TERMINALEYES_VAULT_PASSPHRASE="<master>"  (child env only)     │
  │  exec terminaleyes do --vault desktop "unlock the screen"       │
  └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─ terminaleyes do (python) ──────────────────────────────────────┐
  │  controller.plan_intent(...) → [login {vault_name: desktop}]    │
  │  LoginAgent.run()                                               │
  │    Vault(master_from_env).get("desktop") → <password> in memory │
  │    keyboard.send_text(<password>, secret=True)                  │
  │  ── dev-side keyboard log records: length=14 (no value)         │
  └─────────────────────────────────────────────────────────────────┘
                              │
                  HTTPS not used; HTTP over USB-ECM
                              ▼
  ┌─ Raspberry Pi (terminaleyes-pi) ────────────────────────────────┐
  │  POST /bt/text  {"text": "<password>"}                          │
  │  ── server log records: length=14 (no value)                    │
  │  bt_hid writes HID reports onto the L2CAP interrupt channel     │
  └─────────────────────────────────────────────────────────────────┘
                              │
                            Bluetooth HID
                              ▼
                        Target machine
                        types the password
                        into the focused
                        password field
```

At no step is the cleartext password written to a log, persisted to
a file, or exposed on a CLI argv (`ps -e ww` would only show
`terminaleyes do --vault desktop "unlock the screen"` — the value
itself never appears).

---

## Threat model

What this design protects against:

- **A casual reader of your terminal / chat / screen-share.** Names
  visible, values not.
- **A reader of your shell history and your `.env` files.** No
  plaintext password lives in either.
- **A reader of `ps -e ww` while terminaleyes is running.** Argv
  contains the entry name only.
- **A reader of `terminaleyes-pi` logs on the Raspberry Pi.** The Pi
  side records only `length=N` for `/bt/text`.
- **A reader of `~/.config/terminaleyes/vault.enc` on a stolen
  laptop where macOS Keychain is locked** (laptop powered off /
  account logged out). The vault file is useless without the master,
  and the master is encrypted at rest by Apple's Keychain using a
  key derived from your macOS account password.
- **An attacker who corrupts the vault file.** AES-GCM is AEAD —
  any tampering fails decryption rather than producing garbage that
  would be typed at the target.

What this design does **NOT** protect against:

- **A logged-in attacker with root or your user ID on the dev Mac.**
  Once `TERMINALEYES_VAULT_PASSPHRASE` is exported into a child
  process, anyone who can read `/proc/<pid>/environ` (Linux) or
  attach `lldb` to that process (macOS) can recover it for the
  process's lifetime. Python doesn't zero secret memory, so even
  after the agent releases the reference a memory dump can recover
  it briefly. Treat the dev Mac as **inside the trust boundary**.
- **A logged-in attacker who can keylog your macOS session.**
  Anyone who can run as you can call `security find-generic-password`
  too. Filesystem encryption (FileVault) and the macOS login
  password are the actual boundary here, not the vault.
- **Eavesdropping on the USB-ECM link between dev Mac and Pi.** The
  HTTP between your Mac (10.0.0.1) and the Pi (10.0.0.2) is
  plaintext. The link is point-to-point over USB — physical access
  to the cable is required — but if you don't trust the cable, treat
  it like any unencrypted leg.
- **Keystrokes on the Bluetooth-HID link from Pi to target.** BT
  HID is encrypted by the BlueZ stack after pairing, but a
  pre-paired malicious receiver in proximity could still capture
  the typed bytes. Pair only the intended target machine.
- **Memory dumps of `terminaleyes-pi`.** The Pi briefly holds the
  bytes it's about to send over HID. Same trust assumption as the
  dev Mac.
- **Casual shoulder-surfers watching the target screen.** Many
  login fields mask, but not all. The controller doesn't choose
  *what* to type into.

---

## Command Center (cc) daemon

When you run `terminaleyes cc` (the FastAPI web UI), the daemon
process holds the vault state in memory for the lifetime of the
daemon — every `POST /api/run` from the browser reuses the same
process. Implication: **the daemon needs the master in its env at
startup**, and that master then lives in the daemon's environ for
its lifetime.

The supported launch pattern is therefore:

```bash
scripts/te-secrets exec terminaleyes cc --port 8765
```

…which is just `terminaleyes cc` with the Keychain master injected
the same way as any other `te-secrets` invocation. Starting cc
directly without the wrapper means it has no master and any intent
that reaches `LoginAgent` will fail with "Vault decryption failed".

If you don't need the web UI for that session, run intents through
the bare CLI (`scripts/te-secrets run …`) and skip cc entirely.

---

## Operational hygiene

- **Don't commit `vault.enc` to git.** Even though it's encrypted,
  rotating the master means changing your macOS Keychain entry —
  and if you've ever published the file alongside a weak master,
  brute-force becomes the threat. `.config/terminaleyes/` is outside
  the repo by default; keep it that way.
- **Don't commit `.env` files containing `TERMINALEYES_VAULT_PASSPHRASE`.**
  The wrapper avoids needing one. If you've added one for any
  reason, `chmod 600` it and never push it.
- **Rotate the master if you suspect leakage.** Pick a new master
  with `scripts/te-secrets init`, then `add` each entry again. The
  old `vault.enc` becomes unusable.
- **Delete stale entries.** `scripts/te-secrets rm <name>` writes a
  new vault file omitting the entry. The old plaintext is gone from
  the file; memory remnants are out of scope.

---

## Linux & other clients

`scripts/te-secrets` refuses to run on non-Darwin so you don't end up
with a half-implemented Keychain. The Vault file format itself is
portable — the Python `Vault` class works on any platform. To use
the vault from a Linux client today you have two options:

1. **Prompt the master interactively each invocation.** Don't set
   `TERMINALEYES_VAULT_PASSPHRASE`; the Vault falls back to
   `getpass.getpass`. Each command pauses for a single hidden prompt.
2. **Bring your own secret store.** Stash the master in libsecret /
   GNOME Keyring / KWallet, and export `TERMINALEYES_VAULT_PASSPHRASE`
   from there yourself before invoking `terminaleyes`. The vault
   doesn't care where the env var comes from.

Native Linux support (a libsecret backend equivalent to the macOS
Keychain path) is on the roadmap in `CLAUDE.md` under "Pending" as
the `keyring`-based vault backend.
