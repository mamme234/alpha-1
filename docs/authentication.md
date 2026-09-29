# Alpha Authentication

Alpha owns its accounts. There is no identity provider behind this, no federated
or delegated sign-in, and no vendor account that Alpha depends on. What exists
is a small, explicit surface: an account table, a session table, and the
functions that operate on them.

```
client                        Alpha backend (Convex)
──────                        ──────────────────────
email + password   ─────────► register / signIn        (action: PBKDF2 + action only)
                   ◄───────── { token, expiresAt, user }
token (localStorage) ────────► every api.alpha.* call  (arg: sessionToken)
                   ◄───────── data scoped to that account
logout             ─────────► sessions.signOut          (mutation: revoke one row)
```

Status: **implemented and verified end to end** against a live deployment
(register, duplicate rejection, sign-in, session resolution, wrong-password
rejection, sign-out, and post-sign-out rejection were each exercised through the
Convex CLI).

---

## The account

`alphaUsers` holds one row per account.

| Field | Why |
| --- | --- |
| `email`, `emailKey` | the address as typed, plus a lowercased unique key used for every lookup |
| `displayName` | what Alpha calls you; trimmed, control characters refused |
| `passwordHash`, `passwordSalt`, `passwordIterations`, `passwordAlgorithm` | the derivation and the parameters that produced it |
| `role` | `user` or `admin`; checked server-side, never taken from the client |
| `status` | `active` or `suspended`; a suspended account cannot sign in |
| `failedSignIns`, `lockedUntil` | the lockout counter and its deadline |
| `sessionEpoch` | bumped to retire every existing session at once |

**Nothing derived from a password is reversible, and the password itself is
never stored, logged or sent anywhere.**

### Password handling

Passwords are derived with **PBKDF2-HMAC-SHA256** (`600,000` iterations,
`16`-byte random salt, `256`-bit output) using the Web Crypto API, which the
Convex runtime provides natively. The salt is generated per account with
`crypto.getRandomValues`, and the iteration count is stored next to the hash, so:

- two accounts with the same password have different hashes;
- the work factor can be raised later without invalidating anyone;
- an account still at an older work factor verifies at its own cost and is
  re-derived transparently on the next successful sign-in (`needsRehash`).

Derivation happens in an **action**, never a query or mutation. Convex requires
those to be deterministic, and a fresh salt is not.

### Password policy

Enforced in `src/convex/alphaAuth/validation.ts`, with a code the client maps to
a message:

- at least 10 characters, at most 200;
- not only whitespace;
- not present in a list of common breach passwords;
- must not contain the email address or its local part.

Rejections carry a stable code (`weak-password`, `invalid-email`, …) in a
`ConvexError`, so the UI shows the exact reason instead of guessing.

---

## Sessions

`alphaSessions` holds one row per signed-in device.

| Field | Why |
| --- | --- |
| `tokenHash` | `sha256(token)`. The token itself is never stored |
| `epoch` | the account's `sessionEpoch` when this session was issued |
| `expiresAt` | absolute end of life (30 days) |
| `lastSeenAt` | drives the idle window (7 days) |
| `revokedAt`, `revokedReason` | a session that ended, and why |
| `userAgent` | shown in the account controls so a device can be recognised |

The token is 32 random bytes, base64url-encoded (43 characters). It is handed to
the client exactly once, at sign-in. Hashing is a pure-JS SHA-256, so a session
can be resolved inside a query while staying deterministic.

A session stops working when any of these is true — all checked in one function,
`resolveSession`:

1. no row matches the token hash;
2. it was revoked;
3. `expiresAt` has passed (absolute expiry);
4. it has been idle longer than 7 days;
5. its `epoch` no longer matches the account's (a password change, or
   "sign out everywhere");
6. the account is suspended.

Revocation is recorded, not implied, so the account controls can list what ended
and why. Because the session list is a live query, ending a session in one tab
removes it from every other tab.

### Where the token lives, and what that costs

The token is kept in browser `localStorage` under `alpha.session.token`.

Alpha's API is called directly as Convex functions rather than over Alpha's own
HTTP endpoints, and functions do not carry cookies, so an `httpOnly` cookie is
not available for this transport. Stated plainly: **any script running on
Alpha's origin can read that token.** The controls that make this survivable are
server-side — hashing at rest, absolute expiry, idle expiry, and revocation that
takes effect immediately.

The alternative, an Alpha HTTP auth layer with cookie sessions and CSRF
protection, is **not built**. It is the natural next step if Alpha is exposed to
untrusted third-party scripts.

---

## Authorization

Two layers, deliberately separate:

- **Tool and agent permissions** live in the Alpha AI stack
  (`src/alpha/security/policy.ts`): roles, scopes, approval gates, and the rule
  that an agent cannot execute a powerful action without permission. That layer
  is about what a *run* may do.
- **API authorization** lives on the backend: `requireActorId(ctx, sessionToken)`
  resolves the account, and `requireRole(ctx, sessionToken, roles)` additionally
  checks the role. Every Alpha record is scoped to that resolved account, so no
  handler accepts an owner id from the client.

Registration always creates a `user`. There is no first-user-is-admin rule, no
special address, and no environment flag that grants privileges — promoting an
account is a deliberate operation on the deployment, not a code path.

---

## Rate limiting and lockouts

- **Failed sign-ins** are counted per account. After 8 failures the account is
  locked for 15 minutes; the lock is stored in the database, so it survives a
  reload and cannot be cleared by retrying quickly.
- **Sign-in is deliberately uniform.** A missing account still performs a PBKDF2
  derivation against a fixed dummy record before failing, and both cases return
  the same message — "Email or password is incorrect." — so the endpoint cannot
  be used to discover who has an account, and does not leak the difference
  through timing.
- **Alpha's own rate limiter** (`src/alpha/security/rate-limit.ts`) governs tool,
  agent, workflow, embedding and inference work inside a run.

---

## Audit

Every authentication decision that matters is written to `alphaSecurityEvents`:
account creation, successful sign-in, failed sign-in, lockout, password change,
and session revocation. Records carry the account, the outcome, a human-readable
detail and a timestamp. Alpha's separate hash-chained audit log
(`alphaAuditLogs`) records what the AI stack did during a run.

---

## Backend API for authentication

| Function | Kind | Purpose |
| --- | --- | --- |
| `alphaAuth.actions.register` | action | create an account and sign it in |
| `alphaAuth.actions.signIn` | action | verify a password and issue a session |
| `alphaAuth.actions.changePassword` | action | replace the password; retires every other session |
| `alphaAuth.sessions.current` | query | the account behind a token, or `null` |
| `alphaAuth.sessions.listMine` | query | active sessions on this account |
| `alphaAuth.sessions.touch` | mutation | extend the idle window, at most once a minute |
| `alphaAuth.sessions.signOut` | mutation | end this session (idempotent) |
| `alphaAuth.sessions.signOutEverywhere` | mutation | end every session on the account |
| `alphaAuth.sessions.revokeSession` | mutation | end one session by id |
| `users.currentUser` | query | the signed-in account's profile |
| `users.updateDisplayName` | mutation | change the display name |

Operator-only, `internalMutation`, unreachable from any client:
`alphaAuth.maintenance.purgeEndedSessions` and
`alphaAuth.maintenance.removeAccount`.

On the client, `useAuth()` from `@/hooks/use-auth` exposes
`{ status, isLoading, isAuthenticated, user, session, sessions, sessionToken,
signIn, signUp, signOut, signOutEverywhere, revokeSession, changePassword,
error, clearError }`.

---

## What is not implemented

Honest list, so nobody has to infer it from the code:

- **No HTTP auth layer.** No cookie sessions, no CSRF tokens, no OAuth/OIDC
  client, no email verification, no password reset by email, no 2FA.
- **No full account deletion.** `maintenance.removeAccount` clears the account,
  its sessions and its security events; it does not walk the model, vector,
  memory or conversation tables.
- **No migration of pre-existing accounts.** Alpha's identity tables are new;
  accounts created under the previous auth stack do not carry over.
