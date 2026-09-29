# Setup

Alpha is a Vite + React front end over a Convex backend, with a framework-free
AI stack in `src/alpha`. This page covers running it locally, how the pieces fit
together, and what to do when something looks wrong.

Alpha is self-contained: **no external AI provider, no hosted vector database,
no identity provider.** The only service involved is the Convex deployment that
holds Alpha's own data.

---

## Requirements

- **bun** (package manager and script runner)
- **Node 20+** (for the Convex CLI and Vitest)
- A Convex deployment — your own, or a dev deployment from `bunx convex dev`

No GPU, no model download, and no AI provider key: there is nothing to
authenticate against.

---

## Install and verify

```bash
bun install
bun run test          # vitest: gradients, training, retrieval, security, auth primitives
bun run typecheck     # tsc -b --noEmit
```

The test suite is the fastest way to confirm the AI stack works: it trains a
model on the bundled corpus and asserts the validation loss falls below the
uniform baseline. It also exercises Alpha's password derivation, session token
hashing and input validation directly.

---

## Run

```bash
bunx convex dev       # Convex backend + generated types (leave running)
bun run dev           # Vite dev server on http://localhost:5173
```

Open `/` for the landing page, `/auth` to create an account or sign in, and
`/dashboard` for the Alpha workspace.

If your environment runs the dev server and the Convex process for you, do not
start or kill them yourself — edit files and let the running processes pick the
changes up.

---

## Environment variables

Alpha needs three variables, and nothing else. See `.env.example`.

| Variable | Used by | Notes |
| --- | --- | --- |
| `VITE_CONVEX_URL` | the Convex React client in the browser | public by design; it is a deployment URL, not a secret |
| `CONVEX_DEPLOYMENT` | the Convex CLI (`bunx convex dev`) | identifies which deployment to push to |
| `CONVEX_URL` | any server-side or CLI client of Alpha's API | optional; only needed outside the browser |

There are no provider keys, because Alpha has no provider. Nothing in the
repository reads an OpenAI, Gemini, Claude or other AI service variable, and
there is no fallback path that would use one.

If your `.env.local` still defines `CONVEX_SITE_URL`, `VLY_APP_NAME`,
`VLY_CONVEX_AUTH_ISSUER` or `VLY_INTEGRATION_KEY`, those belonged to the removed
identity/integration stack. Alpha reads none of them; delete them from the
environment UI at your convenience. Alpha's own environment values are managed
outside the repository, so `.env.example` is documentation rather than a
required file.

**Never commit secrets.** Alpha keeps only password *derivations* and session
token *hashes*; `redactSecrets()` in `src/alpha/security` strips anything
key-shaped before it reaches a log or an audit record.

---

## Persistence

Alpha stores everything in Convex tables, defined in `src/convex/schema.ts`, and
every row is scoped to the account that owns it by `requireActorId`.

| Group | Tables |
| --- | --- |
| Identity | `alphaUsers`, `alphaSessions`, `alphaSecurityEvents` |
| Conversations | `alphaConversations`, `alphaMessages` |
| Model management | `alphaModels`, `alphaTokenizers`, `alphaDatasets`, `alphaCheckpoints` |
| Retrieval and memory | `alphaVectors`, `alphaMemories` |
| Observability | `alphaRuns`, `alphaSpans`, `alphaAuditLogs` |
| Tools and automation | `alphaTools`, `alphaWorkflows`, `alphaJobs` |

After changing `src/convex/schema.ts` or any Convex function, regenerate types:

```bash
bunx convex dev --once && bunx tsc -b --noEmit
```

Never hand-edit `src/convex/_generated/**`; it is generated and git-ignored.

### Schema changes and existing data

Changing a table's columns does not migrate existing rows. Two consequences are
worth stating plainly:

- Rows written before a field existed will not have it. Queries here tolerate
  missing optional fields; add a migration before making a field required.
- Removing a table from the schema deletes that table's data on the next push.
  That is how the previous auth tables were removed when Alpha took over
  identity — any Alpha records written under the old account ids are still
  present but are no longer reachable by a signed-in account.

---

## A first session in the workspace

1. **Create an account** at `/auth`. Passwords are derived with PBKDF2, so
   sign-up takes a moment by design.
2. **Overview** — see the model's real stage (`UNTRAINED`), parameter count,
   architecture table and the status of every module, including `Alpha
   Authentication` and `Alpha API`.
3. **Training** — press *Train Alpha*. Watch the loss curve against the uniform
   baseline; a checkpoint is written to Convex and the stage badge changes.
4. **Inference** — generate from the model. Compare it with the untrained
   output, and read the context-window report showing what was actually sent.
5. **Knowledge** — ingest a document, ask a question, read the cited chunks,
   write a memory, approve it, delete it.
6. **Work** — run an agent task, read its plan and tool calls, register and run
   the sample workflow.
7. **Governance** — policy decisions, audit-chain integrity, latency histograms.
8. **Account** — the account record, every live session, end a session, sign out
   everywhere, change the password.

---

## Repository conventions

- **Authentication.** Use `useAuth()` from `@/hooks/use-auth`. Identity is
  Alpha's own (`src/convex/alphaAuth`); do not add a provider. Route the token
  only through the provider — components should read `user` / `sessions`, not
  reach for storage.
- **Protected routes.** Wrap them in `RequireAuth` from
  `@/components/RequireAuth`; it preserves the requested path in
  `/auth?returnTo=…` and explains the block on the page the visitor asked for.
- **API calls.** Browser code calls `api.alpha.*` and `api.alphaAuth.*`. Every
  Alpha function takes a `sessionToken` and resolves the account server-side; if
  you add one, do the same and go through `requireActorId`.
- **The AI stack.** `src/alpha` must stay free of React, Vite and Convex
  imports. It is the runtime other clients will embed, so its only dependencies
  are its own modules.
- **Styling.** Tailwind v4 with the tokens in `src/index.css`. Keep the
  `@import "tailwindcss"` line, the `@theme inline` block and the CSS variables;
  Alpha's Studio theme extends them.
- **Rendering.** Routes are lazy-loaded in `src/main.tsx`. The provider tree is
  `RootErrorBoundary → AlphaAuthProvider → BrowserRouter → Suspense`.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Blank preview | a compile error is blocking the app | `bun run typecheck`, then `bunx convex dev --once` |
| `api.alpha.*` or `api.alphaAuth.*` missing | Convex codegen is stale | `bunx convex dev --once` |
| Signed out immediately after signing in | the deployment has no `alphaUsers`/`alphaSessions` tables yet | `bunx convex dev --once` to push the schema |
| Sign-in says "incorrect" for a password you know | the account is locked after repeated failures | wait 15 minutes, or clear `lockedUntil` on the account row |
| Workspace stuck on "training the tokenizer" | an exception during initialisation | the panel prints the error; usually a config conflict (vocabulary > model vocab, or sequence length > context) |
| Training seems slow | every step is a real forward and backward pass in JavaScript, and there is no KV cache in inference | lower `totalSteps`, `batchSize` or `seqLen`, or use the `nano` preset |
| Generation is nonsense | the model is untrained | train it; the badge and warning change when a checkpoint exists |
| MCP shows `NOT CONFIGURED` | no endpoint has been supplied | construct an `HttpMcpClient` with a real endpoint and call `registerMcpTools` |
