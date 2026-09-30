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

### Prove the model actually trains

The test suite is not the same as seeing the pipeline run. Two commands execute
the real lifecycle end to end in a terminal — corpus, tokenizer, transformer,
loss, backpropagation, AdamW, checkpoint, reload, resume, inference — and print
what happened. Neither writes to disk.

```bash
bun run alpha:train                  # 60 steps, nano preset, seed 1337
bun run alpha:verify --steps 40      # the nine verification checks, A through I
```

Expected from `bun run alpha:train`, reproduced from a real run in this
repository:

```
Model        alpha-nano v0.1.0 (nano) · 128,768 params · 2L/64d/4h · context 64 · vocab 384
Tokenizer    v0.1.0 · 384 tokens · 333 merges · tok_02690d5b
Loss         first 5.9428 → last 3.9579 · best 3.8190 · uniform baseline 5.9506 · validation 4.2403
Checkpoint   ckpt_… · step 60 · stage trained · 515,072 bytes · valid true
Reload       compatible true · max absolute weight difference 0
Resume       step 60 → 62 (+2) · loss after resume 3.8460
Inference    24 token(s) · stop max-tokens · stage trained · deterministic true
```

`bun run alpha:verify` additionally prints the nine checks with their measured
detail, ending in `Verification PASSED`. If either command fails, Alpha's claim
that it can train a model is false and you now know it.

Flags: `--steps`, `--preset nano|micro|small`, `--batch`, `--seq`, `--seed`,
`--prompt`, and `--json` for machine-readable output.

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
identity/integration stack. Alpha reads none of them — the package that would
have used them is no longer a dependency — so delete them from the environment UI
at your convenience. Alpha's own environment values are managed outside the
repository, so `.env.example` is documentation rather than a required file.

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
   baseline; a checkpoint is written to Convex and the stage badge changes. The
   panel also shows the **run record** (run id, seed, fingerprints, step, tokens,
   losses, checkpoints, resume count), the **corpus report**, the **resource
   envelope**, and a **Verify** action that runs checks A–I on a fresh model
   without touching the live weights. *Pause*, *Continue run* and *Stop* appear
   while a run is in progress.
4. **Inference** — generate from the model. Compare it with the untrained
   output, and read the context-window report showing what was actually sent.
   Sampling presets, a deterministic switch, stop sequences and stop token ids
   are all editable, and the result shows its token ids, decoding mode, mean NLL,
   stop reason and request id. A failed request shows the real error — never a
   written answer.
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
| Training seems slow | every step is a real forward and backward pass in JavaScript, and there is no KV cache in inference | lower `totalSteps`, `batchSize` or `seqLen`, or use the `nano` preset — 60 steps at nano takes about 8 seconds |
| Training refuses to start | the configuration exceeds `ALPHA_RESOURCE_LIMITS` | the Resource envelope frame states the limit and the estimate; shrink the model or the sequence length |
| Resume rejected with a compatibility error | the checkpoint's architecture or tokenizer fingerprint does not match the current model | retrain, or point the workspace at the matching configuration — mismatched weights are refused rather than approximately loaded |
| A verification check fails | something in the core is wrong | the check prints its measured values; this is a genuine bug, not a tolerance to widen |
| Generation is nonsense | the model is untrained, or trained on the tiny seed corpus | train it; the badge and warning change when a checkpoint exists. Note that even trained, the shipped corpus yields a working pipeline rather than useful language |
| A generation shows an error instead of text | the model or tokenizer could not produce a token | the real message is shown; there is no fallback answer by design |
| MCP shows `NOT CONFIGURED` | no endpoint has been supplied | construct an `HttpMcpClient` with a real endpoint and call `registerMcpTools` |
