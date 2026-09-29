# Setup

Alpha is a Vite + React + Convex application with a framework-free AI stack in
`src/alpha`. This page covers running it locally and on the Freebuff platform.

---

## Requirements

- **bun** (package manager and script runner)
- **Node 20+** (for the Convex CLI and Vitest)
- A Convex deployment (the platform provides one; see below)

No GPU, no model download and no API key for an AI provider — there is nothing
to authenticate against.

---

## Install and verify

```bash
bun install
bun run test          # vitest: gradient checks, training, retrieval, security
bun run typecheck     # tsc -b --noEmit
```

The test suite is the fastest way to confirm the AI stack works: it trains a
model on the bundled corpus and asserts the validation loss falls below the
uniform baseline.

---

## Run

```bash
bun run dev           # Vite dev server on http://localhost:5173
bunx convex dev       # Convex backend + generated types
```

On the Freebuff platform both processes are already running in managed
background sessions: edit files and the preview updates. Do not start, stop or
kill those processes yourself, and do not change the Vite server configuration.

Open `/` for the landing page, `/auth` to sign in (email code or guest), and
`/dashboard` for the Alpha workspace.

---

## Environment variables

Client-side (see `.env.example`):

| Variable | Used by |
| --- | --- |
| `VITE_CONVEX_URL` | the Convex React client |
| `CONVEX_DEPLOYMENT` | the Convex CLI |
| `CONVEX_SITE_URL` | auth callbacks |

Auth-related Convex environment variables (`JWKS`, `JWT_PRIVATE_KEY`,
`SITE_URL`) are set on the deployment, not in the repository.

**Never commit secrets.** There is no AI provider key anywhere in Alpha, and
`redactSecrets()` strips anything key-shaped before it reaches a log or an audit
record.

---

## Persistence

Alpha stores its artifacts in Convex tables (`src/convex/alpha/`). Every record
is scoped to the signed-in user by `requireActorId`. Tables created by this
project:

`alphaModels`, `alphaTokenizers`, `alphaDatasets`, `alphaCheckpoints`,
`alphaVectors`, `alphaMemories`, `alphaRuns`, `alphaSpans`, `alphaAuditLogs`,
`alphaTools`, `alphaWorkflows`, `alphaJobs`.

After changing `src/convex/schema.ts` or any Convex function, regenerate types:

```bash
bunx convex dev --once && bunx tsc -b --noEmit
```

Never hand-edit `src/convex/_generated/**`; it is generated and git-ignored.

---

## A first session in the workspace

1. **Overview** — see the model's real stage (`UNTRAINED`), parameter count,
   architecture table and the status of all sixteen modules.
2. **Training** — press *Train Alpha*. Watch the loss curve against the uniform
   baseline; a checkpoint is written to Convex and the stage badge changes.
3. **Inference** — generate from the model you just trained. Compare it with the
   untrained output you saw a moment earlier.
4. **Knowledge** — ingest a document, ask a question, and read the cited chunks.
   Then write a memory, approve it, and delete it.
5. **Work** — run an agent task and read its plan, tool calls and verification
   steps. Register the sample workflow and run it.
6. **Governance** — see the policy decisions, the audit chain's integrity state,
   the latency histograms and the structured logs your session produced.

---

## Working inside the Freebuff template

The application shell is a template with its own conventions. Keep them:

- **Auth.** Use `useAuth()` from `@/hooks/use-auth`. Do not modify
  `src/convex/auth.ts`, `src/convex/auth.config.ts`,
  `src/convex/auth/emailOtp.ts` or `src/convex/users.ts`.
- **Protected routes.** Wrap them in `RequireAuth` from
  `@/components/RequireAuth`; it preserves the requested path in
  `/auth?returnTo=…` and explains the block on the page the visitor asked for.
  Do not hand-roll redirects to `/auth`.
- **Styling.** Tailwind v4 with the tokens in `src/index.css`. Keep the
  `@import "tailwindcss"` line, the `@theme inline` block and the CSS variables;
  Alpha's Studio theme is an extension of them, not a replacement.
- **Vite.** Do not change `vite.config.ts` or the dev-server settings.
- **Rendering.** Routes are lazy-loaded in `src/main.tsx`; keep `Suspense` and
  the existing provider tree (`ConvexAuthProvider`, `BrowserRouter`, `Toaster`).

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Blank preview | a compile error is blocking the app | `bun run typecheck`, then `bunx convex dev --once` |
| `api.alpha.*` missing | Convex codegen is stale | `bunx convex dev --once` |
| Workspace stuck on "training the tokenizer" | an exception during initialisation | the panel prints the error; usually a config conflict (vocabulary > model vocab, or sequence length > context) |
| Training seems slow | every step is a real forward and backward pass in JavaScript, and there is no KV cache in inference | lower `totalSteps`, `batchSize` or `seqLen`, or use the `nano` preset |
| Generation is nonsense | the model is untrained | train it; the badge and warning will change when a checkpoint exists |
| MCP shows `NOT CONFIGURED` | no endpoint has been supplied | construct an `HttpMcpClient` with a real endpoint and call `registerMcpTools` |
