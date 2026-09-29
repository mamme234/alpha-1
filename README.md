# Alpha

**A self-owned AI/LLM system.** Alpha's language model is defined, initialised
and trained inside this repository. There is no OpenAI, Gemini, Claude or any
other external AI provider anywhere in the stack — not as a fallback, not behind
a flag, not as a placeholder waiting for a key.

```
ALPHA
├── Alpha LLM Core          decoder-only transformer, built from scratch
├── Alpha Tokenizer         BPE trained on Alpha's own corpus
├── Alpha Context Engine    window budgeting, block priority, trim reporting
├── Alpha Training Engine   batching, backprop, AdamW, checkpoints, resuming
├── Alpha Inference Engine  sampling, streaming, stop conditions
├── Alpha Embeddings        vectors from Alpha's own hidden states
├── Alpha RAG               ingest → chunk → embed → retrieve → answer
├── Alpha Memory            conversation / session / long-term, with approval
├── Alpha Agents            plan → act → verify → synthesise
├── Alpha Tool / MCP Layer  registration, schemas, permissions, execution
├── Alpha Vector Store      collections, metadata, exact similarity search
├── Alpha Automation        workflows, triggers, conditions, jobs, retries
├── Alpha Security          roles, scopes, approvals, audit, injection defence
└── Alpha Observability     traces, metrics, structured logs
```

## The honest state of Alpha

This project is a **foundation**, and it labels itself accordingly. Model state
is derived from what actually exists — never asserted:

| Label | Meaning |
| --- | --- |
| `ARCHITECTURE ONLY` | code shape defined, no weights instantiated |
| `UNTRAINED` | parameters exist, but they are random initialisation |
| `TRAINED (FROM SCRATCH)` | a training run in this repository produced a checkpoint |
| `FINE-TUNED` | a trained model continued learning on a narrower corpus |
| `PRODUCTION` | a human promoted it |

Module state uses the same discipline: `PLANNED`, `IN DEVELOPMENT`,
`NOT CONFIGURED` and `READY` are read from
[`src/alpha/modules.ts`](src/alpha/modules.ts), which is also what the landing
page and the workspace render. A subsystem is never described as better than it
is — see [`docs/architecture.md`](docs/architecture.md#status-of-each-module).

**Alpha does not have a pretrained large model.** What it has is a real
transformer with real gradients, a real training loop and a real sampling
stack, sized so it can be trained in a browser tab or a Node script. Scale it up
by changing a config.

## What Alpha refuses to do

- No external AI/LLM API calls, at any layer, for any reason.
- No fake AI responses, no hardcoded demo answers, no "example" generated text.
- No hidden external model dependency and no downloadable weights.
- No placeholder integrations presented as working features.
- No claiming an untrained model is a finished assistant.

If a capability is architected but unfinished, the code throws
`AlphaNotImplementedError` and the manifest says `PLANNED` or
`IN DEVELOPMENT` instead of quietly returning something made up.

## Quickstart

```bash
bun install
bun run test          # 96+ tests: gradient checks, training, retrieval, security
bun run typecheck     # tsc -b --noEmit
bun run dev           # Vite dev server (the platform runs this for you)
```

The browser workspace (`/dashboard`) is where Alpha actually runs: it trains the
tokenizer on the bundled corpus, builds the model, trains it with real gradient
descent, plots the loss against a uniform baseline, stores the checkpoint, and
lets you generate from the model it produced.

Full instructions: [`docs/setup.md`](docs/setup.md) ·
[`docs/development.md`](docs/development.md).

## Repository layout

```
src/alpha/                  the AI stack — framework-free TypeScript
  core/                     tensor engine + reverse-mode autodiff
  model/                    transformer, configuration, versioning
  tokenizer/                trainable BPE vocabulary
  datasets/                 corpus type, splitting, batching, seed corpus
  training/                 optimiser, schedule, checkpoints, trainer
  inference/                sampler, generation, streaming
  embeddings/               pooled hidden states, similarity functions
  vector/                   self-owned vector store
  rag/                      ingestion, chunking, retrieval, citations
  memory/                   scoped memory with local relevance scoring
  tools/                    registry, JSON Schema validation, built-ins
  mcp/                      MCP protocol shapes, HTTP client, tool adapter
  agents/                   planner, runtime, verification, run records
  automation/               workflows, queue, retries, history
  security/                 policy, validation, rate limits, audit, sandbox
  observability/            tracer, metrics, logger, facade
  configs/                  one configuration object for the whole stack
  modules.ts                the manifest every status is read from
  workspace.ts              composition root (wires every module together)
  tests/                    vitest suites, including numerical gradient checks

src/convex/                 Alpha's own persistence (Convex) + auth
src/components/alpha/       workspace UI (Studio theme)
src/pages/                  landing, auth, workspace
docs/                       architecture, model, training, inference, API, setup
configs/                    example configuration files
```

The app layer is thin on purpose: it imports from `src/alpha` only through
[`src/alpha/index.ts`](src/alpha/index.ts), the public API surface, and stores
artifacts in Convex tables (`alphaModels`, `alphaCheckpoints`, `alphaVectors`,
`alphaMemories`, `alphaRuns`, `alphaSpans`, `alphaAuditLogs`, …). Convex is the
database here, never a model provider.

## Documentation

| Document | Covers |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | every module, its interface, data flows, and its real status |
| [`docs/model.md`](docs/model.md) | the transformer, configuration, parameter counts, versioning and stages |
| [`docs/training.md`](docs/training.md) | corpus, tokenizer training, the training loop, checkpoints, resuming |
| [`docs/inference.md`](docs/inference.md) | sampling, streaming, stop conditions, embeddings, what is missing |
| [`docs/api.md`](docs/api.md) | the public API, grouped by module, with usage examples |
| [`docs/setup.md`](docs/setup.md) | install, run, environment variables, platform notes |
| [`docs/development.md`](docs/development.md) | tests, typechecking, conventions, adding a module |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | how to contribute without breaking Alpha's honesty rules |

## Working in this repository

- **Package manager:** bun.
- **Auth:** use `useAuth()` from `@/hooks/use-auth`. Do not modify
  `src/convex/auth.ts`, `src/convex/auth.config.ts` or
  `src/convex/auth/emailOtp.ts`.
- **Protected routes:** wrap them in `RequireAuth`, which preserves the
  requested path in `/auth?returnTo=…`.
- **Persistence:** mutations in `src/convex/alpha/` are scoped to the signed-in
  user and are the only writer for Alpha artifacts.
- **Never** add an external AI provider, a model download or an API key for a
  hosted model. That constraint is the point of the project.

## Licence

The Alpha source in this repository is released under the MIT licence — see
[`LICENSE`](LICENSE). The bundled seed corpus
([`src/alpha/datasets/seed-corpus.ts`](src/alpha/datasets/seed-corpus.ts)) was
written for this repository and is released as CC0-1.0.
