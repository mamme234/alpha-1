# Development

How to work on Alpha without breaking it — and specifically without breaking its
honesty rules, which are the point of the project.

---

## Commands

```bash
bun install
bun run test           # vitest run (all suites)
bun run test:watch     # vitest in watch mode
bun run typecheck      # tsc -b --noEmit
bun run lint           # eslint
bun run format         # prettier
bunx convex dev --once # regenerate Convex types after touching src/convex
```

The full check after a change to Convex or the AI stack:

```bash
bunx convex dev --once && bunx tsc -b --noEmit && bun run test
```

---

## Repository rules

1. **`src/alpha` is framework-free.** No React, Vite or Convex imports inside it.
   Persistence, UI and hosting are injected by the app layer. This is what makes
   the same stack runnable in a browser, a script and a test.
2. **Import from the index.** The app layer imports from `@/alpha` only; the
   subdirectories import from their siblings.
3. **No external AI provider.** Not as a fallback, not behind a flag, not in a
   comment. If you need a capability Alpha lacks, implement it or label it
   `PLANNED`.
4. **No fake output.** Nothing may return a canned string that looks like model
   output. If a capability is missing, throw `AlphaNotImplementedError`.
5. **Statuses come from reality.** Add a module to `src/alpha/modules.ts` with an
   honest status and specific notes; the landing page and workspace render it.
6. **Generated code is not edited.** `src/convex/_generated/**` is produced by
   the Convex CLI and git-ignored.

---

## Testing

Tests live in `src/alpha/tests/` and run under Vitest.

| File | Covers |
| --- | --- |
| `tensor.test.ts` | every op's analytic gradient against central differences; causal masking; head split/merge; cross-entropy |
| `model.test.ts` | forward shapes, exact parameter counts, causality (a later token cannot change earlier logits), serialisation round trip, stage derivation |
| `tokenizer.test.ts` | training, exact round trip, special tokens, truncation/padding, unknown counting, JSON round trip |
| `training.test.ts` | corpus encoding, shifted targets, loss falling below the uniform baseline, checkpoint round trip, resuming, schedule, clipping |
| `vector-memory.test.ts` | vector CRUD/search/persistence, embeddings and similarity, memory scopes, approval, relevance scoring, deletion |
| `security-tools.test.ts` | policy allow/deny, agent scopes, approval gates, injection detection, redaction, rate limits, audit chain, tool registry, calculator, schema validation |
| `rag-agents.test.ts` | chunking with overlap, retrieval, citations, injection-safe context, agent runs and sandboxing, automation conditions/retries, workspace integration |

`src/alpha/tests/helpers.ts` builds a real stack (trained tokenizer, real
transformer, real stores) rather than mocks — the tests exist to prove the
modules work together, so mocking them would miss the point.

### Writing a gradient test

```ts
expectGradientMatches([weight, bias], () => layerNorm(x, weight, bias), sumSquares);
```

The helper runs the op, backpropagates a scalar loss, then perturbs each
parameter and compares against the central difference. If you add an operation,
add it there. A numerical gradient check is the only convincing evidence that a
backward pass is correct.

---

## Adding a module

1. Create `src/alpha/<module>/` with focused files; keep the public surface in
   that directory's types.
2. Export the new API from `src/alpha/index.ts`.
3. Add a descriptor to `ALPHA_MODULES` in `src/alpha/modules.ts` with an honest
   status and notes that name what is missing.
4. Wire it into `AlphaWorkspace` if it participates in the live system, and add
   it to `snapshot()` if it has state worth showing.
5. Add tests. If it has a backward pass, add a numerical gradient check.
6. Document it in `docs/architecture.md` (module tables and interfaces) and, if
   it is user-facing, in the relevant doc under `docs/`.

---

## Adding a tool

```ts
registry.register({
  name: "alpha.text.stats",                 // dotted lowercase
  description: "…",
  module: "tools",
  inputSchema: objectSchema({ text: { type: "string" } }, ["text"]),
  permission: "tool.execute",               // or tool.execute.dangerous
  requiresApproval: false,
  characteristics: { mutates: false, networked: false, fileSystem: false },
  tags: ["text", "count"],
  handler: (input, context) => ({ … }),
  verify: (output) => ({ ok: true, reason: "…" }),
});
```

- Declare the **permission** the tool genuinely needs; a write tool should not
  ask for `tool.execute`.
- Mark destructive tools `requiresApproval: true` and give them a narrow
  permission (`tool.execute.dangerous`). The policy engine enforces the gate.
- Implement `verify` when "no output" is not the same as "failure" — the agent
  runtime uses it to decide whether a step succeeded.
- Never accept an argument you do not validate; schemas are enforced before the
  handler runs.

---

## Conventions

- **TypeScript strict**, `erasableSyntaxOnly`: no enums, no constructor parameter
  properties, no namespaces.
- **Errors** are `AlphaError` subclasses with a code, a module and details.
- **Comments** explain *why*, especially where a decision protects honesty (for
  example, refusing to plan with untrained weights).
- **Naming** is explicit: `AlphaTrainer`, `AlphaRagPipeline`, `AlphaPolicyEngine`.
  Parameter names in the model are stable strings (`layer0.attn.wq`).
- **Formatting** is Prettier with the repository config.

---

## Performance notes

- The tensor engine is straightforward JavaScript with `Float32Array` storage.
  It is fast enough for the presets in `src/alpha/model/config.ts` and honest
  about its limits.
- Training in the browser yields to the event loop between steps so the UI keeps
  painting; keep that behaviour when touching the trainer.
- Inference has no KV cache yet — that is the highest-value optimisation to add.
  If you implement it, add a test that proves cached and uncached decoding
  produce the same tokens for the same seed.

---

## Definition of done

A change is finished when:

1. `bunx tsc -b --noEmit` passes (and `bunx convex dev --once` after Convex edits).
2. `bun run test` passes, with new tests for new behaviour.
3. Anything unfinished is labelled, not hidden: `PLANNED`, `IN DEVELOPMENT`,
   `NOT CONFIGURED` or `UNTRAINED`, with a note saying what is missing.
4. Docs and the module manifest match the code.
