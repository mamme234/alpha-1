# Contributing to Alpha

Alpha is a self-owned AI stack. That constraint is not a preference — it is the
project. Contributions are welcome, and the fastest way to have one accepted is
to respect the rules below.

---

## The four non-negotiables

1. **No external AI/LLM provider.** No OpenAI, Gemini, Claude, Cohere, Mistral,
   Hugging Face Inference, or any other hosted model. No import of an SDK for
   one, no HTTP call to one, no environment variable holding a key for one.
2. **No fake functionality.** No hardcoded responses standing in for model
   output, no mocked inference path, no "example" text presented as generation.
3. **No hidden dependencies.** Weights are initialised in this repository and
   trained by this repository. Do not download a checkpoint.
4. **No mislabelling.** An untrained model is never described as finished, and an
   unimplemented feature is never described as working.

If a change would violate any of these, it will be declined however useful it
looks. If you need a capability Alpha lacks, implement it inside Alpha or label
it `PLANNED` with a note saying what is missing.

---

## Getting started

```bash
bun install
bun run test
bun run typecheck
```

Read [`docs/architecture.md`](docs/architecture.md) first — it is the map, and
it lists the known gaps so you can pick one with confidence.
[`docs/development.md`](docs/development.md) covers conventions, tests and how to
add a module or a tool.

---

## What is most useful right now

| Area | The gap |
| --- | --- |
| Inference | KV caching and batched decoding (generation currently re-runs the prefix) |
| Embeddings | a contrastive objective so embeddings are useful earlier in training |
| Agents | parallel tool execution, multi-agent delegation, a stronger verifier |
| RAG | PDF and HTML parsing, with failures that stay loud |
| Training | gradient accumulation, mixed setups for larger presets, throughput work in the tensor engine |
| Model | larger presets with a corpus big enough to justify them |

Small, correct, well-tested changes beat large speculative ones.

---

## How to submit a change

1. **Open an issue first** for anything large: it is cheaper to agree on the
   interface than to rewrite the implementation.
2. **One concern per change.** A refactor and a feature in one diff is two
   reviews.
3. **Tests are required** for behaviour, and a numerical gradient check is
   required for a new backward pass.
4. **Update the documents.** If you change an interface, update
   `docs/architecture.md` and `docs/api.md`. If you change a module's
   completeness, update `src/alpha/modules.ts` — the landing page and workspace
   read their status from there.
5. **Run the full check** before opening a pull request:

   ```bash
   bunx convex dev --once && bunx tsc -b --noEmit && bun run test
   ```

---

## Code standards

- TypeScript strict mode; `erasableSyntaxOnly` means no enums, no constructor
  parameter properties, no namespaces.
- `src/alpha` must stay free of React, Vite and Convex imports. It is a library,
  and the app layer injects its dependencies.
- Errors are `AlphaError` subclasses with a code, module and details.
- Comments explain *why*. Where a decision exists to protect honesty — refusing
  to plan with untrained weights, requiring approval for durable memory — say so
  in the comment, because that reasoning is part of the interface.
- Prefer explicit names over clever ones. `AlphaInferenceEngine`, not `Engine`.
- Keep functions small enough to read in one pass.

---

## Reviewing your own change

Before pushing, answer these honestly:

- Does anything here produce output that could be mistaken for a real model's?
  If so, is the model stage attached to it?
- Did I leave a capability undocumented or mislabelled?
- Does every new privileged action pass through the policy engine, and is it
  audited?
- Would a reader of `docs/architecture.md` know that this change exists?
- Do the tests prove the behaviour, or only exercise it?

---

## Licensing of contributions

Alpha's source is MIT licensed (see [`LICENSE`](LICENSE)). By contributing you
agree that your contribution is licensed under the same terms. Do not contribute
code, text or data you do not have the right to license, and always record the
licence of a corpus you add — it is copied into every checkpoint trained on it.
