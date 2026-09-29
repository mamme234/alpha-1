# Alpha Inference

Alpha generates text with Alpha's own weights, in the same process that defines
them. There is no provider call, no fallback path and no demo branch: if the
weights are untrained, the output is what untrained weights produce, and the
result says so.

```
src/alpha/inference/engine.ts   sampling, generation, streaming, stop conditions
src/alpha/embeddings/embedder.ts vectors from the same model
```

---

## Generating

```ts
import { AlphaInferenceEngine } from "@/alpha";

const engine = new AlphaInferenceEngine({ model, tokenizer, stage: "untrained" });

const result = engine.generate("Alpha is a self owned", {
  temperature: 0.8,
  topK: 40,
  topP: 0.95,
  maxNewTokens: 40,
  repetitionPenalty: 1.1,
  stopSequences: ["\n\n"],
  seed: 2026,
});

result.text;            // sampled continuation
result.stopReason;      // "eos" | "stop-sequence" | "max-tokens" | "context-limit"
result.modelStage;      // "untrained" until a checkpoint exists
result.warning;         // why that output should not be trusted as language
```

### Sampling order

For each step, over the logits of the final position:

1. **Repetition penalty** — divide positive logits and multiply negative ones for
   tokens already generated.
2. **Special tokens** — `<pad>` is banned from free generation.
3. **Temperature** — divide logits, then softmax.
4. **Top-k** — keep the `k` most probable tokens (0 disables).
5. **Top-p** — keep the smallest set whose cumulative probability reaches `p`.
6. **Sample** from the surviving mass using Alpha's deterministic RNG.

The same seed and settings reproduce the same sample exactly, which is what
makes a generation reproducible rather than merely similar.

### Stop conditions

| Reason | When |
| --- | --- |
| `eos` | the model emitted `<eos>` |
| `stop-sequence` | the decoded text ended with one of your stop strings |
| `max-tokens` | `maxNewTokens` reached |
| `context-limit` | prompt + completion reached the model's context window |

`context-limit` is a hard stop, not a silent wrap-around. Prompts longer than
the window are truncated from the left, and the tokenizer reports it.

---

## Streaming

```ts
const stream = engine.generateStream(prompt, sampling);
let next = await stream.next();
while (!next.done) {
  render(next.value.text);      // cumulative text so far
  next = await stream.next();
}
const result = next.value;      // the same GenerationResult as generate()
```

The workspace uses this path, so tokens appear as they are produced while the
metrics panel records latency, throughput and the stop reason for the finished
generation.

---

## Through the workspace

```ts
const workspace = await new AlphaWorkspace({ actorId }).initialise();
const result = workspace.generate("Alpha is a self owned", { maxNewTokens: 40 });
```

Every generation records a span, updates metrics (requests, latency histogram,
tokens generated, stop reason) and writes a run record
(`kind: "inference"`) to Convex with the model stage attached.

---

## Retrieval-augmented answering

```ts
const answer = workspace.ask("how are approvals recorded?");

answer.answer;             // generated from the retrieved context
answer.sources;            // [{ rank, chunkId, title, score, excerpt }]
answer.contextTokens;
answer.answeredWithoutContext;   // true when retrieval found nothing
answer.modelStage;
```

The prompt template is a plain, inspectable function
(`buildRagPrompt`): the sources sit inside a delimited block introduced as
*data*, not as instructions, and the model is told to answer from that block
only. Injection patterns are scored before the block is built, and delimiters
inside the content are neutralised so a document cannot close the block itself.

Citation scores are real cosine similarities from Alpha's own embeddings — so
even while the model is untrained, the retrieval half of the pipeline is
verifiable in the UI.

---

## Embeddings

```ts
const record = embedder.embed("Alpha owns its own stack");
record.vector;       // number[], L2-normalised, dimension = model dModel
record.tokens;       // tokens used
record.truncated;    // true when the text exceeded the embedding window
record.modelStage;
```

`embedQuery` and `embedDocuments` share the encoder — Alpha has no separate
retrieval model. Similarity helpers (`cosineSimilarity`, `dotProduct`,
`euclideanDistance`, `centroid`) are in the same module.

**Honest statement about quality.** Embeddings are pooled hidden states of a
model trained on next-token prediction. Until training converges, that geometry
is close to meaningless, and the embedder reports `modelStage: "untrained"`
rather than implying otherwise. A contrastive objective is listed as missing in
[`docs/architecture.md`](architecture.md#known-gaps).

---

## What is not implemented

Stated plainly, because pretending would defeat the point of the project:

- **KV cache.** Generation re-runs the prefix for every token. Cost is O(T²) per
  token; the UI says so. This is the first optimisation to add.
- **Batched decoding.** One sequence at a time.
- **Speculative decoding, beam search, constrained decoding.** Not implemented.
- **Grammar/JSON-constrained output.** The agent runtime validates JSON after the
  fact instead of constraining the sampler.

Until a KV cache exists, keep `maxNewTokens` and `contextLength` modest, and
expect generation to slow down as the context grows.
