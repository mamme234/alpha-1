# Alpha Architecture

Alpha is one system made of sixteen subsystems. This document describes what
each one is, the interface it exposes, and how data moves between them. It is
written to be checkable against the code: every module listed here exists at the
path given, and every status matches
[`src/alpha/modules.ts`](../src/alpha/modules.ts).

---

## Status of each module

| Module | Path | Status | Meaning of that status |
| --- | --- | --- | --- |
| Core | `src/alpha/core` | **READY** | Complete; gradients verified numerically in tests |
| Model | `src/alpha/model` | **READY** | Architecture complete; weights are untrained until a run completes |
| Tokenizer | `src/alpha/tokenizer` | **READY** | Trains from a corpus in one call; versioned and serialisable |
| Context | `src/alpha/context` | **READY** | Budget measured on the exact prompt; every block reported as included, truncated or dropped |
| Datasets | `src/alpha/datasets` | **READY** | Corpus type, splitting, tokenisation, batching |
| Training | `src/alpha/training` | **READY** | Real backprop, AdamW, checkpoints, resumable |
| Inference | `src/alpha/inference` | **IN DEVELOPMENT** | Sampling and streaming work; no KV cache or batched decoding yet |
| Embeddings | `src/alpha/embeddings` | **IN DEVELOPMENT** | Pooled hidden states work; no contrastive objective yet |
| Vector store | `src/alpha/vector` | **READY** | Exact search, metadata, persistence, deterministic ties |
| RAG | `src/alpha/rag` | **READY** | Full pipeline; text and markdown only (others fail loudly) |
| Memory | `src/alpha/memory` | **READY** | Three scopes, local scoring, approval, deletion |
| Agents | `src/alpha/agents` | **IN DEVELOPMENT** | Capability planner works; model planner needs trained weights; no parallel tools |
| Tools | `src/alpha/tools` | **READY** | Registry, validation, permissions, execution records |
| MCP | `src/alpha/mcp` | **NOT CONFIGURED** | Client is implemented and tested; no endpoint is configured by default |
| Automation | `src/alpha/automation` | **READY** | Workflows, conditions, queue, retries, history |
| Security | `src/alpha/security` | **READY** | Policy, validation, rate limits, audit chain, sandboxes |
| Observability | `src/alpha/observability` | **READY** | Spans, metrics, logs, cross-module recording |

---

## The composition root

`src/alpha/workspace.ts` builds the system:

```
AlphaConfig
   │
   ├─► AlphaTokenizer.train(corpus)         vocab size, merges, version
   │        │
   │        └─► AlphaTransformer(config)    vocab matches the tokenizer exactly
   │                 │
   │                 ├─► AlphaEmbedder          pooled hidden states
   │                 ├─► AlphaInferenceEngine   sampling + streaming
   │                 ├─► AlphaContextEngine     window budgeting + trim report
   │                 └─► AlphaTrainer           batching + backprop + AdamW
   │
   ├─► AlphaVectorStore ──► AlphaRagPipeline ──► AlphaInferenceEngine
   ├─► AlphaMemoryStore ──► AlphaEmbedder
   ├─► AlphaToolRegistry ──► AlphaAgentRuntime
   ├─► AlphaAutomationEngine ──► (tools, agents, memory)
   └─► AlphaPolicyEngine · AlphaRateLimiter · AlphaAuditLog · AlphaObservability
```

Nothing in the tree imports the application layer. `src/alpha` has no Convex,
React or Vite dependency, which is why the same object runs in a browser tab,
in a Node script and in tests. Persistence is injected as an optional adapter.

---

## Data flows

### Training

```
dataset documents
  → encodeCorpus (BOS/EOS per document, train/validation split)
  → BatchSampler (random windows, targets shifted by one)
  → AlphaTransformer.forward (causal attention, real graph)
  → crossEntropy (fused softmax + NLL, padding-aware)
  → backward (topological reverse pass over the recorded graph)
  → AdamW with global-norm clipping and a warmup/decay schedule
  → metrics → checkpoint (weights, optimiser moments, RNG position, metrics)
```

### Retrieval and answering

```
document text
  → parse (text | markdown; anything else raises AlphaNotImplementedError)
  → chunk (token windows with overlap, character offsets preserved)
  → embed (Alpha hidden states, mean-pooled, L2-normalised)
  → vector store (collection per corpus, metadata per chunk)
  → query embed → exact cosine scan → top-k
  → context assembly (delimited data block + citations)
  → AlphaInferenceEngine → answer + source references + model stage
```

### Agent run

```
goal
  → planner (capability match, or Alpha's model when it is trained)
  → for each step: sandbox check → policy check → tool execute → verify
  → synthesis (model, when available; otherwise a labelled structured summary)
  → run record: plan, steps, tool calls, verification, sandbox report, blocker
```

---

## Module interfaces

### `core` — tensor and autodiff

```ts
class Tensor { shape: number[]; data: Float32Array; grad: Float32Array | null }

backward(root: Tensor): void
matmul(a, b) · add(a, b) · layerNorm(a, w, b, eps) · gelu(a) · causalSoftmax(scores)
crossEntropy(logits, targets, ignoreIndex) · splitHeads / mergeHeads / permute
setGradEnabled(false)  // pure inference, no graph is recorded
```

Every operation has an analytic backward pass; the tests compare them against
central differences of the loss.

### `model` — the transformer

```ts
new AlphaTransformer(config, seed?)
model.forward(ids, batch, seq, { training, rng, returnHidden }) → { logits, hidden }
model.embedHidden(ids, batch, seq) → Float32Array       // mean-pooled
model.parameters() / parameterMap() / serializeWeights() / loadWeights()
countParameters(config) · describeArchitecture(config) · deriveStage(...)
```

### `context` — fitting the window

```ts
const engine = new AlphaContextEngine({ tokenizer, contextLength })
engine.countTokens(text)                                  // this vocabulary's count
engine.fit(text, { maxTokens, strategy })                 // right | left | middle
engine.assemble(blocks, { reserveForOutput })             // blocks + full report
engine.assembleConversation({ prompt, instruction, memory, sources, conversation })
```

A model has one hard number — its context length — and everything Alpha wants
to say competes for it. The engine allocates by priority (pinned blocks first,
then instruction, then memory, sources and conversation, with the prompt
outranking all of them), truncates before it drops, and **measures the budget on
the exact rendered prompt**, so `usedTokens` describes the text that is actually
sent rather than an estimate of it.

Three properties make the report trustworthy:

- **Nothing is lost silently.** Every block appears in `blocks` as `included`,
  `truncated` or `dropped` with a reason, including blocks removed by a per-kind
  cap, and `notes` carries the human-readable summary.
- **Content outranks decoration.** Headers and labels are scaffolding. When they
  would starve a block, they are dropped and the words are kept.
- **Pinned means pinned.** A pinned block is included whole or the assembly
  throws; it is never quietly trimmed to nothing.

---

### `training` — the trainer

```ts
const trainer = new AlphaTrainer({ model, tokenizer, dataset, config })
for (const event of trainer.run()) { /* step | eval | checkpoint | done */ }
trainer.evaluate({ maxBatches })      // deterministic validation pass
trainer.buildCheckpoint()             // artifact with weights + optimiser + RNG
trainer.resumeFrom(checkpoint)        // continues rather than restarts
```

`run()` is a generator so a browser can train in slices without freezing the
tab; `trainToCompletion()` is the synchronous convenience for scripts and tests.

### `inference` — generation

```ts
const engine = new AlphaInferenceEngine({ model, tokenizer, stage })
engine.generate(prompt, sampling) → GenerationResult   // text + provenance
engine.generateStream(prompt, sampling) → AsyncGenerator<chunk, GenerationResult>
```

Every `GenerationResult` carries `modelStage` and a `warning` while the weights
are untrained; there is no code path that omits it.

### `embeddings`, `vector`, `rag`, `memory`

```ts
embedder.embed(text) / embedQuery(text) / embedDocuments(texts) → EmbeddingRecord
store.createCollection({ name, dimension, metric }) · insert · update · delete
store.search({ collection, vector, topK, minScore, filter }) → VectorSearchHit[]
pipeline.ingest({ title, content, license }) · retrieve(query) · answer(query)
memory.write({ scope, key, content, approved }) · retrieve(query) · forget(id)
```

Long-term memory writes require `approved: true`, which the security layer
enforces again through the `memory.write.long-term` permission.

### `tools`, `mcp`, `agents`

```ts
registry.register({ name, description, module, inputSchema, permission, handler })
await registry.execute(name, args, { actorId, traceId }) → ToolRunResult
registry.discover({ query, actorId }) → ToolDescriptor[]   // scope-filtered

await registerMcpTools(registry, client, { permission, prefix })
runtime.run({ goal, actorId, maxSteps }) → AsyncGenerator<AgentEvent, AgentRunResult>
```

Tool execution order is fixed: rate limit → authorization (permission + agent
scope + approval) → schema validation → handler → result. Failures come back as
values; permission failures throw, because they are orchestration bugs.

### `automation`, `security`, `observability`

```ts
engine.registerWorkflow({ name, trigger, conditions, actions, enabled })
await engine.run(workflowId, payload) → JobExecution
engine.start() / engine.stop()          // interval triggers, host-owned

policy.assignRole(actorId, role) · policy.registerAgentScope(scope)
policy.checkTool(actorId, tool) · policy.approveTool({ actorId, toolName, grantedBy })
limiter.assert(actorId, kind) · audit.append(entry) · audit.verifyChain()
sandbox.enterTool(tool, { networked, fileSystem })

observability.recordInference(result, traceId) · recordTrainingStep(point)
observability.recordRagRetrieval(answer) · recordAgentRun(run) · snapshot()
```

---

## Security model

1. **Every action has an actor.** Tools, agents, workflows and memory writes
   carry an `actorId`; there is no anonymous internal caller.
2. **Roles grant permissions; scopes narrow them.** An agent's effective
   permissions are its role's permissions intersected with its registered scope.
3. **Allow-lists, not deny-lists.** An agent with an empty tool list can call
   nothing. The agent actor Alpha registers holds six non-destructive tools.
4. **Approval gates.** A tool marked `requiresApproval` cannot execute until a
   human approval is recorded; the approval is short-lived and audited.
5. **Sandboxes.** Step, token and time budgets; network and filesystem are denied
   by default and must be enabled explicitly on the sandbox spec.
6. **Untrusted text is data.** Retrieved chunks are wrapped in delimiters with
   their own markers neutralised, and injection patterns are scored before the
   context reaches the model.
7. **Append-only audit.** Records are hash-chained, so editing or deleting one is
   detectable by replaying the chain (`AlphaAuditLog.verifyChain()`).

Detection is heuristic and is labelled as such in the workspace. The point is to
reduce risk and to make failures visible, not to claim safety.

---

## Observability

Traces carry a trace id and per-module spans (inference, retrieval, tool, agent,
workflow, training, embedding). Metrics are counters and histograms over a
bounded sample window, with percentiles computed from the actual samples.
Structured logs are objects with `module`, `traceId` and redacted data, so a
pasted key never lands in a stored record.

The workspace shows three things that make this checkable: the loss curve from
the trainer's own history, the token/latency numbers from the inference engine,
and the audit chain's integrity state.

---

## Persistence

Alpha owns its artifacts and stores them in Convex — the application database,
not an AI provider:

| Table | Holds |
| --- | --- |
| `alphaModels` | model records with derived stage and parameter counts |
| `alphaTokenizers` | trained vocabularies (merges, stats, provenance) |
| `alphaDatasets` | corpora with licences |
| `alphaCheckpoints` | weights, optimiser state, RNG position, metrics |
| `alphaVectors` | embeddings with metadata and collection names |
| `alphaMemories` | memory records with scope and approval state |
| `alphaRuns` | inference, training, rag, agent and workflow records |
| `alphaSpans` | spans written by Alpha's tracer |
| `alphaAuditLogs` | hash-chained audit records |
| `alphaTools` | the mirrored tool surface with permissions |
| `alphaWorkflows` / `alphaJobs` | workflow definitions and executions |

Every table is scoped to the signed-in user by
`src/convex/alpha/helpers.ts#requireActorId`.

---

## Known gaps

Listed here so the architecture document is not a sales page:

- **Inference:** no KV cache and no batched decoding; generation re-runs the
  prefix each token. Cost is O(T²) per token and is stated in the UI.
- **Embeddings:** no contrastive training objective yet, so embedding quality
  tracks the language objective and is poor while the model is undertrained.
- **Agents:** no parallel tool execution, no multi-agent delegation, and the
  model planner refuses to run without trained weights.
- **RAG:** PDF and HTML parsers are not implemented; unsupported kinds raise.
- **MCP:** no server is configured by default; the client is real but idle.
- **Model scale:** the presets are deliberately small enough to train in a tab.
  Scaling up is a configuration change, not a code change — and it will need
  proportionally more data and compute.
