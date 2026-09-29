# Alpha API

Everything the application layer may import comes from
`src/alpha/index.ts`. The subdirectories are implementation; the index is the
contract. This page is the map with short examples.

```ts
import { AlphaWorkspace, createAlphaConfig } from "@/alpha";
```

---

## Workspace

```ts
const workspace = new AlphaWorkspace({
  config: { preset: "nano", training: { totalSteps: 60 } },
  dataset: ALPHA_SEED_CORPUS,
  actorId: "user_123",
  persistence,                 // optional AlphaPersistenceAdapter
  logLevel: "info",
});

await workspace.initialise();   // trains the tokenizer, builds the model, wires modules

for await (const event of workspace.train({ totalSteps: 60 })) { /* step | eval | checkpoint */ }
workspace.generate("Alpha is", { maxNewTokens: 20 });   // GenerationResult
await workspace.ingestDocument({ title, content, license });
workspace.ask("what does Alpha store?");                // RagAnswer with citations
workspace.writeMemory({ scope: "long-term", key, content, approved: true });
await workspace.runAgent("calculate 12 * 4", { maxSteps: 2 });
await workspace.runWorkflow(workflowId, { run: true });
workspace.approveTool("alpha.admin.clear_vector_store");
workspace.resumeFrom(checkpoint);
const snapshot = workspace.snapshot();                  // everything the UI renders
```

`snapshot()` is the single read model: model record, tokenizer, corpus,
training state, inference metrics, RAG documents and collections, memory,
agents, tools, automation, security, observability and the module manifest.

---

## Core

```ts
import { Tensor, matmul, backward, crossEntropy, setGradEnabled } from "@/alpha";

const a = Tensor.from([[1, 2], [3, 4]], true);   // requiresGrad
const y = matmul(a, Tensor.from([[1, 0], [0, 1]]));
backward(y);
a.grad;                                          // real gradients

setGradEnabled(false);                           // pure inference: no graph
```

Ops: `add`, `addScalar`, `scale`, `mulElementwise`, `matmul`, `reshape`,
`transposeLastTwo`, `permute`, `splitHeads`, `mergeHeads`, `sliceSeq`,
`gatherRows`, `layerNorm`, `gelu`, `causalSoftmax`, `dropout`, `crossEntropy`,
`backward`, `resetGrad`, `gradL2Norm`, `maxAbsDiff`.

Also: `AlphaRng` (deterministic, serialisable), `alphaId`, `newTraceId`,
`statusLabel`, `modelStageLabel`, `formatBytes`, and the error classes
(`AlphaValidationError`, `AlphaPermissionError`, `AlphaRateLimitError`,
`AlphaNotImplementedError`, `AlphaToolError`, `AlphaUntrainedModelError`).

---

## Model

```ts
import { AlphaTransformer, ALPHA_MODEL_PRESETS, countParameters, describeArchitecture } from "@/alpha";

const model = new AlphaTransformer(ALPHA_MODEL_PRESETS.nano);
const { logits, hidden } = model.forward(Int32Array.from([1, 2, 3]), 1, 3, {
  training: false,
  returnHidden: true,
});
model.parameterCount;             // exact, equals describeArchitecture's total
model.serializeWeights();         // checkpoint payload
model.embedHidden(ids, 1, 3);     // mean-pooled vector
```

`createModelConfig`, `validateModelConfig`, `deriveStage`, `createModelArtifact`
and the `AlphaModelConfig` / `AlphaModelStage` types are exported alongside.

---

## Tokenizer

```ts
import { AlphaTokenizer } from "@/alpha";

const tokenizer = AlphaTokenizer.train(documents, { vocabSize: 384 });
tokenizer.encode("alpha owns its stack");          // number[]
tokenizer.encodeDetailed(text, { maxLength: 32, padToMaxLength: true, addBos: true });
tokenizer.decode(ids);                              // exact round trip
tokenizer.toJSON();                                 // portable artifact
tokenizer.idFor("<bos>");                           // 2
```

---

## Context

```ts
import { AlphaContextEngine } from "@/alpha";

const context = new AlphaContextEngine({ tokenizer, contextLength: model.config.contextLength });
context.countTokens("how many tokens is this?");
context.fit(text, { maxTokens: 64, strategy: "middle" });   // { text, tokens, truncated }

const assembled = context.assembleConversation({
  prompt: "what does the training engine store?",
  instruction: "Answer from what Alpha knows; say so when it does not.",
  memory: recalledText,
  sources: retrievedText,
  conversation: transcript,
  reserveForOutput: 40,
  maxConversationTurns: 6,
});

assembled.text;         // the exact prompt to send
assembled.usedTokens;   // measured on assembled.text, never an estimate
assembled.budgetTokens; // contextLength - reserveForOutput
assembled.blocks;       // per block: requestedTokens, includedTokens, status, reason
assembled.notes;        // what was truncated, dropped or capped, in words
```

`assemble(blocks, options)` takes arbitrary `ContextBlock`s (`kind`, `text`,
`priority`, `pinned`, `label`). The prompt and pinned blocks are budgeted first;
a block is truncated before it is dropped; headers and labels are dropped before
content; and a pinned block that cannot be included whole raises
`AlphaValidationError` instead of being silently emptied.

`AlphaWorkspace` composes raw generation through this engine, records the result
as `snapshot().context.last`, and logs the adjustments at debug level.

---

## Datasets

```ts
import { encodeCorpus, BatchSampler, datasetStats, splitDocuments } from "@/alpha";

const corpus = encodeCorpus(dataset, tokenizer, { validationFraction: 0.15 });
corpus.stats;                        // trainTokens, validationTokens, unknownTokens
const sampler = new BatchSampler(corpus.trainIds, { batchSize: 4, seqLen: 32 });
sampler.next();                      // { input, target, batch, seqLen }
```

---

## Training

```ts
import { AlphaTrainer, AdamW, learningRateAt, createCheckpoint } from "@/alpha";

const trainer = new AlphaTrainer({ model, tokenizer, dataset, config: { totalSteps: 60 } });
trainer.evaluate({ maxBatches: 4 });        // { loss, perplexity, uniformLoss, … }
trainer.buildCheckpoint();                  // AlphaCheckpoint
trainer.resumeFrom(checkpoint);
trainer.trainToCompletion();                // TrainingSummary
```

---

## Inference

```ts
import { AlphaInferenceEngine, DEFAULT_SAMPLING, argmax } from "@/alpha";
```

See [`docs/inference.md`](inference.md) for the sampling order, stop conditions
and streaming example.

---

## Embeddings, vectors, RAG, memory

```ts
import { AlphaEmbedder, AlphaVectorStore, AlphaRagPipeline, AlphaMemoryStore } from "@/alpha";

const embedder = new AlphaEmbedder({ model, tokenizer, maxTokens: 32 });

const store = new AlphaVectorStore();
store.createCollection({ name: "docs", dimension: embedder.dimension, metric: "cosine" });
store.insert({ id, collection: "docs", vector, text, metadata, sourceId });
store.search({ collection: "docs", vector, topK: 4, minScore: 0.05 });

const rag = new AlphaRagPipeline({ tokenizer, embedder, store, inference });
rag.ingest({ title, content, license });
rag.answer("what does Alpha remember?");             // sources + citations

const memory = new AlphaMemoryStore({ embedder });
memory.write({ scope: "long-term", key, content, approved: true });
memory.retrieve("training pace", { topK: 3 });       // scored, with relevance
memory.forget(id);
```

---

## Tools, MCP, agents

```ts
import { AlphaToolRegistry, registerBuiltinTools, objectSchema } from "@/alpha";

const registry = new AlphaToolRegistry({ policy, rateLimiter, audit });
registerBuiltinTools(registry);
registry.register({
  name: "my.tool",
  description: "Does one thing.",
  module: "custom",
  inputSchema: objectSchema({ text: { type: "string" } }, ["text"]),
  permission: "tool.execute",
  handler: (input) => ({ length: input.text.length }),
});

await registry.execute("my.tool", { text: "hi" }, { actorId: "user_123" });

import { HttpMcpClient, registerMcpTools } from "@/alpha";
const client = new HttpMcpClient({ endpoint: "https://mcp.example/rpc" });
await registerMcpTools(registry, client, { prefix: "mcp" });

import { AlphaAgentRuntime } from "@/alpha";
const runtime = new AlphaAgentRuntime({ registry, policy, inference, memory, audit });
for await (const event of runtime.run({ goal, actorId, maxSteps: 4 })) { /* … */ }
```

---

## Automation

```ts
import { AlphaAutomationEngine } from "@/alpha";

const engine = new AlphaAutomationEngine({ registry, policy, audit, rateLimiter });
const workflow = engine.registerWorkflow({
  name: "nightly stats",
  description: "",
  trigger: { kind: "interval", intervalMs: 3_600_000 },
  conditions: [{ path: "enabled", operator: "equals", value: true }],
  actions: [{ kind: "tool", toolName: "alpha.text.stats", args: { text: "…" } }],
  enabled: true,
  actorId: "alpha.owner",
});
await engine.run(workflow.id, { enabled: true });
engine.start(); engine.stop();
```

---

## Security

```ts
import {
  AlphaPolicyEngine, AlphaRateLimiter, AlphaAuditLog, AlphaSandbox,
  detectPromptInjection, assessInjectionRisk, wrapUntrustedContent, redactSecrets,
} from "@/alpha";

policy.assignRole("user_123", "owner");
policy.registerAgentScope({ agentId, permissions, allowedTools, maxSteps });
policy.assert(actorId, "model.train");
policy.checkTool(actorId, { name, permission, requiresApproval });
policy.approveTool({ actorId, toolName, grantedBy });

const assessment = assessInjectionRisk(documentText);   // level, score, findings
const wrapped = wrapUntrustedContent(documentText, "doc-1");
audit.verifyChain();                                     // { intact, brokenAt }
```

---

## Observability

```ts
import { AlphaObservability, ALPHA_METRICS } from "@/alpha";

const observability = new AlphaObservability({ logLevel: "info" });
observability.recordInference(result, traceId);
observability.recordAgentRun(run);
const snapshot = observability.snapshot();   // metrics, traces, logs, counts
```

---

## Configuration

```ts
import { createAlphaConfig, parseAlphaConfig, serialiseAlphaConfig } from "@/alpha";

const config = createAlphaConfig({ preset: "micro", training: { totalSteps: 200 } });
serialiseAlphaConfig(config);        // JSON for a file
parseAlphaConfig(json);              // validated, merged over defaults
```

Complete examples live in `configs/`.

---

# Application API

Everything above is the AI stack (`src/alpha`), which is framework-free and can
run in any JavaScript host. The application layer sits on top of it and is
documented here because it is the interface a client actually uses.

## Authentication (client)

```tsx
import { useAuth } from "@/hooks/use-auth";

const {
  status,            // "loading" | "signed-in" | "signed-out"
  isAuthenticated,
  user,              // { id, email, displayName, role, status, createdAt, lastSignInAt } | null
  session,           // { createdAt, expiresAt, lastSeenAt, userAgent } | null
  sessions,          // active sessions on the account, with `current`
  sessionToken,
  signIn,            // (credentials) => Promise<user>
  signUp,            // (credentials + displayName?) => Promise<user>
  signOut,           // () => Promise<void>
  signOutEverywhere, // () => Promise<void>
  revokeSession,     // (sessionId) => Promise<void>
  changePassword,    // ({ currentPassword, newPassword }) => Promise<void>
  error,
  clearError,
} = useAuth();
```

`AlphaAuthProvider` (exported from the same module) wraps the app once in
`src/main.tsx`. Calls that fail reject with the server's message;
`alphaAuthErrorMessage(error)` turns any thrown value into something printable.

## Authentication and data (backend)

```ts
import { api } from "@/convex/_generated/api";

// Every Alpha function takes the session token and resolves the account itself.
await api.alphaAuth.actions.register({ email, password, displayName, userAgent });
await api.alphaAuth.actions.signIn({ email, password, userAgent });
await api.alphaAuth.actions.changePassword({ token, currentPassword, newPassword });
await api.alphaAuth.sessions.current({ token });        // { user, session } | null
await api.alphaAuth.sessions.listMine({ token });         // active sessions
await api.alphaAuth.sessions.signOut({ token });
await api.alphaAuth.sessions.signOutEverywhere({ token });
await api.users.currentUser({ sessionToken });
await api.users.updateDisplayName({ sessionToken, displayName });
```

Operator-only (internal, unreachable from a client):

```bash
bunx convex run alphaAuth/maintenance:purgeEndedSessions '{"olderThanMs":0}'
bunx convex run alphaAuth/maintenance:removeAccount '{"emailKey":"someone@example.com"}'
```

## Conversations

```ts
const { conversationId } = await api.alpha.conversations.start({ sessionToken, title, kind });
await api.alpha.conversations.appendMessage({
  sessionToken,
  conversationId,
  role: "assistant",       // user | assistant | system
  content,
  sources,                 // [{ title, score, chunkId }]
  modelStage,              // which Alpha produced this turn
  tokens,
});
await api.alpha.conversations.list({ sessionToken, limit });
await api.alpha.conversations.get({ sessionToken, conversationId });
await api.alpha.conversations.rename({ sessionToken, conversationId, title });
await api.alpha.conversations.remove({ sessionToken, conversationId });
```

Each assistant turn records the `modelStage` that produced it, so a stored
transcript cannot later be read as though a finished model wrote it.

See [`docs/authentication.md`](authentication.md) for the security model behind
these calls, and [`docs/setup.md`](setup.md) for environment variables.
