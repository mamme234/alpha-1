/**
 * Alpha — module manifest.
 *
 * The single source of truth for what each Alpha subsystem actually is right
 * now. The workspace, the landing page and the docs all read from here, so a
 * status can never drift between the code and the screen.
 *
 * Status vocabulary (see `core/types`):
 *   PLANNED · IN DEVELOPMENT · UNTRAINED · NOT CONFIGURED · READY
 */

import type { AlphaModuleDescriptor } from "./core/types";

export const ALPHA_MODULES: AlphaModuleDescriptor[] = [
  {
    id: "auth",
    name: "Alpha Authentication",
    summary:
      "Alpha's own accounts and sessions: PBKDF2-HMAC-SHA256 passwords, hashed session tokens, absolute and idle expiry, per-session and account-wide revocation.",
    status: "ready",
    notes: [
      "No identity provider is involved. Registration, sign-in, sign-out and password change are Alpha functions in src/convex/alphaAuth.",
      "A session token is stored only as sha256(token); the plaintext exists only in the client that created it.",
      "Failed sign-ins are counted and lock the account; a password change retires every existing session.",
      "Session tokens are held in browser localStorage because Alpha's API is called directly rather than over cookies — see docs/authentication.md.",
    ],
  },
  {
    id: "api",
    name: "Alpha API",
    summary:
      "The application boundary: authentication, user data, model management, conversations, memory, vectors, tools, agents, RAG, observability and security, each with its own module.",
    status: "in-development",
    notes: [
      "Every Alpha function resolves its account from a session, so an owner can never be chosen by the client.",
      "The AI runtime itself (src/alpha) is framework-free and has no backend dependency, so other clients can host it without this API.",
      "Served over Convex functions today. A transport-neutral HTTP surface with token auth is planned, not built.",
    ],
  },
  {
    id: "core",
    name: "Alpha Core",
    summary: "Tensor engine and reverse-mode autodiff with analytic gradients for every operation.",
    status: "ready",
    notes: [
      "Rank 1-3 float32 tensors, batched matmul, layer norm, GELU, causal softmax, dropout, cross-entropy.",
      "Every op has a real backward pass; gradients are verified against numerical differences in tests.",
      "No numerical library is used as a backend.",
    ],
  },
  {
    id: "model",
    name: "Alpha LLM Core",
    summary: "Decoder-only transformer: causal multi-head attention, feed-forward blocks, residual stream, tied output projection.",
    status: "ready",
    notes: [
      "Configurable width, depth, heads, context length and feed-forward size.",
      "Weights are random initialisation until a training run produces a checkpoint — labelled UNTRAINED until then.",
      "Model versioning is explicit; five stages are never conflated (architecture, untrained, trained, fine-tuned, production).",
    ],
  },
  {
    id: "tokenizer",
    name: "Alpha Tokenizer",
    summary: "Byte-of-character BPE trained on Alpha's own corpus, with special tokens, padding, truncation and vocabulary versioning.",
    status: "ready",
    notes: [
      "The merge table is learned from the corpus and stored in the artifact.",
      "Characters outside the trained alphabet map to <unk> and are counted, not hidden.",
      "A trained vocabulary is portable: it serialises to JSON and reloads exactly.",
    ],
  },
  {
    id: "context",
    name: "Alpha Context Engine",
    summary:
      "Fits instructions, memory, retrieved sources, conversation and the prompt into the model's window, and reports exactly what was truncated or dropped.",
    status: "ready",
    notes: [
      "The prompt and pinned blocks are budgeted first; retrieved context and conversation are the first things shortened.",
      "Every block reports requested tokens, included tokens and the reason, so a trimmed prompt is visible rather than silent.",
      "The budget is measured on the exact rendered prompt, so the reported usage is the text that was actually sent.",
      "The last assembly is shown in the inference workspace and logged at debug level; retrieval answers fence and budget their sources separately.",
    ],
  },
  {
    id: "training",
    name: "Alpha Training Engine",
    summary: "Corpus encoding, batching, train/validation split, cross-entropy, AdamW, LR schedules, evaluation, checkpoints, resumable runs.",
    status: "ready",
    notes: [
      "Training runs through Alpha's own autodiff — backpropagation, not a stub.",
      "Checkpoints carry weights, optimiser moments, RNG position and metrics so a run resumes rather than restarts.",
      "Runs as a generator so a browser can train in slices without freezing.",
    ],
  },
  {
    id: "inference",
    name: "Alpha Inference Engine",
    summary: "Local generation with temperature, top-k, top-p, repetition penalty, stop sequences, max tokens and streaming.",
    status: "in-development",
    notes: [
      "Runs only Alpha's own weights; there is no fallback provider and no demo response path.",
      "Every result carries the model stage and a warning while the weights are untrained.",
      "Missing: KV caching and batched decoding — generation currently re-runs the prefix each token.",
    ],
  },
  {
    id: "embeddings",
    name: "Alpha Embeddings",
    summary: "Vectors from Alpha's own hidden states: mean or last-token pooling, L2 normalisation, cosine/dot/euclidean similarity.",
    status: "in-development",
    notes: [
      "No embedding API is used; the encoder is Alpha's transformer.",
      "Quality tracks training progress — untrained weights give meaningless geometry, which is reported, not hidden.",
      "Missing: a contrastive objective to make embeddings useful before the language objective converges.",
    ],
  },
  {
    id: "rag",
    name: "Alpha RAG",
    summary: "Ingestion pipeline: parse, chunk with overlap, embed, store, retrieve, assemble context with citations, answer with Alpha.",
    status: "ready",
    notes: [
      "Retrieved text is wrapped as data with delimiters so a document cannot become an instruction.",
      "Answers carry source references with scores; an answer with no retrieval says so.",
      "Missing: PDF/HTML parsers — unsupported kinds fail loudly instead of being mis-parsed.",
    ],
  },
  {
    id: "memory",
    name: "Alpha Memory",
    summary: "Conversation, session and long-term memory with local relevance scoring, explicit approval and deletion.",
    status: "ready",
    notes: [
      "Long-term writes require the memory.write.long-term permission and explicit approval.",
      "Relevance = similarity x recency decay x importance x usage; the formula is in the code, not a service.",
      "Every memory can be listed, updated and deleted from the workspace.",
    ],
  },
  {
    id: "agents",
    name: "Alpha Agents",
    summary: "Planning, bounded reasoning loops, tool selection and execution, result verification, run history and sandboxing.",
    status: "in-development",
    notes: [
      "Deterministic capability-match planner works today; the model planner refuses to run on untrained weights.",
      "Final answers are either model-generated from tool results or labelled as a structured summary.",
      "Missing: parallel tool execution and multi-agent delegation.",
    ],
  },
  {
    id: "tools",
    name: "Alpha Tool Layer",
    summary: "Registration, discovery, JSON Schema validation, permissions, approval gates, execution records and error handling.",
    status: "ready",
    notes: [
      "Seven built-in tools: text stats, calculator, tokenizer analysis, corpus search, memory search, memory write, admin vector purge.",
      "Every call passes authorization, validation, rate limiting and audit before the handler runs.",
      "Failed calls return as values so an agent can react; permission failures throw.",
    ],
  },
  {
    id: "mcp",
    name: "Alpha MCP Layer",
    summary: "Model Context Protocol shapes, JSON-RPC 2.0 HTTP client, and adapters that register an MCP server's tools as Alpha tools.",
    status: "not-configured",
    notes: [
      "The client performs a real initialize handshake and real tools/list and tools/call requests.",
      "MCP tools inherit Alpha's permissions, sandbox rules and audit trail.",
      "Nothing is connected until an endpoint is configured; until then every MCP surface reads NOT CONFIGURED.",
    ],
  },
  {
    id: "vector",
    name: "Alpha Vector Store",
    summary: "Self-owned collections with vectors, metadata, namespaces, exact similarity search, updates, deletion and JSON persistence.",
    status: "ready",
    notes: [
      "Cosine, dot and euclidean metrics with deterministic tie-breaking.",
      "Exact scan, honestly O(records x dimension); an approximate index is not pretended.",
      "Snapshots import and export, so the store persists in Alpha's own tables.",
    ],
  },
  {
    id: "automation",
    name: "Alpha Automation",
    summary: "Workflows with manual/interval/event triggers, conditions, ordered actions, a job queue, retries and execution history.",
    status: "ready",
    notes: [
      "Conditions are evaluated against the trigger payload; skipped jobs record why.",
      "Jobs retry with exponential backoff and every attempt is recorded.",
      "Timers only run while the engine is started by the host.",
    ],
  },
  {
    id: "security",
    name: "Alpha Security",
    summary: "Roles and permissions, agent scopes, approval gates, input/output validation, prompt-injection defence, rate limits, hash-chained audit log, sandboxes.",
    status: "ready",
    notes: [
      "An agent cannot call a tool outside its allow-list, and a gated tool cannot run without a recorded approval.",
      "Injection detection is heuristic and labelled as such — it reduces risk, it does not prove safety.",
      "The audit log is hash-chained; tampering is detectable by replaying the chain.",
    ],
  },
  {
    id: "observability",
    name: "Alpha Observability",
    summary: "Traces with span IDs, counters/histograms with real percentiles, structured logs with redaction, and cross-module recording.",
    status: "ready",
    notes: [
      "Inference latency, token usage, stop reasons, training loss, tool durations, agent runs, RAG retrievals and workflow executions are all measured.",
      "Percentiles come from the sampled window, not from an assumption about the distribution.",
      "Secrets are redacted before anything is logged or stored.",
    ],
  },
  {
    id: "datasets",
    name: "Alpha Datasets",
    summary: "Corpus representation with provenance and licence, deterministic train/validation split, tokenisation and windowed batching.",
    status: "ready",
    notes: [
      "Ships with an original seed corpus written for this repository.",
      "Every dataset records its licence, so training data provenance is auditable.",
      "Bring your own documents; the pipeline does not care where the strings came from.",
    ],
  },
];

export function moduleById(id: string): AlphaModuleDescriptor | null {
  return ALPHA_MODULES.find((module) => module.id === id) ?? null;
}

export function moduleStatusCounts(): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const module of ALPHA_MODULES) {
    counts[module.status] = (counts[module.status] ?? 0) + 1;
  }
  return counts;
}
