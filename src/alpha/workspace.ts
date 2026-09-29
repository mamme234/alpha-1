/**
 * Alpha Workspace — the composition root.
 *
 * This is where the modules become a system: a config produces a tokenizer, a
 * tokenizer produces a model vocabulary, the model produces an inference engine
 * and embeddings, and those feed RAG, memory, tools, agents, automation,
 * security and observability. Nothing is stubbed: if a subsystem is missing or
 * unconfigured, the workspace says so rather than fabricating a result.
 *
 * The workspace is transport-agnostic. Persistence is an optional adapter, so
 * the same object runs in a browser tab (persisting to Alpha's own Convex
 * tables) or in a Node script (persisting to files).
 */

import { alphaId, newTraceId, type AlphaStatus } from "./core/types";
import type { AlphaModelStage } from "./core/types";
import { AlphaError, describeError } from "./core/errors";
import { AlphaConfig, createAlphaConfig, type AlphaConfigOverrides } from "./configs/alpha.config";
import { ALPHA_MODULES } from "./modules";
import { AlphaTransformer } from "./model/transformer";
import { countParameters, createModelArtifact, type AlphaModelArtifact } from "./model/config";
import { AlphaTokenizer, type AlphaTokenizerSnapshot } from "./tokenizer/bpe";
import { ALPHA_SEED_CORPUS } from "./datasets/seed-corpus";
import { datasetStats, type AlphaDataset } from "./datasets/types";
import { AlphaTrainer, type TrainingConfig, type TrainingEvent, type TrainingSummary } from "./training/trainer";
import type { AlphaCheckpoint, AlphaCheckpointSummary } from "./training/checkpoint";
import { summariseCheckpoint } from "./training/checkpoint";
import { AlphaInferenceEngine, type GenerationResult, type SamplingConfig } from "./inference/engine";
import { AlphaContextEngine, type AssembledContext } from "./context/engine";
import { AlphaEmbedder } from "./embeddings/embedder";
import { AlphaVectorStore, type VectorCollectionInfo, type VectorRecord } from "./vector/store";
import { AlphaRagPipeline, type IngestedDocument, type RagAnswer } from "./rag/pipeline";
import { AlphaMemoryStore, type MemoryRecord, type MemoryScope } from "./memory/store";
import { AlphaPolicyEngine } from "./security/policy";
import { AlphaRateLimiter, type RateLimitPolicy } from "./security/rate-limit";
import { AlphaAuditLog } from "./security/audit";
import { AlphaToolRegistry, type ToolDescriptor, type ToolExecutionRecord } from "./tools/registry";
import { registerBuiltinTools } from "./tools/builtin";
import { AlphaAgentRuntime } from "./agents/runtime";
import type { AgentRunResult } from "./agents/types";
import { AlphaAutomationEngine, type AlphaWorkflow, type JobExecution } from "./automation/engine";
import { AlphaObservability, type ObservabilitySnapshot } from "./observability/index";
import type { Span } from "./observability/tracer";
import type { AuditRecord } from "./security/audit";

export type PersistedRunInput = {
  kind: "inference" | "training" | "agent" | "rag" | "embedding" | "workflow";
  status: "succeeded" | "failed";
  traceId: string;
  input: string;
  output: string;
  metrics: Record<string, unknown>;
  modelStage: AlphaModelStage;
  error?: string | null;
};

/** Optional persistence boundary. Every method is best-effort and optional. */
export type AlphaPersistenceAdapter = {
  saveModel?: (artifact: AlphaModelArtifact) => void | Promise<void>;
  saveTokenizer?: (snapshot: AlphaTokenizerSnapshot) => void | Promise<void>;
  saveCheckpoint?: (checkpoint: AlphaCheckpoint) => void | Promise<void>;
  saveVectors?: (records: VectorRecord[]) => void | Promise<void>;
  saveMemories?: (records: MemoryRecord[], removedIds: string[]) => void | Promise<void>;
  saveSpans?: (spans: Span[]) => void | Promise<void>;
  saveAudit?: (records: AuditRecord[]) => void | Promise<void>;
  saveRun?: (run: PersistedRunInput) => void | Promise<void>;
  saveWorkflow?: (workflow: AlphaWorkflow) => void | Promise<void>;
  saveJob?: (job: JobExecution) => void | Promise<void>;
};

export type AlphaWorkspaceOptions = {
  config?: AlphaConfigOverrides;
  /** Dataset used to train the tokenizer and, by default, the model. */
  dataset?: AlphaDataset;
  /** Actor id of the signed-in user; becomes the workspace owner. */
  actorId?: string;
  agentActorId?: string;
  persistence?: AlphaPersistenceAdapter;
  /** Train the tokenizer during `initialise()` (default true). */
  prepareVocabulary?: boolean;
  logLevel?: "debug" | "info" | "warn" | "error";
};

export type AlphaWorkspaceSnapshot = {
  initialised: boolean;
  model: {
    name: string;
    version: string;
    stage: AlphaModelStage;
    parameterCount: number;
    config: AlphaTransformer["config"];
    trainedTokens: number;
    validationLoss: number | null;
    checkpointId: string | null;
  };
  tokenizer: {
    ready: boolean;
    version: string;
    vocabSize: number;
    trainedOn: string;
    documents: number;
    characters: number;
    mergeSteps: number;
  };
  corpus: {
    name: string;
    version: string;
    license: string;
    documents: number;
    characters: number;
    trainTokens: number;
    validationTokens: number;
  } | null;
  training: {
    running: boolean;
    step: number;
    totalSteps: number;
    lastSummary: TrainingSummary | null;
    checkpoint: AlphaCheckpointSummary | null;
    history: { step: number; loss: number; learningRate: number; gradNorm: number; elapsedMs: number }[];
  };
  inference: {
    requests: number;
    tokensGenerated: number;
    averageLatencyMs: number;
    p95LatencyMs: number;
    last: GenerationResult | null;
  };
  rag: {
    documents: IngestedDocument[];
    collections: VectorCollectionInfo[];
    vectors: number;
  };
  context: {
    contextLength: number;
    defaultReserveForOutput: number;
    last: AssembledContext | null;
  };
  memory: {
    counts: Record<MemoryScope, number>;
    entries: MemoryRecord[];
  };
  agents: {
    runs: number;
    completed: number;
    failed: number;
    averageSteps: number;
    averageDurationMs: number;
    recent: AgentRunResult[];
  };
  tools: {
    registered: ToolDescriptor[];
    stats: ReturnType<AlphaToolRegistry["stats"]>;
    recent: ToolExecutionRecord[];
  };
  automation: {
    workflows: AlphaWorkflow[];
    stats: ReturnType<AlphaAutomationEngine["stats"]>;
    history: JobExecution[];
  };
  security: {
    auditSize: number;
    chainIntact: boolean;
    policyEvents: ReturnType<AlphaPolicyEngine["recentEvents"]>;
    roles: { actorId: string; role: string | null }[];
  };
  observability: ObservabilitySnapshot;
  modules: typeof ALPHA_MODULES;
  statuses: Record<string, AlphaStatus>;
  errors: string[];
  takenAt: number;
};

export const ALPHA_OWNER_ACTOR = "alpha.owner";
export const ALPHA_AGENT_ACTOR = "alpha.agent";
export const ALPHA_SYSTEM_ACTOR = "alpha.system";

export class AlphaWorkspace {
  readonly config: AlphaConfig;
  readonly observability: AlphaObservability;
  readonly policy: AlphaPolicyEngine;
  readonly rateLimiter: AlphaRateLimiter;
  readonly audit: AlphaAuditLog;
  readonly vectorStore = new AlphaVectorStore();
  readonly tools: AlphaToolRegistry;
  readonly dataset: AlphaDataset;

  model: AlphaTransformer;
  tokenizer: AlphaTokenizer | null = null;
  embedder: AlphaEmbedder | null = null;
  inference: AlphaInferenceEngine | null = null;
  rag: AlphaRagPipeline | null = null;
  context: AlphaContextEngine | null = null;
  memory: AlphaMemoryStore | null = null;
  agents: AlphaAgentRuntime | null = null;
  automation: AlphaAutomationEngine;
  private trainer: AlphaTrainer | null = null;
  private checkpoint: AlphaCheckpoint | null = null;
  private lastSummary: TrainingSummary | null = null;
  private lastGeneration: GenerationResult | null = null;
  private readonly persistence: AlphaPersistenceAdapter | null;
  private readonly ownerActorId: string;
  private readonly agentActorId: string;
  private readonly seedDataset: AlphaDataset;
  private errors: string[] = [];
  private initialised = false;
  private training = false;
  private tokenizerSnapshot: AlphaTokenizerSnapshot | null = null;
  private lastContext: AssembledContext | null = null;
  private pendingSpans: Span[] = [];
  private flushedSpans = 0;

  constructor(options: AlphaWorkspaceOptions = {}) {
    this.config = createAlphaConfig(options.config);
    this.dataset = options.dataset ?? ALPHA_SEED_CORPUS;
    this.seedDataset = this.dataset;
    this.persistence = options.persistence ?? null;
    this.ownerActorId = options.actorId ?? ALPHA_OWNER_ACTOR;
    this.agentActorId = options.agentActorId ?? ALPHA_AGENT_ACTOR;

    this.observability = new AlphaObservability({
      logLevel: options.logLevel ?? this.config.observability.logLevel,
      sampleWindow: this.config.observability.sampleWindow,
      maxSpans: this.config.observability.maxSpans,
    });
    this.policy = new AlphaPolicyEngine({ approvalTtlMs: this.config.security.approvalTtlMs });
    this.rateLimiter = new AlphaRateLimiter(buildRateLimits(this.config));
    this.audit = new AlphaAuditLog();
    this.tools = new AlphaToolRegistry({
      policy: this.policy,
      rateLimiter: this.rateLimiter,
      audit: this.audit,
      onExecution: (record) => {
        this.observability.recordToolExecution(record);
      },
    });
    this.automation = new AlphaAutomationEngine({
      registry: this.tools,
      policy: this.policy,
      audit: this.audit,
      rateLimiter: this.rateLimiter,
      maxAttempts: this.config.automation.maxAttempts,
      baseRetryDelayMs: this.config.automation.baseRetryDelayMs,
      historyLimit: this.config.automation.historyLimit,
    });

    // Actors, roles and the agent scope, established before anything can run.
    this.policy.assignRole(this.ownerActorId, "owner");
    this.policy.assignRole(ALPHA_SYSTEM_ACTOR, "system");
    this.policy.assignRole(this.agentActorId, "agent");

    // The model exists as an architecture immediately; its stage is untrained.
    this.model = new AlphaTransformer({ ...this.config.model });
    this.observability.setModelStage("architecture", { reason: "weights are random initialisation" });
  }

  get ownerId(): string {
    return this.ownerActorId;
  }

  get agentId(): string {
    return this.agentActorId;
  }

  get isInitialised(): boolean {
    return this.initialised;
  }

  get isTraining(): boolean {
    return this.training;
  }

  get currentCheckpoint(): AlphaCheckpoint | null {
    return this.checkpoint;
  }

  get modelStage(): AlphaModelStage {
    return this.modelStageDerived();
  }

  /**
   * Stage derived from artefacts that actually exist: architecture before a
   * vocabulary is trained, untrained once the model is instantiable, and the
   * checkpoint's own stage after a real training run.
   */
  private modelStageDerived(): AlphaModelStage {
    if (!this.tokenizer) return "architecture";
    if (!this.checkpoint || this.checkpoint.step === 0) return "untrained";
    return this.checkpoint.stage;
  }

  /**
   * Train the tokenizer on the corpus, then rebuild the model so its vocabulary
   * exactly matches the learned vocabulary. No weights are downloaded.
   */
  async initialise(): Promise<AlphaWorkspace> {
    if (this.initialised) return this;

    const tokenizer = AlphaTokenizer.train(this.seedDataset.documents, {
      vocabSize: this.config.tokenizer.targetVocabSize,
      specialTokens: this.config.tokenizer.specialTokens,
      version: this.config.tokenizer.version,
      trainedOn: `${this.seedDataset.name}@${this.seedDataset.version}`,
    });
    this.tokenizer = tokenizer;
    this.tokenizerSnapshot = tokenizer.toJSON();

    const modelConfig = { ...this.config.model, vocabSize: Math.max(tokenizer.vocabSize, 64) };
    this.model = new AlphaTransformer(modelConfig);
    this.embedder = new AlphaEmbedder({
      model: this.model,
      tokenizer,
      stage: this.modelStageDerived(),
      pooling: "mean",
      maxTokens: Math.min(32, modelConfig.contextLength),
    });
    this.inference = new AlphaInferenceEngine({
      model: this.model,
      tokenizer,
      stage: this.modelStageDerived(),
    });
    this.rag = new AlphaRagPipeline({
      tokenizer,
      embedder: this.embedder,
      store: this.vectorStore,
      inference: this.inference,
      config: this.config.rag,
    });
    this.context = new AlphaContextEngine({
      tokenizer,
      contextLength: modelConfig.contextLength,
    });
    this.memory = new AlphaMemoryStore({ embedder: this.embedder, config: this.config.memory });

    // Register tools now that every service they may need exists.
    registerBuiltinTools(this.tools);
    this.tools.setServices({
      tokenizer,
      embedder: this.embedder,
      vectorStore: this.vectorStore,
      memory: this.memory,
      inference: this.inference,
    });

    this.policy.registerAgentScope({
      agentId: this.agentActorId,
      permissions: [
        "model.read",
        "inference.run",
        "vector.read",
        "rag.query",
        "memory.read",
        "memory.write",
        "tool.execute",
      ],
      // Deliberately excludes the destructive admin tool.
      allowedTools: [
        "alpha.text.stats",
        "alpha.calculator",
        "alpha.tokenizer.analyze",
        "alpha.corpus.search",
        "alpha.memory.search",
        "alpha.memory.write",
      ],
      maxSteps: this.config.security.sandbox.maxSteps,
    });

    this.agents = new AlphaAgentRuntime({
      registry: this.tools,
      policy: this.policy,
      inference: this.inference,
      memory: this.memory,
      audit: this.audit,
      rateLimiter: this.rateLimiter,
      onEvent: (event) => {
        this.observability.logger.log("debug", "agents", `event ${event.type}`, {}, null);
      },
    });

    this.observability.tracer.onSpanEnd((event) => {
      this.pendingSpans.push(event.span);
      if (this.pendingSpans.length > 200) {
        this.pendingSpans = this.pendingSpans.slice(-100);
      }
    });

    this.initialised = true;
    this.observability.setModelStage(this.modelStageDerived(), {
      reason: "tokenizer trained, model instantiated, no checkpoint yet",
    });
    this.observability.logger.log(
      "info",
      "workspace",
      `initialised ${modelConfig.name}: vocab ${tokenizer.vocabSize}, ${countParameters(modelConfig).toLocaleString()} parameters`,
      { corpus: this.seedDataset.name, documents: this.seedDataset.documents.length },
    );

    await this.persist("model");
    await this.persist("tokenizer");
    return this;
  }

  private requireReady(): { tokenizer: AlphaTokenizer; inference: AlphaInferenceEngine; embedder: AlphaEmbedder; rag: AlphaRagPipeline; memory: AlphaMemoryStore; agents: AlphaAgentRuntime } {
    if (!this.initialised || !this.tokenizer || !this.inference || !this.embedder || !this.rag || !this.memory || !this.agents) {
      throw new AlphaError("alpha.not_initialised", "core", "workspace must be initialised before use");
    }
    return {
      tokenizer: this.tokenizer,
      inference: this.inference,
      embedder: this.embedder,
      rag: this.rag,
      memory: this.memory,
      agents: this.agents,
    };
  }

  /**
   * Compose the prompt through the context engine so the instruction, recent
   * memory and the user's question share the model's window deliberately, and
   * the trimming that happened is recorded rather than invisible.
   */
  private composePrompt(prompt: string, sampling: Partial<SamplingConfig>): string {
    if (!this.context) return prompt;
    const memory = this.memory?.retrieve(prompt, { topK: 3, minRelevance: 0.05 }) ?? [];
    const assembled = this.context.assembleConversation({
      prompt,
      instruction:
        "You are Alpha, a self-owned language model trained by this repository. Answer from what you know or from the data provided. If you do not know, say so.",
      memory: memory.length ? this.memory?.buildContext(memory) : undefined,
      // Keep room for the answer without starving the prompt: the engine clamps
      // this to the window and the prompt is budgeted ahead of everything else.
      reserveForOutput: Math.min(
        sampling.maxNewTokens ?? 40,
        Math.floor((this.context.contextLength || 1) / 3),
      ),
      maxConversationTurns: 4,
    });
    this.lastContext = assembled;
    if (assembled.truncated) {
      this.observability.logger.log(
        "debug",
        "context",
        `context assembled with ${assembled.usedTokens}/${assembled.budgetTokens} tokens; ${assembled.notes.length} adjustment(s)`,
        { notes: assembled.notes },
      );
    }
    return assembled.text;
  }

  private refreshQuality(): void {
    const stage = this.modelStageDerived();
    if (this.inference) this.inference = new AlphaInferenceEngine({ model: this.model, tokenizer: this.tokenizer!, stage });
    if (this.embedder) {
      this.embedder = new AlphaEmbedder({
        model: this.model,
        tokenizer: this.tokenizer!,
        stage,
        pooling: "mean",
        maxTokens: this.embedder.maxTokens,
      });
    }
    if (this.rag) this.rag = new AlphaRagPipeline({
      tokenizer: this.tokenizer!,
      embedder: this.embedder!,
      store: this.vectorStore,
      inference: this.inference!,
      config: this.config.rag,
    });
    if (this.memory) this.memory = new AlphaMemoryStore({ embedder: this.embedder!, config: this.config.memory });
    if (this.tokenizer) {
      this.tools.setServices({ embedder: this.embedder, memory: this.memory, inference: this.inference });
    }
    if (this.agents) {
      this.agents = new AlphaAgentRuntime({
        registry: this.tools,
        policy: this.policy,
        inference: this.inference ?? undefined,
        memory: this.memory ?? undefined,
        audit: this.audit,
        rateLimiter: this.rateLimiter,
      });
    }
    this.observability.setModelStage(stage);
  }

  /**
   * Train Alpha. Yields the real training events so a UI can stream progress;
   * persist a checkpoint every time the trainer emits one.
   */
  async *train(options: Partial<TrainingConfig> = {}): AsyncGenerator<TrainingEvent, TrainingSummary> {
    const { tokenizer } = this.requireReady();
    if (this.training) {
      throw new AlphaError("alpha.training_in_progress", "training", "a training run is already in progress");
    }
    this.training = true;
    const traceId = newTraceId();
    const span = this.observability.startSpan({
      name: "training.run",
      kind: "training",
      module: "training",
      traceId,
      attributes: { dataset: this.dataset.name, license: this.dataset.license },
    });

    try {
      const trainer = new AlphaTrainer({
        model: this.model,
        tokenizer,
        dataset: this.dataset,
        config: { ...this.config.training, ...options },
        checkpointLabel: `${this.model.config.name}-seed`,
        isFineTune: Boolean(this.checkpoint),
      });
      this.trainer = trainer;
      const iterator = trainer.run();
      let next = iterator.next();
      while (!next.done) {
        const event = next.value;
        if (event.type === "step") {
          this.observability.recordTrainingStep(event.point, traceId);
        }
        if (event.type === "eval") {
          this.observability.logger.log(
            "info",
            "training",
            `validation loss ${event.evaluation.loss.toFixed(4)} (perplexity ${event.evaluation.perplexity.toFixed(2)}) at step ${event.evaluation.step}`,
            { uniformBaseline: event.evaluation.uniformLoss },
            traceId,
          );
        }
        if (event.type === "checkpoint") {
          this.checkpoint = event.checkpoint;
          this.refreshQuality();
          await this.persist("checkpoint");
          this.observability.setModelStage(event.checkpoint.stage, {
            step: event.checkpoint.step,
            tokensSeen: event.checkpoint.tokensSeen,
            validationLoss: event.checkpoint.metrics.validationLoss,
          });
          await this.persistence?.saveRun?.({
            kind: "training",
            status: "succeeded",
            traceId,
            input: `train ${event.checkpoint.step} steps on ${this.dataset.name}`,
            output: `checkpoint ${event.checkpoint.id}`,
            metrics: {
              step: event.checkpoint.step,
              trainLoss: event.checkpoint.metrics.trainLoss,
              validationLoss: event.checkpoint.metrics.validationLoss,
              tokensSeen: event.checkpoint.tokensSeen,
              sizeBytes: event.checkpoint.sizeBytes,
              parameterCount: this.model.parameterCount,
            },
            modelStage: this.modelStageDerived(),
          });
        }
        yield event;
        next = iterator.next();
      }
      const summary = iterator.next().value as TrainingSummary | undefined;
      const resolved: TrainingSummary =
        summary ??
        ({
          steps: trainer.step,
          tokensSeen: trainer.tokensSeen,
          firstLoss: null,
          lastLoss: null,
          bestLoss: null,
          validationLoss: null,
          validationPerplexity: null,
          uniformLossBaseline: trainer.uniformLoss,
          durationMs: 0,
          throughputTokensPerSecond: 0,
          checkpoint: this.checkpoint,
        } satisfies TrainingSummary);
      this.lastSummary = resolved;
      this.observability.endSpan(span, {
        attributes: {
          steps: resolved.steps,
          firstLoss: resolved.firstLoss,
          lastLoss: resolved.lastLoss,
          validationLoss: resolved.validationLoss,
        },
      });
      return resolved;
    } catch (error) {
      const described = describeError(error);
      this.errors.push(described.message);
      this.observability.recordError("training", described.message);
      this.observability.endSpan(span, { status: "error", error: described.message });
      throw error;
    } finally {
      this.training = false;
    }
  }

  /**
   * Restore Alpha from a stored checkpoint and continue from its step.
   * The vocabulary must match, because weights loaded into a different
   * architecture would silently mean nothing.
   */
  resumeFrom(checkpoint: AlphaCheckpoint): void {
    const { tokenizer } = this.requireReady();
    if (checkpoint.config.vocabSize !== this.model.config.vocabSize) {
      throw new AlphaError(
        "alpha.checkpoint_mismatch",
        "training",
        `checkpoint vocabulary (${checkpoint.config.vocabSize}) does not match the current model (${this.model.config.vocabSize})`,
      );
    }
    this.model.loadWeights(checkpoint.weights);
    this.checkpoint = checkpoint;
    const trainer = new AlphaTrainer({
      model: this.model,
      tokenizer,
      dataset: this.dataset,
      config: this.config.training,
      checkpointLabel: checkpoint.label,
      isFineTune: true,
    });
    trainer.resumeFrom(checkpoint);
    this.trainer = trainer;
    this.refreshQuality();
    this.observability.setModelStage(checkpoint.stage, {
      reason: "resumed from a stored checkpoint",
      step: checkpoint.step,
      tokensSeen: checkpoint.tokensSeen,
    });
    this.observability.logger.log(
      "info",
      "workspace",
      `resumed ${checkpoint.modelName} at step ${checkpoint.step}`,
      { validationLoss: checkpoint.metrics.validationLoss },
    );
  }

  /** One-shot generation through Alpha's own model. */
  generate(prompt: string, sampling: Partial<SamplingConfig> = {}): GenerationResult {
    const { inference } = this.requireReady();
    const traceId = newTraceId();
    const span = this.observability.startSpan({
      name: "inference.generate",
      kind: "inference",
      module: "inference",
      traceId,
      attributes: { promptTokensHint: prompt.length },
    });
    try {
      const result = inference.generate(this.composePrompt(prompt, sampling), sampling);
      this.lastGeneration = result;
      this.observability.recordInference(result, traceId);
      this.observability.endSpan(span, {
        attributes: {
          generatedTokens: result.generatedTokens,
          stopReason: result.stopReason,
          tokensPerSecond: result.tokensPerSecond,
        },
      });
      void this.persistence?.saveRun?.({
        kind: "inference",
        status: "succeeded",
        traceId,
        input: prompt.slice(0, 2000),
        output: result.text.slice(0, 4000),
        metrics: {
          generatedTokens: result.generatedTokens,
          promptTokens: result.promptTokens,
          latencyMs: result.latencyMs,
          tokensPerSecond: result.tokensPerSecond,
          stopReason: result.stopReason,
        },
        modelStage: result.modelStage,
      });
      return result;
    } catch (error) {
      const described = describeError(error);
      this.observability.recordError("inference", described.message);
      this.observability.endSpan(span, { status: "error", error: described.message });
      this.errors.push(described.message);
      throw error;
    }
  }

  async *streamGenerate(
    prompt: string,
    sampling: Partial<SamplingConfig> = {},
  ): AsyncGenerator<{ text: string; done: boolean }, GenerationResult | null, void> {
    const { inference } = this.requireReady();
    const traceId = newTraceId();
    const span = this.observability.startSpan({
      name: "inference.stream",
      kind: "inference",
      module: "inference",
      traceId,
    });
    const stream = inference.generateStream(this.composePrompt(prompt, sampling), sampling);
    let next = await stream.next();
    while (!next.done) {
      yield { text: next.value.text, done: next.value.done };
      next = await stream.next();
    }
    const result = next.value;
    this.lastGeneration = result;
    this.observability.recordInference(result, traceId);
    this.observability.endSpan(span, { attributes: { generatedTokens: result.generatedTokens } });
    void this.persistence?.saveRun?.({
      kind: "inference",
      status: "succeeded",
      traceId,
      input: prompt.slice(0, 2000),
      output: result.text.slice(0, 4000),
      metrics: {
        generatedTokens: result.generatedTokens,
        latencyMs: result.latencyMs,
        stopReason: result.stopReason,
      },
      modelStage: result.modelStage,
    });
    return result;
  }

  async ingestDocument(input: {
    title: string;
    content: string;
    kind?: string;
    license?: string;
    metadata?: Record<string, string>;
  }): Promise<IngestedDocument> {
    const { rag } = this.requireReady();
    const span = this.observability.startSpan({ name: "rag.ingest", kind: "retrieval", module: "rag" });
    const { document } = rag.ingest(input);
    this.observability.recordRagIngestion(document);
    this.observability.recordEmbedding(document.chunks, this.vectorStore.getCollection(rag.collectionName)?.dimension ?? 0);
    this.observability.endSpan(span, { attributes: { chunks: document.chunks, tokens: document.tokens } });
    await this.persist("vectors");
    return document;
  }

  /** Remove a document and every vector derived from it. */
  removeDocument(documentId: string): number {
    const { rag } = this.requireReady();
    const removed = rag.removeDocument(documentId);
    this.audit.append({
      actor: this.ownerActorId,
      module: "rag",
      action: "document.remove",
      resource: documentId,
      decision: "allow",
      reason: `removed ${removed} chunk vector(s)`,
    });
    void this.persist("vectors");
    return removed;
  }

  ask(question: string, sampling: Partial<SamplingConfig> = {}): RagAnswer {
    const { rag } = this.requireReady();
    const traceId = newTraceId();
    const span = this.observability.startSpan({ name: "rag.answer", kind: "retrieval", module: "rag", traceId });
    const answer = rag.answer(question, sampling);
    this.lastGeneration = answer.generation;
    this.observability.recordRagRetrieval(answer, traceId);
    this.observability.recordInference(answer.generation, traceId);
    this.observability.endSpan(span, { attributes: { retrieved: answer.retrieved } });
    void this.persistence?.saveRun?.({
      kind: "rag",
      status: "succeeded",
      traceId,
      input: question.slice(0, 2000),
      output: answer.answer.slice(0, 4000),
      metrics: {
        retrieved: answer.retrieved,
        contextTokens: answer.contextTokens,
        sources: answer.sources.map((source) => source.title),
      },
      modelStage: answer.modelStage,
    });
    return answer;
  }

  writeMemory(input: {
    scope: MemoryScope;
    key: string;
    content: string;
    sessionId?: string | null;
    importance?: number;
    approved?: boolean;
  }): MemoryRecord {
    const { memory } = this.requireReady();
    if (input.scope === "long-term" && input.approved !== true) {
      this.observability.recordPolicyDenial(this.ownerActorId, "memory.write.long-term", "approval missing");
      throw new AlphaError("alpha.permission_denied", "memory", "long-term memory requires explicit approval");
    }
    const record = memory.write({ ...input, source: "user", approved: input.approved });
    this.observability.recordMemoryWrite(record.scope, record.key);
    void this.persist("memories");
    return record;
  }

  forgetMemory(id: string): boolean {
    const { memory } = this.requireReady();
    const removed = memory.forget(id);
    if (removed) {
      this.audit.append({
        actor: this.ownerActorId,
        module: "memory",
        action: "memory.delete",
        resource: id,
        decision: "allow",
        reason: "user deleted a memory",
      });
      void this.persist("memories");
    }
    return removed;
  }

  recall(query: string, options: { scopes?: MemoryScope[]; topK?: number; sessionId?: string | null } = {}) {
    const { memory } = this.requireReady();
    const entries = memory.retrieve(query, options);
    this.observability.recordMemoryRetrieval(entries);
    return entries;
  }

  approveTool(toolName: string): string {
    const approval = this.policy.approveTool({
      actorId: this.agentActorId,
      toolName,
      grantedBy: this.ownerActorId,
    });
    this.audit.append({
      actor: this.ownerActorId,
      module: "security",
      action: "tool.approve",
      resource: toolName,
      decision: "allow",
      reason: `approval granted until ${new Date(approval.expiresAt).toISOString()}`,
      data: { approvalId: approval.id },
    });
    return approval.id;
  }

  async runAgent(goal: string, options: { maxSteps?: number; preferModelPlanner?: boolean; sessionId?: string } = {}): Promise<AgentRunResult> {
    const { agents } = this.requireReady();
    const iterator = agents.run({
      goal,
      actorId: this.agentActorId,
      maxSteps: options.maxSteps,
      preferModelPlanner: options.preferModelPlanner,
      sessionId: options.sessionId ?? `session_${this.ownerActorId}`,
    });
    let next = await iterator.next();
    let result: AgentRunResult | null = null;
    while (!next.done) {
      if (next.value.type === "done") result = next.value.result;
      next = await iterator.next();
    }
    if (!result) {
      throw new AlphaError("alpha.agent_no_result", "agents", "agent run produced no result");
    }
    this.observability.recordAgentRun(result);
    await this.runScheduledWorkflows();
    return result;
  }

  registerWorkflow(input: Omit<AlphaWorkflow, "id" | "createdAt" | "updatedAt" | "actorId"> & { actorId?: string }): AlphaWorkflow {
    const workflow = this.automation.registerWorkflow({ ...input, actorId: input.actorId ?? this.ownerActorId });
    void this.persistence?.saveWorkflow?.(workflow);
    return workflow;
  }

  async runWorkflow(workflowId: string, payload: Record<string, unknown> = {}): Promise<JobExecution> {
    const job = await this.automation.run(workflowId, payload);
    this.observability.recordWorkflowExecution(job);
    await this.persistence?.saveJob?.(job);
    return job;
  }

  private async runScheduledWorkflows(): Promise<void> {
    const pending = this.automation.pendingJobs();
    if (pending.length === 0) return;
    const processed = await this.automation.drain();
    for (const job of processed) {
      this.observability.recordWorkflowExecution(job);
      await this.persistence?.saveJob?.(job);
    }
  }

  /** Push the current state to the persistence adapter. */
  async persist(scope: "model" | "tokenizer" | "checkpoint" | "vectors" | "memories" | "spans" | "audit" | "all" = "all"): Promise<void> {
    if (!this.persistence) return;
    try {
      if (scope === "model" || scope === "all") {
        const artifact = this.currentArtifact();
        await this.persistence.saveModel?.(artifact);
      }
      if ((scope === "tokenizer" || scope === "all") && this.tokenizerSnapshot) {
        await this.persistence.saveTokenizer?.(this.tokenizerSnapshot);
      }
      if ((scope === "checkpoint" || scope === "all") && this.checkpoint) {
        await this.persistence.saveCheckpoint?.(this.checkpoint);
      }
      if (scope === "vectors" || scope === "all") {
        await this.persistence.saveVectors?.(this.vectorStore.list());
      }
      if (scope === "memories" || scope === "all") {
        await this.persistence.saveMemories?.(this.memory ? this.memory.list() : [], []);
      }
      if (scope === "spans" || scope === "all") {
        const fresh = this.pendingSpans.length > this.flushedSpans;
        if (fresh) {
          await this.persistence.saveSpans?.(this.pendingSpans);
          this.flushedSpans = this.pendingSpans.length;
        }
      }
      if (scope === "audit" || scope === "all") {
        const records = this.audit.list({ limit: 50 });
        await this.persistence.saveAudit?.(records);
      }
    } catch (error) {
      const described = describeError(error);
      this.errors.push(`persistence: ${described.message}`);
      this.observability.recordError("workspace", `persistence failed: ${described.message}`);
    }
  }

  private currentArtifact(): AlphaModelArtifact {
    const artifact = createModelArtifact(this.model.config, alphaId("model"));
    return {
      ...artifact,
      stage: this.modelStageDerived(),
      parameterCount: this.model.parameterCount,
      checkpointId: this.checkpoint?.id,
      trainedTokens: this.checkpoint?.tokensSeen,
      validationLoss: this.checkpoint?.metrics.validationLoss ?? undefined,
      notes: this.checkpoint
        ? [
            `Trained from scratch by Alpha's training engine (${this.checkpoint.step} steps).`,
            `Corpus: ${this.dataset.name} (${this.dataset.license}).`,
            "No external model weights were used.",
          ]
        : artifact.notes,
    };
  }

  /** Full, real status for the workspace UI. */
  snapshot(): AlphaWorkspaceSnapshot {
    const tokenizer = this.tokenizer;
    const corpus = this.trainer
      ? {
          name: this.dataset.name,
          version: this.dataset.version,
          license: this.dataset.license,
          documents: datasetStats(this.dataset).documents,
          characters: datasetStats(this.dataset).characters,
          trainTokens: this.trainer.corpus.stats.trainTokens,
          validationTokens: this.trainer.corpus.stats.validationTokens,
        }
      : null;
    const statuses: Record<string, AlphaStatus> = {};
    for (const module of ALPHA_MODULES) statuses[module.id] = module.status;
    if (this.initialised && this.modelStageDerived() === "trained") {
      statuses.model = "ready";
      statuses.training = "ready";
    } else if (this.initialised) {
      statuses.model = "untrained";
    }
    const inferenceMetrics = this.observability.metrics;
    const latency = inferenceMetrics.histogram("alpha.inference.latency_ms");

    return {
      initialised: this.initialised,
      model: {
        name: this.model.config.name,
        version: this.model.config.version,
        stage: this.modelStageDerived(),
        parameterCount: this.model.parameterCount,
        config: this.model.config,
        trainedTokens: this.checkpoint?.tokensSeen ?? 0,
        validationLoss: this.checkpoint?.metrics.validationLoss ?? null,
        checkpointId: this.checkpoint?.id ?? null,
      },
      tokenizer: {
        ready: Boolean(tokenizer),
        version: tokenizer?.version ?? this.config.tokenizer.version,
        vocabSize: tokenizer?.vocabSize ?? 0,
        trainedOn: tokenizer?.trainedOn ?? "not trained yet",
        documents: tokenizer?.stats.documents ?? 0,
        characters: tokenizer?.stats.characters ?? 0,
        mergeSteps: tokenizer?.stats.mergeSteps ?? 0,
      },
      corpus,
      training: {
        running: this.training,
        step: this.trainer?.step ?? 0,
        totalSteps: this.trainer?.config.totalSteps ?? this.config.training.totalSteps,
        lastSummary: this.lastSummary,
        checkpoint: this.checkpoint ? summariseCheckpoint(this.checkpoint) : null,
        history: (this.trainer?.history ?? []).map((point) => ({
          step: point.step,
          loss: point.loss,
          learningRate: point.learningRate,
          gradNorm: point.gradNorm,
          elapsedMs: point.elapsedMs,
        })),
      },
      inference: {
        requests: inferenceMetrics.value("alpha.inference.requests"),
        tokensGenerated: inferenceMetrics.value("alpha.inference.tokens_generated"),
        averageLatencyMs: latency?.average ?? 0,
        p95LatencyMs: latency?.p95 ?? 0,
        last: this.lastGeneration,
      },
      rag: {
        documents: this.rag?.listDocuments() ?? [],
        collections: this.vectorStore.listCollections(),
        vectors: this.vectorStore.count(),
      },
      context: {
        contextLength: this.context?.contextLength ?? this.model.config.contextLength,
        defaultReserveForOutput:
          this.context?.describe().defaultReserveForOutput ?? Math.floor(this.model.config.contextLength / 4),
        last: this.lastContext,
      },
      memory: {
        counts: this.memory?.stats() ?? { conversation: 0, session: 0, "long-term": 0 },
        entries: this.memory?.list() ?? [],
      },
      agents: {
        ...(this.agents?.stats() ?? { runs: 0, completed: 0, failed: 0, averageSteps: 0, averageDurationMs: 0 }),
        recent: this.agents?.recentRuns(5) ?? [],
      },
      tools: {
        registered: this.tools.describe(),
        stats: this.tools.stats(),
        recent: this.tools.recentExecutions(10),
      },
      automation: {
        workflows: this.automation.listWorkflows(),
        stats: this.automation.stats(),
        history: this.automation.executionHistory(10),
      },
      security: {
        auditSize: this.audit.size,
        chainIntact: this.audit.verifyChain().intact,
        policyEvents: this.policy.recentEvents(20),
        roles: [
          { actorId: this.ownerActorId, role: this.policy.roleOf(this.ownerActorId) },
          { actorId: this.agentActorId, role: this.policy.roleOf(this.agentActorId) },
          { actorId: ALPHA_SYSTEM_ACTOR, role: this.policy.roleOf(ALPHA_SYSTEM_ACTOR) },
        ],
      },
      observability: this.observability.snapshot(),
      modules: ALPHA_MODULES,
      statuses,
      errors: [...this.errors],
      takenAt: Date.now(),
    };
  }
}

function buildRateLimits(config: AlphaConfig): Record<string, RateLimitPolicy> {
  return {
    inference: { capacity: config.security.rateLimits.inference, refillPerSecond: config.security.rateLimits.inference / 60 },
    tool: { capacity: config.security.rateLimits.tool, refillPerSecond: config.security.rateLimits.tool / 60 },
    "tool.execute.dangerous": { capacity: 4, refillPerSecond: 0.05 },
    agent: { capacity: config.security.rateLimits.agent, refillPerSecond: config.security.rateLimits.agent / 60 },
    workflow: { capacity: config.security.rateLimits.workflow, refillPerSecond: config.security.rateLimits.workflow / 60 },
    embedding: { capacity: 200, refillPerSecond: 10 },
  };
}
