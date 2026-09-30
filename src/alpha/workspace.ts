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
import {
  ALPHA_RESOURCE_LIMITS,
  estimateTrainingMemory,
  type TrainingMemoryEstimate,
} from "./core/limits";
import { AlphaConfig, createAlphaConfig, type AlphaConfigOverrides } from "./configs/alpha.config";
import { ALPHA_MODULES } from "./modules";
import { AlphaTransformer } from "./model/transformer";
import {
  countParameters,
  createModelArtifact,
  modelConfigFingerprint,
  withSpecialTokenIds,
  type AlphaModelArtifact,
} from "./model/config";
import { AlphaTokenizer, type AlphaTokenizerSnapshot } from "./tokenizer/bpe";
import { ALPHA_SEED_CORPUS } from "./datasets/seed-corpus";
import {
  assertValidDataset,
  datasetFingerprint,
  datasetStats,
  type AlphaDataset,
} from "./datasets/types";
import { corpusReport, type CorpusReport } from "./datasets/corpus";
import { AlphaTrainer, type TrainingConfig, type TrainingEvent, type TrainingSummary } from "./training/trainer";
import {
  assertCheckpointCompatible,
  summariseCheckpoint,
  type AlphaCheckpoint,
  type AlphaCheckpointSummary,
} from "./training/checkpoint";
import {
  createTrainingJob,
  recordJobCheckpoint,
  recordJobEvaluation,
  recordJobStep,
  summariseJob,
  transitionJob,
  type AlphaTrainingJob,
} from "./training/job";
import { verifyAlphaModel, type VerificationReport } from "./training/verify";
import { AlphaInferenceEngine, type GenerationResult, type SamplingConfig } from "./inference/engine";
import { AlphaInferenceService, type AlphaModelDescriptor } from "./inference/service";
import { AlphaContextEngine, type AssembledContext } from "./context/engine";
import { AlphaEmbedder } from "./embeddings/embedder";
import { AlphaVectorStore, type VectorCollectionInfo, type VectorRecord } from "./vector/store";
import { AlphaRagPipeline, type IngestedDocument, type RagAnswer } from "./rag/pipeline";
import { AlphaMemoryStore, type MemoryProvenance, type MemoryRecord, type MemoryScope } from "./memory/store";
import { AlphaPolicyEngine } from "./security/policy";
import { AlphaRateLimiter, type RateLimitPolicy } from "./security/rate-limit";
import { AlphaAuditLog } from "./security/audit";
import { AlphaToolRegistry, type ToolDescriptor, type ToolExecutionRecord } from "./tools/registry";
import { registerBuiltinTools } from "./tools/builtin";
import { AlphaAgentRuntime } from "./agents/runtime";
import { AlphaAiRuntime, type RespondRequest, type RespondResult } from "./runtime/orchestrator";
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
  /** The training run record: id, state, references, metrics, checkpoints. */
  saveTrainingJob?: (job: AlphaTrainingJob) => void | Promise<void>;
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
    configFingerprint: string;
    trainedTokens: number;
    validationLoss: number | null;
    checkpointId: string | null;
  };
  /** Models registered with the inference service, with their honest stages. */
  models: AlphaModelDescriptor[];
  /** Resource ceilings and a rough memory estimate for the configured run. */
  resources: {
    limits: typeof ALPHA_RESOURCE_LIMITS;
    estimate: TrainingMemoryEstimate;
  };
  /** The corpus as the trainer will actually consume it. */
  corpusReport: CorpusReport | null;
  /** Result of the last A–I verification run, if one has been executed. */
  verification: VerificationReport | null;
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
    /** The run record: id, state, seed, references, metrics, checkpoints. */
    job: AlphaTrainingJob | null;
    jobSummary: string | null;
    /** Pause/stop requests take effect at the next optimiser step boundary. */
    pauseRequested: boolean;
    stopRequested: boolean;
  };
  inference: {
    requests: number;
    tokensGenerated: number;
    averageLatencyMs: number;
    p95LatencyMs: number;
    last: GenerationResult | null;
    useCache: boolean;
    cache: {
      used: boolean;
      positions: number;
      capacity: number;
      bytes: number;
      writes: number;
      hits: number;
      occupancy: number;
    };
  };
  /** The orchestration layer and the authoritative embedding config. */
  aiRuntime: {
    available: boolean;
    embedding: {
      config: {
        version: string;
        model: string;
        modelVersion: string;
        pooling: string;
        dimension: number;
        maxTokens: number;
        normalized: true;
      };
      modelStage: string;
      parameterCount: number;
    } | null;
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
  /**
   * The transport-neutral generation interface. The workspace registers its
   * current model here and everything that generates (workspace, RAG, agents,
   * tools) goes through it, so there is exactly one place where weights are
   * turned into tokens.
   */
  readonly inferenceService = new AlphaInferenceService();

  /**
   * The orchestration layer. `respond()` is the one entry point a conversation
   * needs: it decides between plain inference, memory, retrieval, a tool and an
   * agent run, and reports exactly what it did. Built during initialisation.
   */
  aiRuntime: AlphaAiRuntime | null = null;

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
  /** The current training run record, if a run has been started in this session. */
  private job: AlphaTrainingJob | null = null;
  private pauseRequested = false;
  private stopRequested = false;
  /** True when a trainer is waiting to continue from a stored checkpoint. */
  private pendingResume = false;
  private verification: VerificationReport | null = null;
  private tokenizerSnapshot: AlphaTokenizerSnapshot | null = null;
  private lastContext: AssembledContext | null = null;
  private pendingSpans: Span[] = [];
  private flushedSpans = 0;

  constructor(options: AlphaWorkspaceOptions = {}) {
    this.config = createAlphaConfig(options.config);
    this.dataset = options.dataset ?? ALPHA_SEED_CORPUS;
    // A malformed corpus is refused up front, not quietly reduced to whatever
    // happened to be encodable.
    assertValidDataset(this.dataset);
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

  /** The current training run record, if this session has started one. */
  get trainingJob(): AlphaTrainingJob | null {
    return this.job;
  }

  /** Result of the last A–I verification run, if one has been executed here. */
  get lastVerification(): VerificationReport | null {
    return this.verification;
  }

  /**
   * Publish the current weights to the inference service under a stable id, so
   * `generate({ modelId, prompt, generationConfig })` addresses exactly the
   * model the workspace is holding — including its stage.
   */
  private registerModelHandle(): void {
    const tokenizer = this.tokenizer;
    if (!tokenizer) return;
    const fingerprint = tokenizer.fingerprint();
    this.inferenceService.registerModel({
      id: AlphaInferenceService.modelId(this.model.config.name, this.model.config.version, fingerprint),
      name: this.model.config.name,
      version: this.model.config.version,
      stage: this.modelStageDerived(),
      model: this.model,
      tokenizer,
      contextLength: this.model.config.contextLength,
      checkpointId: this.checkpoint?.id ?? null,
    });
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

    // The model vocabulary is bound to the trained tokenizer: the size it
    // actually produced plus the special-token ids it reserves.
    const modelConfig = withSpecialTokenIds(
      { ...this.config.model, vocabSize: Math.max(tokenizer.vocabSize, 64) },
      tokenizer.specialTokenIds,
    );
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
    this.registerModelHandle();
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

    this.aiRuntime = new AlphaAiRuntime({
      inference: this.inference!,
      context: this.context!,
      memory: this.memory!,
      rag: this.rag!,
      tools: this.tools,
      agents: this.agents,
      policy: this.policy,
      rateLimiter: this.rateLimiter,
      audit: this.audit,
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
    // The orchestrator holds references to the pieces above, so it is rebuilt
    // whenever the model, memory or agent runtime is replaced.
    if (this.inference && this.context && this.memory && this.rag) {
      this.aiRuntime = new AlphaAiRuntime({
        inference: this.inference,
        context: this.context,
        memory: this.memory,
        rag: this.rag,
        tools: this.tools,
        agents: this.agents ?? undefined,
        policy: this.policy,
        rateLimiter: this.rateLimiter,
        audit: this.audit,
      });
    }
    // The inference service must never serve a stale stage or stale weights.
    this.registerModelHandle();
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
    // Continuing a run reuses the live trainer, so the optimiser moments, the
    // RNG position and the history all carry over. Two cases continue a run:
    // a paused job, and a run resumed from a stored checkpoint that still has
    // steps to take. Anything else is a new run over the current weights.
    const pausedJob = this.job !== null && this.job.state === "paused";
    const pendingResume =
      this.trainer !== null && this.pendingResume && this.trainer.step < this.trainer.config.totalSteps;
    const continuing = this.trainer !== null && (pausedJob || pendingResume);

    if (pausedJob && this.job) {
      this.job = transitionJob(this.job, "running");
    } else {
      if (!continuing) {
        // The run id is chosen here and given to both the trainer and the job,
        // so a checkpoint, a log line and a job record always agree on which
        // run they belong to.
        const runId = alphaId("run");
        this.trainer = new AlphaTrainer({
          model: this.model,
          tokenizer,
          dataset: this.dataset,
          config: { ...this.config.training, ...options },
          checkpointLabel: `${this.model.config.name}-${this.checkpoint ? "finetune" : "scratch"}`,
          isFineTune: Boolean(this.checkpoint),
          runId,
        });
      }
      this.pendingResume = false;
      const started = this.trainer!;
      const job = createTrainingJob({
        modelName: this.model.config.name,
        modelVersion: this.model.config.version,
        tokenizerVersion: tokenizer.version,
        tokenizerFingerprint: tokenizer.fingerprint(),
        datasetName: this.dataset.name,
        datasetVersion: this.dataset.version,
        datasetFingerprint: datasetFingerprint(this.dataset),
        datasetLicense: this.dataset.license,
        corpusTokens: started.corpus.stats.totalTokens,
        corpusDocuments: started.corpus.stats.documents,
        config: started.config,
        seed: started.config.seed,
        resumedFromCheckpointId: this.checkpoint?.id ?? null,
        id: started.runId,
      });
      this.job = transitionJob(job, "running");
    }

    const trainer = this.trainer!;
    let job = this.job!;
    this.pauseRequested = false;
    this.stopRequested = false;
    this.training = true;
    const traceId = newTraceId();
    const span = this.observability.startSpan({
      name: "training.run",
      kind: "training",
      module: "training",
      traceId,
      attributes: {
        runId: job.id,
        dataset: this.dataset.name,
        datasetFingerprint: job.datasetFingerprint,
        license: this.dataset.license,
        seed: job.seed,
        resumedFrom: job.resumedFromCheckpointId,
      },
    });

    try {
      await this.persistence?.saveTrainingJob?.(job);
      yield { type: "job", job };
      const iterator = trainer.run({
        shouldStop: () => {
          if (this.stopRequested) return "stop";
          if (this.pauseRequested) return "pause";
          return null;
        },
      });
      let next = iterator.next();
      while (!next.done) {
        const event = next.value;
        if (event.type === "step") {
          job = recordJobStep(job, event.point);
          this.job = job;
          this.observability.recordTrainingStep(event.point, traceId);
        }
        if (event.type === "eval") {
          job = recordJobEvaluation(job, event.evaluation);
          this.job = job;
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
          job = recordJobCheckpoint(job, event.checkpoint);
          this.job = job;
          await this.persistence?.saveTrainingJob?.(job);
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
      const resolved = next.value;
      if (!resolved) {
        throw new AlphaError(
          "alpha.training_no_summary",
          "training",
          "the training run ended without producing a summary",
        );
      }
      this.lastSummary = resolved;
      // The trainer reports how the run ended; the job record mirrors it, and a
      // paused run stays resumable instead of being marked finished.
      if (resolved.state === "completed") {
        this.job = transitionJob(job, "completed", {
          step: trainer.step,
          tokensSeen: trainer.tokensSeen,
          trainLoss: resolved.lastLoss ?? job.trainLoss,
          validationLoss: resolved.validationLoss ?? job.validationLoss,
          bestLoss: resolved.bestLoss ?? job.bestLoss,
        });
      } else if (resolved.state === "paused") {
        this.job = transitionJob(job, "paused", { step: trainer.step, tokensSeen: trainer.tokensSeen });
      } else if (resolved.state === "stopped") {
        this.job = transitionJob(job, "stopped", { step: trainer.step, tokensSeen: trainer.tokensSeen });
      }
      await this.persistence?.saveTrainingJob?.(this.job);
      yield { type: "job", job: this.job };
      this.observability.endSpan(span, {
        attributes: {
          runId: this.job.id,
          state: this.job.state,
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
      if (this.job && (this.job.state === "running" || this.job.state === "paused")) {
        try {
          this.job = transitionJob(this.job, "failed", { error: described.message });
          await this.persistence?.saveTrainingJob?.(this.job);
        } catch {
          // A run that cannot even be marked failed must not mask the original
          // error, which is what the caller actually needs to see.
        }
      }
      throw error;
    } finally {
      this.training = false;
    }
  }

  /**
   * Ask the running job to pause. It takes effect at the next optimiser step
   * boundary: the trainer writes a checkpoint, records the run as paused and
   * returns, leaving the run resumable rather than restartable.
   */
  pauseTraining(): AlphaTrainingJob | null {
    if (!this.job || this.job.state !== "running") return this.job;
    this.pauseRequested = true;
    return this.job;
  }

  /**
   * Clear a pause request. The caller then drives `train()` again in streaming
   * form: the same trainer continues from the step it stopped at.
   */
  resumeTraining(): AlphaTrainingJob | null {
    if (!this.job) return null;
    this.pauseRequested = false;
    return this.job;
  }

  /** Ask the running job to stop for good. It is not resumable afterwards. */
  stopTraining(): AlphaTrainingJob | null {
    if (!this.job || (this.job.state !== "running" && this.job.state !== "paused")) return this.job;
    this.stopRequested = true;
    return this.job;
  }

  /**
   * Run the deterministic A–I verification suite.
   *
   * Verification trains its own fresh instance of the configured architecture,
   * so the weights this workspace is holding are never disturbed by it. That is
   * stated here because a verification that silently retrained the live model
   * would be a lie about what was verified.
   */
  verify(options: { training?: Partial<TrainingConfig>; prompt?: string } = {}): VerificationReport {
    const tokenizer = this.tokenizer;
    if (!tokenizer) {
      throw new AlphaError("alpha.not_initialised", "core", "workspace must be initialised before verification");
    }
    const span = this.observability.startSpan({ name: "training.verify", kind: "training", module: "training" });
    const report = verifyAlphaModel({
      model: new AlphaTransformer({ ...this.model.config }),
      tokenizer,
      dataset: this.dataset,
      training: options.training,
      prompt: options.prompt ?? "Alpha is a self owned",
    });
    this.verification = report;
    this.observability.endSpan(span, {
      status: report.passed ? "ok" : "error",
      attributes: {
        passed: report.passed,
        failed: report.checks.filter((check) => !check.passed).map((check) => check.id).join(","),
      },
    });
    if (!report.passed) {
      this.errors.push(`verification failed: ${report.checks.filter((c) => !c.passed).map((c) => c.id).join(", ")}`);
    }
    return report;
  }

  /**
   * The corpus as the trainer will consume it: documents, tokens, vocabulary,
   * split sizes, example counts, sequence length and batch size.
   */
  corpusSummary(options: { seqLen?: number; batchSize?: number } = {}): CorpusReport {
    const tokenizer = this.tokenizer;
    if (!tokenizer) {
      throw new AlphaError("alpha.not_initialised", "core", "workspace must be initialised before reading the corpus");
    }
    return corpusReport(this.dataset, tokenizer, {
      seqLen: options.seqLen ?? this.config.training.seqLen,
      batchSize: options.batchSize ?? this.config.training.batchSize,
      validationFraction: this.config.training.validationFraction,
    });
  }

  /**
   * Restore Alpha from a stored checkpoint and continue from its step.
   * The vocabulary must match, because weights loaded into a different
   * architecture would silently mean nothing.
   */
  resumeFrom(checkpoint: AlphaCheckpoint, options: { totalSteps?: number } = {}): void {
    const { tokenizer } = this.requireReady();
    // Architecture, vocabulary and tokenizer fingerprint must all match, or the
    // weights would load into shape-compatible but meaningless positions.
    assertCheckpointCompatible(checkpoint, { config: this.model.config, tokenizer });

    this.model.loadWeights(checkpoint.weights);
    this.checkpoint = checkpoint;

    // Continue with the configuration the run was started with, so the
    // schedule and step budget are the same run rather than a new one. If the
    // checkpoint is already at its recorded end, the caller extends it (or we
    // extend by a single step so a resume is never a silent no-op).
    const recorded = checkpoint.trainingConfig;
    const requestedTotal = options.totalSteps ?? recorded.totalSteps;
    const totalSteps =
      requestedTotal > checkpoint.step ? requestedTotal : Math.max(requestedTotal, checkpoint.step + 1);
    const trainer = new AlphaTrainer({
      model: this.model,
      tokenizer,
      dataset: this.dataset,
      config: { ...recorded, totalSteps },
      checkpointLabel: checkpoint.label,
      runId: checkpoint.runId,
      isFineTune: true,
    });
    trainer.resumeFrom(checkpoint);
    this.trainer = trainer;
    // A run resumed from a checkpoint continues that run's record; `train()`
    // sees the pending resume and continues this trainer rather than building a
    // fresh model.
    this.job = null;
    this.pendingResume = true;
    this.refreshQuality();
    this.observability.setModelStage(checkpoint.stage, {
      reason: "resumed from a stored checkpoint",
      step: checkpoint.step,
      tokensSeen: checkpoint.tokensSeen,
    });
    this.observability.logger.log(
      "info",
      "workspace",
      `resumed ${checkpoint.modelName} at step ${checkpoint.step} toward ${totalSteps} (run ${checkpoint.runId})`,
      {
        validationLoss: checkpoint.metrics.validationLoss,
        datasetFingerprint: checkpoint.datasetFingerprint,
        tokenizerFingerprint: checkpoint.tokenizer.fingerprint,
      },
    );
  }

  /**
   * Run a request through the AI runtime.
   *
   * This is the entry point a conversation uses. It decides between plain
   * inference, memory recall, retrieval, a tool and an agent run, and returns
   * the full provenance: what route it took, what context it assembled, what it
   * cited, what tools ran, and whether the answer is grounded. A failure comes
   * back as an error on the result — never as invented text.
   */
  async respond(request: Omit<RespondRequest, "actorId"> & { actorId?: string }): Promise<RespondResult> {
    this.requireReady();
    if (!this.aiRuntime) {
      throw new AlphaError(
        "alpha.not_initialised",
        "inference",
        "the AI runtime is not available; initialise the workspace first",
      );
    }
    const actorId = request.actorId ?? this.ownerActorId;
    if (!actorId) {
      throw new AlphaError("alpha.validation_failed", "inference", "respond() needs an actorId");
    }
    const result = await this.aiRuntime.respond({ ...request, actorId });
    if (result.generation) this.lastGeneration = result.generation;
    if (result.agentRun) {
      this.observability.recordAgentRun(result.agentRun);
    }
    this.observability.recordInference(
      result.generation ??
        ({
          modelStage: result.modelStage,
          latencyMs: result.durationMs,
          generatedTokens: 0,
          promptTokens: 0,
          tokensPerSecond: 0,
          stopReason: "max-tokens",
        } as unknown as GenerationResult),
      result.traceId,
    );
    void this.persistence?.saveRun?.({
      kind: "inference",
      traceId: result.traceId,
      status: result.error ? "failed" : "succeeded",
      input: result.message.slice(0, 4000),
      output: result.response.slice(0, 4000),
      metrics: {
        route: result.route.decision,
        memories: result.memories.length,
        sources: result.verification.sourceCount,
        toolCalls: result.toolCalls.length,
        grounded: result.verification.grounded,
      },
      modelStage: result.modelStage ?? "untrained",
    });
    return result;
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
    /** Optional: defaults to the workspace's own actor. */
    ownerId?: string;
    kind?: string;
    license?: string;
    metadata?: Record<string, string>;
  }): Promise<IngestedDocument> {
    const { rag } = this.requireReady();
    const ownerId = input.ownerId ?? this.ownerActorId;
    if (!ownerId) {
      throw new AlphaError("alpha.validation_failed", "rag", "ingestDocument needs an ownerId");
    }
    const span = this.observability.startSpan({ name: "rag.ingest", kind: "retrieval", module: "rag" });
    const { document } = rag.ingest({ ...input, ownerId });
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
    const answer = rag.answer(question, sampling, { ownerId: this.ownerActorId });
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
    ownerId?: string;
    provenance?: Partial<MemoryProvenance>;
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
    const record = memory.write({
      ...input,
      ownerId: input.ownerId ?? this.ownerActorId,
      source: "user",
      approved: input.approved,
      provenance: input.provenance ?? {
        origin: "user",
        referenceId: input.sessionId ?? null,
        recordedBy: this.ownerActorId,
      },
    });
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
        configFingerprint: modelConfigFingerprint(this.model.config),
        trainedTokens: this.checkpoint?.tokensSeen ?? 0,
        validationLoss: this.checkpoint?.metrics.validationLoss ?? null,
        checkpointId: this.checkpoint?.id ?? null,
      },
      models: this.inferenceService.listModels(),
      resources: {
        limits: ALPHA_RESOURCE_LIMITS,
        estimate: estimateTrainingMemory(this.model.config, {
          batchSize: this.trainer?.config.batchSize ?? this.config.training.batchSize,
          seqLen: this.trainer?.config.seqLen ?? this.config.training.seqLen,
        }),
      },
      corpusReport: tokenizer
        ? corpusReport(this.dataset, tokenizer, {
            seqLen: this.trainer?.config.seqLen ?? this.config.training.seqLen,
            batchSize: this.trainer?.config.batchSize ?? this.config.training.batchSize,
            validationFraction: this.config.training.validationFraction,
          })
        : null,
      verification: this.verification,
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
        job: this.job,
        jobSummary: this.job ? summariseJob(this.job) : null,
        pauseRequested: this.pauseRequested,
        stopRequested: this.stopRequested,
      },
      inference: {
        requests: inferenceMetrics.value("alpha.inference.requests"),
        tokensGenerated: inferenceMetrics.value("alpha.inference.tokens_generated"),
        averageLatencyMs: latency?.average ?? 0,
        p95LatencyMs: latency?.p95 ?? 0,
        last: this.lastGeneration,
        // Whether decoding is running against the KV cache, and the real
        // counters from the last generation that used it.
        useCache: this.inference?.useCache ?? false,
        cache: this.lastGeneration?.cache ?? {
          used: false,
          positions: 0,
          capacity: 0,
          bytes: 0,
          writes: 0,
          hits: 0,
          occupancy: 0,
        },
      },
      aiRuntime: {
        available: this.aiRuntime !== null,
        embedding: this.embedder?.describe() ?? null,
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
