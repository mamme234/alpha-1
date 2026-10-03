/**
 * Alpha — public API.
 *
 * Everything the application layer is allowed to import lives here. The
 * subdirectories under `src/alpha` are the implementation; this file is the
 * contract, and it mirrors the documented architecture exactly:
 *
 *   core · model · tokenizer · training · inference · embeddings · rag ·
 *   memory · agents · tools · mcp · vector · automation · security ·
 *   observability · datasets · configs
 *
 * Alpha is self-contained: no external AI/LLM provider is called anywhere in
 * this tree, and no placeholder integration exists to be "switched on" later.
 */

// --- core -------------------------------------------------------------------
export {
  Tensor,
  add,
  addScalar,
  backward,
  causalSoftmax,
  crossEntropy,
  dropout,
  gatherRows,
  gelu,
  gradL2Norm,
  layerNorm,
  matmul,
  maxAbsDiff,
  mergeHeads,
  permute,
  resetGrad,
  reshape,
  scale,
  setGradEnabled,
  isGradEnabled,
  shapeSize,
  sliceSeq,
  splitHeads,
  transposeLastTwo,
  type Shape,
} from "./core/tensor";
export { AlphaRng, type RngState } from "./core/rng";
export {
  AlphaCheckpointError,
  AlphaError,
  AlphaNotImplementedError,
  AlphaPermissionError,
  AlphaRateLimitError,
  AlphaToolError,
  AlphaUntrainedModelError,
  AlphaValidationError,
  describeError,
  isAlphaError,
  type AlphaErrorModule,
} from "./core/errors";
export {
  ALPHA_STATUS_ORDER,
  alphaId,
  byteLength,
  formatBytes,
  modelStageLabel,
  newSpanId,
  newTraceId,
  round,
  statusLabel,
  type AlphaModelStage,
  type AlphaModuleDescriptor,
  type AlphaStatus,
} from "./core/types";
export { base64ToFloat32, float32ToBase64 } from "./core/serialize";
export {
  ALPHA_RESOURCE_LIMITS,
  assertResourceLimit,
  countParametersFromConfig,
  estimateParameterBytes,
  estimateTrainingMemory,
  type AlphaResourceKind,
  type TrainingMemoryEstimate,
} from "./core/limits";

// --- model ------------------------------------------------------------------
export {
  ALPHA_MODEL_PRESETS,
  countParameters,
  createModelArtifact,
  createModelConfig,
  deriveStage,
  describeArchitecture,
  modelConfigFingerprint,
  validateModelConfig,
  withSpecialTokenIds,
  type AlphaModelArtifact,
  type AlphaModelConfig,
  type AlphaModelPreset,
  type AlphaSpecialTokenIds,
  type ArchitectureTensorRow,
  type PositionalEncodingKind,
} from "./model/config";
export {
  AlphaTransformer,
  type ForwardOptions,
  type ForwardResult,
  type ModelParameter,
  type SerializedWeights,
} from "./model/transformer";

// --- tokenizer --------------------------------------------------------------
export {
  AlphaTokenizer,
  DEFAULT_SPECIAL_TOKENS,
  preTokenize,
  type AlphaSpecialTokens,
  type AlphaTokenizerSnapshot,
  type BpeTrainingOptions,
  type EncodedSequence,
  type EncodeOptions,
} from "./tokenizer/bpe";

// --- datasets ---------------------------------------------------------------
export {
  assertValidDataset,
  createDataset,
  datasetFingerprint,
  datasetReference,
  datasetStats,
  splitDocuments,
  validateDataset,
  type AlphaDataset,
  type DatasetIssue,
  type DatasetStats,
  type DatasetValidation,
} from "./datasets/types";
export {
  BatchSampler,
  DocumentBatchSampler,
  countTrainingExamples,
  corpusReport,
  encodeCorpus,
  padSequences,
  type CorpusReport,
  type CorpusStats,
  type EncodeCorpusOptions,
  type EncodedCorpus,
  type TrainingBatch,
} from "./datasets/corpus";
export { ALPHA_SEED_CORPUS, seedCorpusSlice } from "./datasets/seed-corpus";

// --- training ---------------------------------------------------------------
export {
  AdamW,
  DEFAULT_ADAMW,
  type AdamWConfig,
  type OptimizerStateSnapshot,
  type OptimizerStepReport,
  type ParameterHandle,
} from "./training/optimizer";
export {
  defaultSchedule,
  describeSchedule,
  learningRateAt,
  type ScheduleConfig,
  type ScheduleKind,
} from "./training/schedule";
export {
  ALPHA_CHECKPOINT_FORMAT_VERSION,
  assertCheckpointCompatible,
  assertValidCheckpoint,
  checkpointToJson,
  compareCheckpoints,
  createCheckpoint,
  estimateCheckpointBytes,
  parseCheckpoint,
  summariseCheckpoint,
  validateCheckpoint,
  withCheckpointId,
  type AlphaCheckpoint,
  type AlphaCheckpointMetrics,
  type AlphaCheckpointSummary,
  type CheckpointCompatibility,
  type CheckpointTokenizerRef,
  type CheckpointValidation,
  type CreateCheckpointInput,
} from "./training/checkpoint";
export {
  AlphaTrainer,
  DEFAULT_TRAINING_CONFIG,
  createTrainingConfig,
  type BatchMode,
  type EarlyStoppingConfig,
  type EarlyStoppingReport,
  type EvaluationResult,
  type TrainerOptions,
  type TrainingConfig,
  type TrainingEvent,
  type TrainingMetricPoint,
  type TrainingSummary,
} from "./training/trainer";
export {
  numericalGradientCheck,
  type GradientCheckBatch,
  type GradientCheckOptions,
  type GradientCheckReport,
} from "./training/gradients";
export {
  TRAINING_JOB_STATES,
  TRAINING_JOB_TRANSITIONS,
  canTransitionJob,
  createTrainingJob,
  failJob,
  jobProgress,
  recordJobCheckpoint,
  recordJobEvaluation,
  recordJobStep,
  summariseJob,
  trainingJobStateLabel,
  transitionJob,
  validateTrainingJob,
  type AlphaTrainingJob,
  type CreateTrainingJobInput,
  type TrainingJobState,
  type TrainingJobStateLabel,
} from "./training/job";
export {
  DEFAULT_VERIFY_TRAINING,
  verifyAlphaModel,
  type VerificationCheck,
  type VerificationCheckId,
  type VerificationReport,
  type VerifyAlphaModelOptions,
} from "./training/verify";
export {
  DEFAULT_LIFECYCLE_TRAINING,
  formatLifecycleReport,
  runAlphaTrainingLifecycle,
  type AlphaLifecycleReport,
  type LifecycleOptions,
  type LifecycleStage,
  type LifecycleStageName,
} from "./training/lifecycle";

// --- context engine ---------------------------------------------------------
export {
  AlphaContextEngine,
  type AssembledContext,
  type AssembleOptions,
  type ContextBlock,
  type ContextBlockKind,
  type ContextBlockReport,
} from "./context/engine";

// --- inference --------------------------------------------------------------
export {
  AlphaInferenceEngine,
  DEFAULT_SAMPLING,
  SAMPLING_PRESETS,
  argmax,
  createGenerationCancellation,
  meanNll,
  untrainedWarning,
  type GenerationCancellation,
  type GenerationResult,
  type GenerationStreamChunk,
  type InferenceEngineOptions,
  type SamplingConfig,
  type StopReason,
} from "./inference/engine";
export {
  createKvCache,
  kvCacheDropOldest,
  kvCacheHasRoom,
  kvCacheOverflow,
  kvCacheStats,
  resetKvCache,
  type KvCache,
  type KvCacheStats,
} from "./model/kv-cache";
export { type CachedForwardOptions } from "./model/transformer";
export {
  AlphaInferenceService,
  type AlphaModelDescriptor,
  type AlphaModelHandle,
  type GenerateRequest,
  type GenerateStreamRequest,
  type RegisterModelInput,
  type ScoreRequest,
} from "./inference/service";

// --- embeddings -------------------------------------------------------------
export {
  AlphaEmbedder,
  SIMILARITY_FUNCTIONS,
  centroid,
  cosineSimilarity,
  createEmbeddingConfig,
  describeEmbeddingConfigMismatch,
  dotProduct,
  embeddingConfigsMatch,
  euclideanDistance,
  normalize,
  type AlphaEmbedderOptions,
  type AlphaEmbeddingConfig,
  type EmbeddingRecord,
  type PoolingStrategy,
} from "./embeddings/embedder";

// --- vector store -----------------------------------------------------------
export {
  AlphaVectorStore,
  type VectorCollectionInfo,
  type VectorMetric,
  type VectorRecord,
  type VectorSearchHit,
  type VectorSearchRequest,
  type VectorStoreSnapshot,
} from "./vector/store";

// --- rag --------------------------------------------------------------------
export {
  AlphaRagPipeline,
  DEFAULT_RAG_CONFIG,
  buildRagPrompt,
  type DocumentChunk,
  type DocumentKind,
  type IngestedDocument,
  type RagAnswer,
  type RagAnswerSource,
  type RagConfig,
  type SourceDocument,
} from "./rag/pipeline";

// --- memory -----------------------------------------------------------------
export {
  AlphaMemoryStore,
  ConversationMemory,
  DEFAULT_MEMORY_CONFIG,
  type MemoryRecord,
  type MemoryRetrieval,
  type MemoryScope,
  type MemoryStoreConfig,
  type MemoryStoreSnapshot,
  type MemoryWriteInput,
} from "./memory/store";

// --- tools ------------------------------------------------------------------
export {
  AlphaToolRegistry,
  AlphaToolTimeoutError,
  DEFAULT_TOOL_TIMEOUT_MS,
  describeTool,
  singleStringInput,
  withToolTimeout,
  type AlphaToolDefinition,
  type ToolCharacteristics,
  type ToolContext,
  type ToolDescriptor,
  type ToolDiscoveryQuery,
  type ToolExecutionRecord,
  type ToolRunResult,
  type ToolServices,
  type ToolVerification,
} from "./tools/registry";
export {
  assertAgainstSchema,
  objectSchema,
  validateAgainstSchema,
  type JsonSchema,
  type JsonSchemaType,
} from "./tools/schema";
export { BUILTIN_TOOL_NAMES, registerBuiltinTools } from "./tools/builtin";
export { ExpressionParser, evaluateExpression } from "./tools/expression";

// --- mcp --------------------------------------------------------------------
export {
  MCP_METHODS,
  MCP_PROTOCOL_VERSION,
  mcpContentToText,
  isJsonRpcFailure,
  type McpCallToolResult,
  type McpServerInfo,
  type McpToolDescriptor,
  type JsonRpcRequest,
  type JsonRpcResponse,
} from "./mcp/protocol";
export {
  DEFAULT_MCP_TIMEOUT_MS,
  HttpMcpClient,
  UnconfiguredMcpClient,
  convertJsonSchema,
  normaliseMcpResult,
  registerMcpTools,
  type McpClient,
  type McpClientInfo,
  type McpNormalisedResult,
} from "./mcp/client";

// --- agents -----------------------------------------------------------------
export {
  AlphaAgentRuntime,
  buildToolArguments,
  createAgentCancellation,
  runAgentTask,
  type AgentCancellation,
  type AgentRunRequest,
  type AgentRuntimeOptions,
} from "./agents/runtime";
export { reconcileOutcome, verifyAgentOutcome, type VerifyAgentOutcomeInput } from "./agents/verify";
export {
  buildPlannerPrompt,
  extractCapabilityPhrases,
  parsePlannerJson,
  planFromCapabilities,
  planWithModel,
} from "./agents/planner";
export type {
  AgentEvent,
  AgentOutcome,
  AgentPlan,
  AgentPlanStep,
  AgentRunResult,
  AgentRunStatus,
  AgentStepRecord,
  AgentSynthesis,
  AgentTask,
  AgentVerification,
  PlannerSource,
} from "./agents/types";

// --- ai runtime orchestrator -----------------------------------------------
export {
  AlphaAiRuntime,
  type AlphaAiRuntimeOptions,
  type RespondRequest,
  type RespondResult,
  type RespondRoute,
  type RespondStreamEvent,
  type RespondStreamOptions,
  type RespondVerification,
  type RouteDecision,
} from "./runtime/orchestrator";

// --- automation -------------------------------------------------------------
export {
  AlphaAutomationEngine,
  evaluateCondition,
  type ActionOutcome,
  type AlphaWorkflow,
  type AutomationEngineOptions,
  type ConditionOperator,
  type JobExecution,
  type JobStatus,
  type WorkflowAction,
  type WorkflowCondition,
  type WorkflowTrigger,
} from "./automation/engine";

// --- security ---------------------------------------------------------------
export {
  ALL_PERMISSIONS,
  AlphaPolicyEngine,
  DEFAULT_ROLE_PERMISSIONS,
  permissionForTool,
  type AgentScope,
  type AlphaPermission,
  type AlphaRole,
  type PolicyDecision,
  type PolicyEvent,
} from "./security/policy";
export {
  INJECTION_PATTERNS,
  assessInjectionRisk,
  detectPromptInjection,
  redactSecrets,
  validateOutput,
  validateTextInput,
  wrapUntrustedContent,
  type InjectionAssessment,
  type InjectionFinding,
  type OutputValidation,
} from "./security/validation";
export {
  AlphaRateLimiter,
  DEFAULT_RATE_LIMITS,
  type RateLimitDecision,
  type RateLimitPolicy,
} from "./security/rate-limit";
export { AlphaAuditLog, type AuditDecision, type AuditRecord } from "./security/audit";
export {
  AlphaSandbox,
  DEFAULT_SANDBOX,
  type SandboxSpec,
  type SandboxViolation,
} from "./security/sandbox";

// --- observability ----------------------------------------------------------
export { AlphaTracer, type Span, type SpanKind, type SpanStatus, type TraceSummary } from "./observability/tracer";
export { ALPHA_METRICS, AlphaMetricsRegistry, type HistogramSummary, type MetricSnapshot } from "./observability/metrics";
export { AlphaLogger, type LogEntry, type LogLevel, type ModuleLogger } from "./observability/logger";
export { AlphaObservability, type ObservabilitySnapshot } from "./observability/index";

// --- configs ----------------------------------------------------------------
export {
  DEFAULT_ALPHA_CONFIG,
  createAlphaConfig,
  describeConfig,
  parseAlphaConfig,
  serialiseAlphaConfig,
  validateAlphaConfig,
  type AlphaConfig,
  type AlphaConfigOverrides,
} from "./configs/alpha.config";

// --- step 4: dataset versioning + quality ---------------------------------
export {
  ALPHA_TRAINABLE_LICENSES,
  DEFAULT_FILTERS,
  DEFAULT_NORMALISATION,
  assertTrainableDatasetVersion,
  buildSplits,
  createDatasetVersion,
  describeDatasetVersion,
  hasInvalidUnicode,
  normaliseDocument,
  repetitionRatio,
  splitDataset,
  toDataset,
  type AlphaDatasetVersion,
  type CreateDatasetVersionInput,
  type DatasetSource,
  type DatasetSplit,
  type DatasetVersionManifest,
  type FilterDecision,
  type FilterSettings,
  type NormalisationSettings,
} from "./datasets/versions";
export {
  analyseDatasetQuality,
  analyseDatasetVersionQuality,
  summariseQualityReport,
  type QualityFinding,
  type QualityOptions,
  type QualityReport,
  type QualitySeverity,
} from "./datasets/quality";
export {
  buildGeneratedCorpus,
  generatedCorpusSize,
  generatedCorpusTopics,
} from "./datasets/generated-corpus";

// --- step 4: evaluation ----------------------------------------------------
export {
  ALPHA_BENCHMARK_VERSION,
  assertNotBenchmark,
  benchmarkIdentity,
  createAlphaBenchmark,
  heldOutCases,
  recallCases,
} from "./evaluation/benchmark";
export {
  evaluateModel,
  metricValue,
  passedChecks,
  summariseEvaluation,
  type CapabilityResult,
  type EvaluationMetric,
  type EvaluationOptions,
  type EvaluationReport,
} from "./evaluation/framework";

// --- step 4: training scaling ----------------------------------------------
export {
  RUNTIME_CAPABILITIES,
  TRAINING_LIFECYCLE_STATES,
  TRAINING_LIFECYCLE_TRANSITIONS,
  assertTrainableWithinLimits,
  canTransitionLifecycle,
  describeRun,
  estimateResources,
  lifecycleFromJobState,
  terminalLifecycle,
  tokenMetricsFromSummary,
  type ResourceEstimate,
  type TokenMetrics,
  type TrainingLifecycleState,
} from "./training/scaling";

// --- step 4: model registry + export ---------------------------------------
export {
  AlphaModelRegistry,
  MODEL_LIFECYCLE_ORDER,
  MODEL_LIFECYCLE_TRANSITIONS,
  modelIdFor,
  newModelId,
  type CompatibilityCheck,
  type CompatibilityReport,
  type ModelLifecycle,
  type ModelPromotion,
  type ModelRelationships,
  type RegisteredModel,
  type RegisterModelRequest,
} from "./model/registry";
export {
  ALPHA_MODEL_EXPORT_VERSION,
  exportId,
  exportModel,
  importModel,
  parseModelExport,
  serialiseModelExport,
  type AlphaModelExport,
  type ExportModelInput,
  type ImportResult,
} from "./model/export";

// --- step 5: provenance, corpus, mixture, splits --------------------------
export {
  ACQUISITION_LOCATIONS,
  ALPHA_MIX_CATEGORIES,
  PROVENANCE_ORIGINS,
  assertProvenance,
  categoryCounts,
  createProvenanceDocument,
  describeProvenance,
  documentFingerprint,
  languageCounts,
  provenanceFingerprint,
  sourceCounts,
  validateProvenance,
  withSourceCounts,
  type AcquisitionLocation,
  type AcquisitionMethod,
  type AcquisitionRecord,
  type MixCategory,
  type ProvenanceCorpus,
  type ProvenanceDocument,
  type ProvenanceIssue,
  type ProvenanceOrigin,
  type ProvenanceSource,
  type ProvenanceValidation,
} from "./datasets/provenance";
export {
  authoredCategories,
  buildAuthoredCorpus,
  multilingualSamples,
  type AuthoredDocument,
} from "./datasets/authored-corpus";
export {
  analyseDiversity,
  compareDiversity,
  splitSentences,
  summariseDiversity,
  words,
  type DiversityDelta,
  type DiversityOptions,
  type DiversityReport,
} from "./datasets/diversity";
export {
  buildMixture,
  describeMixture,
  type MixtureComponent,
  type MixtureInput,
  type MixtureRecord,
  type MixtureResult,
} from "./datasets/mixture";
export {
  INSTRUCTION_SKILLS,
  assertNoInstructionLeakage,
  assertValidInstructionDataset,
  createInstructionDataset,
  createInstructionExample,
  detectInstructionLeakage,
  instructionDocuments,
  instructionFingerprint,
  renderInstructionExample,
  renderInstructionPrompt,
  summariseInstructionDataset,
  validateInstructionDataset,
  type CreateInstructionExampleInput,
  type InstructionDataset,
  type InstructionExample,
  type InstructionIssue,
  type InstructionSkill,
  type InstructionValidation,
  type LeakageReport,
} from "./datasets/instructions";
export {
  assertNoSplitLeakage,
  assignSplits,
  auditSplitOverlap,
  detectOverlap,
  describeSplits,
  nearDuplicateGroupKey,
  repairSplitLeakage,
  shingleFingerprints,
  splitDocuments as splitProvenanceDocuments,
  type OverlapOptions,
  type OverlapReport,
  type SplitAssignment,
  type SplitName,
  type SplitOptions,
} from "./datasets/splits";

// --- step 5: tokenizer measurement -----------------------------------------
export {
  DEFAULT_TOKENIZER_DECISION_THRESHOLDS,
  decideTokenizerChange,
  measureTokenizer,
  summariseTokenizerDecision,
  summariseTokenizerMeasurement,
  type TokenizerDecision,
  type TokenizerDecisionReason,
  type TokenizerDecisionThresholds,
  type TokenizerMeasurement,
} from "./tokenizer/analysis";

// --- step 5: expanded evaluation --------------------------------------------
export {
  ALPHA_EVAL_SUITE_VERSION,
  EVAL_CATEGORIES,
  assertSuiteFrozen,
  assertSuiteNotInTraining,
  auditSuiteLeakage,
  caseWords,
  createEvalSuite,
  describeEvalSuite,
  populatedCategories,
  suiteFingerprint,
  type EvalCase,
  type EvalCategory,
  type EvalSplit,
  type EvalSuite,
  type FormatRequirement,
  type SuiteLeakageReport,
} from "./evaluation/suite";
export {
  DEFAULT_EVAL_GENERATION_TOKENS,
  categoryReport,
  checkFormat,
  generations,
  languageModelingMetrics,
  runEvalSuite,
  summariseCapabilityReport,
  type CapabilityReport,
  type CategoryReport,
  type EvalCaseResult,
  type RunSuiteOptions,
} from "./evaluation/runner";
export {
  CAPABILITY_GATE_CRITERIA,
  GATE_REQUIRED_PASSES,
  assertGateStable,
  describeGate,
  evaluateGate,
  gateFingerprint,
  type GateCriterion,
  type GateCriterionResult,
  type GateDirection,
  type GateEvaluation,
} from "./evaluation/gate";

// --- step 5: experiment tracking --------------------------------------------
export {
  EXPERIMENT_STATUSES,
  abortExperiment,
  completeExperiment,
  compareCapability,
  createExperiment,
  describeComparison,
  describeReproduction,
  failExperiment,
  recordCapability,
  recordTraining,
  summariseExperiment,
  type CapabilityComparisonRow,
  type CreateExperimentInput,
  type Experiment,
  type ExperimentMeasurements,
  type ExperimentStatus,
} from "./training/experiments";

// --- step 6: serving the verified model --------------------------------------
export {
  ALPHA_SERVING_ARTIFACT_FORMAT,
  createServingArtifact,
  describeServingArtifact,
  loadServingArtifact,
  parseServingArtifact,
  type AlphaServingArtifact,
  type LoadedServingArtifact,
} from "./serving/artifact";
export {
  ALPHA_CHAT_TOOL_ALLOWLIST,
  ALPHA_SERVING_SYSTEM_INSTRUCTION,
  AlphaServingRuntime,
  createServingRuntime,
  type AlphaServingRuntimeOptions,
  type ServingHydrationInput,
  type ServingHydrationReport,
  type ServingMemoryRecord,
  type ServingRuntimeHealth,
  type ServingRuntimeInfo,
  type ServingRuntimeLimits,
  type ServingVectorRecord,
} from "./serving/runtime";
export {
  CHAT_MAX_HISTORY_TURNS,
  buildChatRequest,
  runChatTurn,
  runChatTurnStream,
  type ChatHistoryTurn,
  type ChatTurnInput,
  type ChatTurnSettings,
} from "./serving/chat";

// --- manifest + workspace ---------------------------------------------------
export { ALPHA_MODULES, moduleById, moduleStatusCounts } from "./modules";
export {
  ALPHA_AGENT_ACTOR,
  ALPHA_OWNER_ACTOR,
  ALPHA_SYSTEM_ACTOR,
  AlphaWorkspace,
  type AlphaPersistenceAdapter,
  type AlphaWorkspaceOptions,
  type AlphaWorkspaceSnapshot,
  type PersistedRunInput,
} from "./workspace";
