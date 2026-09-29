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

// --- model ------------------------------------------------------------------
export {
  ALPHA_MODEL_PRESETS,
  countParameters,
  createModelArtifact,
  createModelConfig,
  deriveStage,
  describeArchitecture,
  validateModelConfig,
  type AlphaModelArtifact,
  type AlphaModelConfig,
  type AlphaModelPreset,
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
  createDataset,
  datasetStats,
  splitDocuments,
  type AlphaDataset,
  type DatasetStats,
} from "./datasets/types";
export { BatchSampler, encodeCorpus, type EncodedCorpus, type TrainingBatch } from "./datasets/corpus";
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
  checkpointToJson,
  createCheckpoint,
  estimateCheckpointBytes,
  parseCheckpoint,
  summariseCheckpoint,
  type AlphaCheckpoint,
  type AlphaCheckpointMetrics,
  type AlphaCheckpointSummary,
} from "./training/checkpoint";
export {
  AlphaTrainer,
  DEFAULT_TRAINING_CONFIG,
  createTrainingConfig,
  type EvaluationResult,
  type TrainerOptions,
  type TrainingConfig,
  type TrainingEvent,
  type TrainingMetricPoint,
  type TrainingSummary,
} from "./training/trainer";

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
  argmax,
  meanNll,
  type GenerationResult,
  type GenerationStreamChunk,
  type InferenceEngineOptions,
  type SamplingConfig,
  type StopReason,
} from "./inference/engine";

// --- embeddings -------------------------------------------------------------
export {
  AlphaEmbedder,
  SIMILARITY_FUNCTIONS,
  centroid,
  cosineSimilarity,
  dotProduct,
  euclideanDistance,
  normalize,
  type AlphaEmbedderOptions,
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
  describeTool,
  singleStringInput,
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
  HttpMcpClient,
  UnconfiguredMcpClient,
  convertJsonSchema,
  registerMcpTools,
  type McpClient,
  type McpClientInfo,
} from "./mcp/client";

// --- agents -----------------------------------------------------------------
export {
  AlphaAgentRuntime,
  buildToolArguments,
  runAgentTask,
  type AgentRunRequest,
  type AgentRuntimeOptions,
} from "./agents/runtime";
export {
  buildPlannerPrompt,
  extractCapabilityPhrases,
  parsePlannerJson,
  planFromCapabilities,
  planWithModel,
} from "./agents/planner";
export type {
  AgentEvent,
  AgentPlan,
  AgentPlanStep,
  AgentRunResult,
  AgentRunStatus,
  AgentStepRecord,
  AgentSynthesis,
  AgentTask,
  PlannerSource,
} from "./agents/types";

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
