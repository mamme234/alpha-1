/**
 * Training scaling and honest resource estimation.
 *
 * Two jobs live here, and both come from the same principle: a number that is
 * an estimate must be labelled as one, and a feature that is not implemented
 * must not be claimed.
 *
 * Resource estimation exists so a run can be refused *before* it starts rather
 * than discovered to be impossible halfway through. Every figure it returns is
 * derived arithmetic from the architecture and the batch configuration; none of
 * it is a hardware benchmark, and it says so.
 *
 * Token metrics exist because "steps" is the wrong unit for a language model.
 * A step over a 32-token window and a step over a 2048-token window are not
 * comparable. Everything here is normalised to tokens actually processed.
 */

import { AlphaValidationError } from "../core/errors";
import { ALPHA_RESOURCE_LIMITS, estimateTrainingMemory, type TrainingMemoryEstimate } from "../core/limits";
import { countParameters, type AlphaModelConfig } from "../model/config";
import type { TrainingConfig } from "./trainer";
import type { TrainingSummary } from "./trainer";

/** Bytes per float32 value. */
const BYTES_PER_FLOAT = 4;

/**
 * What this runtime can genuinely do. Each entry is a *measured* statement
 * about the code, not an aspiration — Step 5 only turned any of these to `true`
 * after a numerical test proved the behaviour.
 */
export const RUNTIME_CAPABILITIES = {
  /** Alpha's tensors are float32 throughout; there is no mixed-precision path. */
  mixedPrecision: false,
  /**
   * `AlphaTrainer.accumulateGradients` sums the token-weighted gradient of
   * several micro-batches and applies exactly one optimiser update, verified
   * against a single pass over the concatenated batch.
   */
  gradientAccumulation: true,
  /**
   * The training loop can stop on a measured validation metric, recording the
   * monitored metric, patience, best value, best step and stopping reason.
   */
  earlyStopping: true,
  /** A paused run writes a checkpoint and can be reloaded by a new trainer. */
  resumable: true,
  /** Validation runs on a held-out fraction carved from the corpus. */
  periodicValidation: true,
  /** Checkpoints are written on an interval and always at the end of a run. */
  checkpointing: true,
} as const;

/** Why a configuration was or was not accepted for training. */
export type ResourceEstimate = {
  parameterCount: number;
  /** float32 weights only. */
  weightBytes: number;
  /** AdamW keeps two moments: 8 bytes per parameter. */
  optimizerBytes: number;
  /** One gradient per parameter. */
  gradientBytes: number;
  /** Serialised checkpoint: weights + optimiser state + RNG position. */
  checkpointBytes: number;
  /** Peak activation footprint for the configured batch, per `core/limits`. */
  activationBytes: number;
  /** weight + gradient + optimizer + activation. */
  trainingTotalBytes: number;
  /** Inference needs the weights plus one KV cache. */
  inferenceBytes: number;
  /** Tokens one optimiser step consumes. */
  tokensPerStep: number;
  /** Tokens the whole configured run will process. */
  plannedTrainingTokens: number;
  /** `trainingTotalBytes / (2^30)`, for a human. */
  trainingGib: number;
  /** Parameters are float32, so 4 bytes each, always. */
  precision: "float32";
  /** Marks every figure above as arithmetic, not measurement. */
  estimateKind: "derived-estimate";
  note: string;
  /** Non-empty when the configuration is within Alpha's configured ceilings. */
  violations: string[];
  withinLimits: boolean;
};

export function estimateResources(
  config: AlphaModelConfig,
  training: Pick<TrainingConfig, "batchSize" | "seqLen" | "totalSteps">,
): ResourceEstimate {
  const parameterCount = countParameters(config);
  const weightBytes = parameterCount * BYTES_PER_FLOAT;
  const optimizerBytes = parameterCount * 2 * BYTES_PER_FLOAT;
  const gradientBytes = parameterCount * BYTES_PER_FLOAT;
  // Weights plus both AdamW moments: what a checkpoint has to carry.
  const checkpointBytes = weightBytes + optimizerBytes;
  const kvBytes =
    config.nLayers * 2 * config.contextLength * config.dModel * BYTES_PER_FLOAT;

  const memory: TrainingMemoryEstimate = estimateTrainingMemory(config, {
    batchSize: training.batchSize,
    seqLen: training.seqLen,
  });

  const tokensPerStep = training.batchSize * training.seqLen;
  const plannedTrainingTokens = tokensPerStep * training.totalSteps;
  const trainingTotalBytes = memory.totalBytes;

  const violations: string[] = [];
  if (config.contextLength > ALPHA_RESOURCE_LIMITS.maxContextLength) {
    violations.push(
      `contextLength ${config.contextLength} exceeds Alpha's limit of ${ALPHA_RESOURCE_LIMITS.maxContextLength}`,
    );
  }
  if (training.seqLen > config.contextLength) {
    violations.push(
      `training.seqLen ${training.seqLen} exceeds the model's context length ${config.contextLength}`,
    );
  }
  if (parameterCount > ALPHA_RESOURCE_LIMITS.maxParameterCount) {
    violations.push(
      `${parameterCount.toLocaleString()} parameters exceeds Alpha's limit of ${ALPHA_RESOURCE_LIMITS.maxParameterCount.toLocaleString()}`,
    );
  }
  if (training.totalSteps > ALPHA_RESOURCE_LIMITS.maxTotalSteps) {
    violations.push(
      `totalSteps ${training.totalSteps} exceeds Alpha's limit of ${ALPHA_RESOURCE_LIMITS.maxTotalSteps}`,
    );
  }

  return {
    parameterCount,
    weightBytes,
    optimizerBytes,
    gradientBytes,
    checkpointBytes,
    activationBytes: memory.activationBytes,
    trainingTotalBytes,
    inferenceBytes: weightBytes + kvBytes,
    tokensPerStep,
    plannedTrainingTokens,
    trainingGib: trainingTotalBytes / 1024 ** 3,
    precision: "float32",
    estimateKind: "derived-estimate",
    note:
      "Arithmetic derived from the architecture and batch configuration, using the same formula as core/limits. " +
      "Not a measurement and not a hardware benchmark: it does not include allocator overhead, " +
      "interpreter overhead, or the cost of any external storage. Actual peak memory will differ.",
    violations,
    withinLimits: violations.length === 0,
  };
}

/**
 * Refuse a configuration before a run starts, with the reason. Returns the
 * estimate so the caller can still report it alongside the refusal.
 */
export function assertTrainableWithinLimits(
  config: AlphaModelConfig,
  training: Pick<TrainingConfig, "batchSize" | "seqLen" | "totalSteps">,
): ResourceEstimate {
  const estimate = estimateResources(config, training);
  if (!estimate.withinLimits) {
    throw new AlphaValidationError(
      "training",
      `configuration cannot be trained within Alpha's limits: ${estimate.violations.join("; ")}`,
      { parameterCount: estimate.parameterCount, violations: estimate.violations },
    );
  }
  return estimate;
}

/** Token-based progress for a run, measured rather than projected. */
export type TokenMetrics = {
  steps: number;
  tokensSeen: number;
  tokensPerStep: number;
  trainingExamples: number;
  /** Measured from the run's own wall clock. 0 when the run was too fast to time. */
  tokensPerSecond: number;
  durationMs: number;
  firstLoss: number | null;
  lastLoss: number | null;
  bestLoss: number | null;
  validationLoss: number | null;
  validationPerplexity: number | null;
  /** Cross-entropy of a model with no information, for scale. */
  uniformLossBaseline: number;
  /** measured lastLoss - uniformLossBaseline. Negative means the model learned. */
  lossReduction: number | null;
  checkpointCount: number;
  state: string;
};

/** Derive token metrics from a real `TrainingSummary`. No field is invented. */
export function tokenMetricsFromSummary(summary: TrainingSummary): TokenMetrics {
  const tokensPerStep =
    summary.steps > 0 ? Math.round(summary.tokensSeen / summary.steps) : 0;
  return {
    steps: summary.steps,
    tokensSeen: summary.tokensSeen,
    tokensPerStep,
    trainingExamples: summary.steps,
    tokensPerSecond: summary.throughputTokensPerSecond,
    durationMs: summary.durationMs,
    firstLoss: summary.firstLoss,
    lastLoss: summary.lastLoss,
    bestLoss: summary.bestLoss,
    validationLoss: summary.validationLoss,
    validationPerplexity: summary.validationPerplexity,
    uniformLossBaseline: summary.uniformLossBaseline,
    lossReduction:
      summary.lastLoss === null ? null : summary.lastLoss - summary.uniformLossBaseline,
    checkpointCount: summary.checkpointCount,
    state: summary.state,
  };
}

/**
 * The explicit training lifecycle. Step 1's job module used
 * created/running/paused/completed/failed/stopped; Step 4 states the same
 * machine in the vocabulary an operator uses, and maps onto it explicitly
 * rather than renaming behaviour that already works.
 */
export type TrainingLifecycleState = "QUEUED" | "RUNNING" | "PAUSED" | "COMPLETED" | "FAILED" | "CANCELLED";

export const TRAINING_LIFECYCLE_STATES: TrainingLifecycleState[] = [
  "QUEUED",
  "RUNNING",
  "PAUSED",
  "COMPLETED",
  "FAILED",
  "CANCELLED",
];

/** Legal moves. Terminal states have no exits. */
export const TRAINING_LIFECYCLE_TRANSITIONS: Record<TrainingLifecycleState, TrainingLifecycleState[]> = {
  QUEUED: ["RUNNING", "CANCELLED", "FAILED"],
  RUNNING: ["PAUSED", "COMPLETED", "FAILED", "CANCELLED"],
  PAUSED: ["RUNNING", "CANCELLED", "FAILED"],
  FAILED: ["QUEUED"],
  COMPLETED: [],
  CANCELLED: [],
};

/** Map Step 1's `TrainingJobState` onto the Step 4 lifecycle vocabulary. */
export function lifecycleFromJobState(state: string): TrainingLifecycleState {
  switch (state) {
    case "created":
      return "QUEUED";
    case "running":
      return "RUNNING";
    case "paused":
      return "PAUSED";
    case "completed":
      return "COMPLETED";
    case "failed":
      return "FAILED";
    case "stopped":
      return "CANCELLED";
    default:
      return "FAILED";
  }
}

export function canTransitionLifecycle(
  from: TrainingLifecycleState,
  to: TrainingLifecycleState,
): boolean {
  return TRAINING_LIFECYCLE_TRANSITIONS[from].includes(to);
}

/**
 * Decide a terminal state from what actually happened. A run that threw is
 * FAILED with the real error attached; a run that reached its step budget is
 * COMPLETED; a run that was stopped is CANCELLED. Training never reports
 * COMPLETED because it did not fail.
 */
export function terminalLifecycle(input: {
  threw: boolean;
  error?: string | null;
  reachedStepBudget: boolean;
  cancelled?: boolean;
}): { state: TrainingLifecycleState; error: string | null; reason: string } {
  if (input.threw) {
    return {
      state: "FAILED",
      error: input.error ?? "training failed with no message recorded",
      reason: "the run raised an error; the actual error is preserved",
    };
  }
  if (input.cancelled) {
    return {
      state: "CANCELLED",
      error: null,
      reason: "the run was cancelled by the caller before its step budget",
    };
  }
  if (input.reachedStepBudget) {
    return { state: "COMPLETED", error: null, reason: "the run reached its configured step budget" };
  }
  return {
    state: "FAILED",
    error: "the run ended without an error but did not reach its step budget",
    reason: "a run that stops early without an error is a failure, not a completion",
  };
}

/** One-line, honest description of what a run actually did. */
export function describeRun(metrics: TokenMetrics, estimate?: ResourceEstimate): string {
  const loss = metrics.lastLoss === null ? "n/a" : metrics.lastLoss.toFixed(4);
  const val = metrics.validationLoss === null ? "n/a" : metrics.validationLoss.toFixed(4);
  const ppl = metrics.validationPerplexity === null ? "n/a" : metrics.validationPerplexity.toFixed(2);
  const bytes = estimate ? ` · checkpoint ~${(estimate.checkpointBytes / 1024 ** 2).toFixed(1)} MiB` : "";
  return (
    `${metrics.state} · ${metrics.steps} step(s) · ${metrics.tokensSeen.toLocaleString()} tokens · ` +
    `${metrics.tokensPerSecond} tok/s · train loss ${loss} · val loss ${val} · ppl ${ppl} · ` +
    `${metrics.checkpointCount} checkpoint(s) · ${metrics.durationMs} ms${bytes}`
  );
}
