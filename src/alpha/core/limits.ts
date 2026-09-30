/**
 * Alpha Core — resource safety.
 *
 * Alpha is deliberately a *small* self-owned model: a few hundred thousand to a
 * few million parameters that can genuinely be trained in a browser tab or on a
 * laptop CPU. These limits exist so that a mistyped configuration fails with a
 * clear message instead of exhausting the host, and so the numbers Alpha prints
 * about itself are honest rather than aspirational.
 *
 * They are ceilings, not targets, and they live in one place so that a larger
 * machine can raise them deliberately (see `docs/model.md`).
 */

import { AlphaValidationError } from "./errors";
import type { AlphaModelConfig } from "../model/config";

export const ALPHA_RESOURCE_LIMITS = {
  /** Longest context Alpha will instantiate or accept at inference. */
  maxContextLength: 2048,
  maxVocabSize: 65536,
  maxLayers: 64,
  maxDModel: 2048,
  /**
   * Hard ceiling on parameter count. Alpha's stack is real but CPU-only: a
   * dense model above this is refused rather than silently started.
   */
  maxParameterCount: 250_000_000,
  maxBatchSize: 64,
  maxSeqLen: 2048,
  maxTotalSteps: 500_000,
  maxNewTokens: 2048,
  maxPromptTokens: 8192,
  maxDocuments: 100_000,
  maxDocumentCharacters: 2_000_000,
  maxDropout: 0.9,
} as const;

export type AlphaResourceKind = keyof typeof ALPHA_RESOURCE_LIMITS;

/**
 * Throws when a value exceeds the configured ceiling. Used by the model
 * config, the dataset pipeline, the trainer and the inference engine.
 */
export function assertResourceLimit(
  kind: AlphaResourceKind,
  value: number,
  context = "",
): void {
  const limit = ALPHA_RESOURCE_LIMITS[kind];
  if (!Number.isFinite(value)) {
    throw new AlphaValidationError("core", `${kind} must be a finite number (received ${value})`, {
      kind,
    });
  }
  if (value > limit) {
    throw new AlphaValidationError(
      "core",
      `${context ? `${context}: ` : ""}${kind} ${value} exceeds Alpha's configured limit of ${limit}. ` +
        "Raise ALPHA_RESOURCE_LIMITS deliberately for a bigger machine — Alpha is sized for small models.",
      { kind, value, limit },
    );
  }
}

/** Raw float32 weight footprint of an architecture, in bytes. */
export function estimateParameterBytes(parameterCount: number): number {
  return parameterCount * 4;
}

export type TrainingMemoryEstimate = {
  parameterCount: number;
  /** Weights. */
  weightsBytes: number;
  /** Gradients, same shape as the weights. */
  gradientBytes: number;
  /** AdamW first + second moments, two extra copies. */
  optimizerBytes: number;
  /**
   * Rough activation footprint for one batch: every layer keeps a handful of
   * [batch, seq, dModel] tensors alive until backward finishes.
   */
  activationBytes: number;
  totalBytes: number;
  note: string;
};

/**
 * Rough peak-memory estimate for one training batch. It is an estimate — the
 * exact figure depends on which intermediates the autodiff graph still holds —
 * which is why the number is reported with that caveat everywhere it appears.
 */
export function estimateTrainingMemory(
  config: AlphaModelConfig,
  training: { batchSize: number; seqLen: number },
): TrainingMemoryEstimate {
  const parameterCount = countParametersFromConfig(config);
  const weightsBytes = parameterCount * 4;
  const gradientBytes = parameterCount * 4;
  const optimizerBytes = parameterCount * 8;
  // ~24 live [batch, seq, dModel] tensors per layer (attention + mlp + norms).
  const activationBytes =
    training.batchSize * training.seqLen * config.dModel * config.nLayers * 24 * 4 +
    training.batchSize * training.seqLen * config.nHeads * training.seqLen * 4;
  return {
    parameterCount,
    weightsBytes,
    gradientBytes,
    optimizerBytes,
    activationBytes,
    totalBytes: weightsBytes + gradientBytes + optimizerBytes + activationBytes,
    note: "Estimate for one optimiser step: weights + gradients + two AdamW moments + live activations. Exact peak usage depends on the autodiff graph and is reported as an estimate, not a measurement.",
  };
}

/**
 * Canonical parameter count for an architecture. It lives here, next to the
 * memory estimates that depend on it, and `model/config` re-exports it — one
 * formula, so the number in the UI and the number in the memory estimate can
 * never drift apart. Only the *type* of the config is imported, so there is no
 * runtime import cycle between the two modules.
 */
export function countParametersFromConfig(config: AlphaModelConfig): number {
  const c = config.dModel;
  let total = config.vocabSize * c;
  if (config.positionalEncoding === "learned") total += config.contextLength * c;
  const perLayer =
    4 * c * c +
    4 * c +
    2 * c * config.dFeedForward +
    config.dFeedForward +
    c +
    4 * c;
  total += perLayer * config.nLayers;
  total += 2 * c;
  if (!config.tieEmbeddings) total += config.vocabSize * c + config.vocabSize;
  return total;
}
