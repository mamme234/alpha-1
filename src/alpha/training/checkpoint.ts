/**
 * Alpha Training Engine — checkpoints.
 *
 * A checkpoint is the unit of "Alpha has actually trained". It carries the
 * weights, the optimiser moments, the RNG position, the step counter and the
 * metrics measured up to that point, which is what makes training resumable
 * instead of restartable.
 *
 * It also carries the *exact* recipe needed to reload it: the architecture, its
 * fingerprint, the tokenizer (version, fingerprint and, when available, the full
 * vocabulary snapshot), the dataset reference and the seed. A checkpoint whose
 * payload does not match its own metadata, or whose architecture does not match
 * the model it is being loaded into, fails with `AlphaCheckpointError` rather
 * than producing undefined behaviour.
 */

import { AlphaCheckpointError, AlphaValidationError } from "../core/errors";
import { alphaId, type AlphaModelStage } from "../core/types";
import type { RngState } from "../core/rng";
import { base64ByteLength, base64ToFloat32 } from "../core/serialize";
import {
  modelConfigFingerprint,
  validateModelConfig,
  type AlphaModelConfig,
  type AlphaSpecialTokenIds,
} from "../model/config";
import type { SerializedWeights } from "../model/transformer";
import type { AlphaTokenizerSnapshot } from "../tokenizer/bpe";
import type { OptimizerStateSnapshot } from "./optimizer";
import type { TrainingConfig } from "./trainer";

/** Bump when the checkpoint payload changes shape. */
export const ALPHA_CHECKPOINT_FORMAT_VERSION = "1.0.0";

export type AlphaCheckpointMetrics = {
  trainLoss: number;
  validationLoss: number | null;
  validationPerplexity: number | null;
  /** Uniform-predictor baseline measured at the same vocabulary size. */
  uniformLoss: number | null;
};

/** Everything needed to rebuild the tokenizer that produced this checkpoint. */
export type CheckpointTokenizerRef = {
  version: string;
  vocabSize: number;
  /** FNV-1a fingerprint of the vocabulary and merge table. */
  fingerprint: string;
  trainedOn: string;
  specialTokenIds: AlphaSpecialTokenIds;
  /**
   * Full vocabulary snapshot. Present for checkpoints written by this version
   * of Alpha, so a checkpoint alone is enough to restore the tokenizer.
   */
  snapshot: AlphaTokenizerSnapshot | null;
};

export type AlphaCheckpoint = {
  id: string;
  label: string;
  /** Checkpoint payload version. */
  formatVersion: string;
  /** Identifier of the training run that produced this checkpoint. */
  runId: string;
  /** Seed that drives sampling, dropout and window order for the run. */
  seed: number;
  /**
   * The training configuration the run was started with. Resuming uses this
   * rather than whatever the workspace happens to default to, so a resumed run
   * continues the same schedule toward the same step budget.
   */
  trainingConfig: TrainingConfig;
  modelName: string;
  modelVersion: string;
  /** Fingerprint of `config` — recomputed and compared on validation. */
  configFingerprint: string;
  config: AlphaModelConfig;
  tokenizer: CheckpointTokenizerRef;
  /** Convenience mirror of `tokenizer.version` for lists and UI. */
  tokenizerVersion: string;
  datasetName: string;
  datasetVersion: string;
  datasetFingerprint: string;
  datasetLicense: string;
  /** Optimiser steps completed when this checkpoint was written. */
  step: number;
  tokensSeen: number;
  learningRate: number;
  metrics: AlphaCheckpointMetrics;
  /** Base64 float32 payload per parameter name. */
  weights: SerializedWeights;
  optimizer: OptimizerStateSnapshot;
  rng: RngState;
  createdAt: number;
  sizeBytes: number;
  /** Stage implied purely by the existence of this checkpoint. */
  stage: AlphaModelStage;
  /** True when the run continued from an earlier checkpoint of a trained model. */
  isFineTune: boolean;
  notes: string[];
};

export type CreateCheckpointInput = {
  label: string;
  modelName: string;
  modelVersion: string;
  config: AlphaModelConfig;
  tokenizer: CheckpointTokenizerRef;
  datasetName: string;
  datasetVersion: string;
  datasetFingerprint: string;
  datasetLicense: string;
  step: number;
  tokensSeen: number;
  learningRate: number;
  metrics: AlphaCheckpointMetrics;
  weights: SerializedWeights;
  optimizer: OptimizerStateSnapshot;
  rng: RngState;
  runId: string;
  seed: number;
  trainingConfig: TrainingConfig;
  /** True when the run continued from an earlier checkpoint of a trained model. */
  isFineTune?: boolean;
  id?: string;
  createdAt?: number;
};

/** Rough payload size in bytes, computed from the encoded weights. */
export function estimateCheckpointBytes(weights: SerializedWeights): number {
  let bytes = 0;
  for (const encoded of Object.values(weights.tensors)) bytes += base64ByteLength(encoded);
  return bytes;
}

export function createCheckpoint(input: CreateCheckpointInput): AlphaCheckpoint {
  const sizeBytes = estimateCheckpointBytes(input.weights);
  return {
    id: input.id ?? alphaId("ckpt"),
    label: input.label,
    formatVersion: ALPHA_CHECKPOINT_FORMAT_VERSION,
    runId: input.runId,
    seed: input.seed,
    trainingConfig: { ...input.trainingConfig },
    modelName: input.modelName,
    modelVersion: input.modelVersion,
    configFingerprint: modelConfigFingerprint(input.config),
    config: input.config,
    tokenizer: input.tokenizer,
    tokenizerVersion: input.tokenizer.version,
    datasetName: input.datasetName,
    datasetVersion: input.datasetVersion,
    datasetFingerprint: input.datasetFingerprint,
    datasetLicense: input.datasetLicense,
    step: input.step,
    tokensSeen: input.tokensSeen,
    learningRate: input.learningRate,
    metrics: input.metrics,
    weights: input.weights,
    optimizer: input.optimizer,
    rng: input.rng,
    createdAt: input.createdAt ?? Date.now(),
    sizeBytes,
    stage: input.step > 0 ? (input.isFineTune ? "fine-tuned" : "trained") : "untrained",
    isFineTune: input.isFineTune ?? false,
    notes: [
      `Trained from scratch by the Alpha training engine in ${input.step} optimiser steps.`,
      `Tokens seen: ${input.tokensSeen.toLocaleString()}.`,
      `Corpus: ${input.datasetName}@${input.datasetVersion} (${input.datasetLicense}).`,
      `Tokenizer: ${input.tokenizer.version} (${input.tokenizer.fingerprint}, ${input.tokenizer.vocabSize} tokens).`,
      `Run ${input.runId}, seed ${input.seed}.`,
      "No external model weights were used at any point.",
    ],
  };
}

export function checkpointToJson(checkpoint: AlphaCheckpoint, options: { pretty?: boolean } = {}): string {
  return options.pretty ? JSON.stringify(checkpoint, null, 2) : JSON.stringify(checkpoint);
}

/**
 * Parse a stored checkpoint. Structural problems are reported as an
 * `AlphaCheckpointError`; call `validateCheckpoint` for the deep check.
 */
export function parseCheckpoint(json: string): AlphaCheckpoint {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch (error) {
    throw new AlphaCheckpointError("checkpoint payload is not valid JSON", [
      error instanceof Error ? error.message : String(error),
    ]);
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new AlphaCheckpointError("checkpoint payload is not an object");
  }
  const candidate = parsed as Partial<AlphaCheckpoint>;
  const missing = (["id", "config", "weights", "optimizer", "rng", "step"] as const).filter(
    (key) => candidate[key] === undefined,
  );
  if (missing.length > 0) {
    throw new AlphaCheckpointError("checkpoint is missing required fields", missing.map(String));
  }
  return candidate as AlphaCheckpoint;
}

export type CheckpointValidation = {
  valid: boolean;
  issues: string[];
};

/**
 * Deep validation: metadata against payload, payload against architecture,
 * base64 lengths against shapes, and stage against step count.
 */
export function validateCheckpoint(checkpoint: AlphaCheckpoint): CheckpointValidation {
  const issues: string[] = [];
  const fail = (message: string) => issues.push(message);

  if (!checkpoint.id?.trim()) fail("id is empty");
  if (!checkpoint.label?.trim()) fail("label is empty");
  if (!checkpoint.runId?.trim()) fail("runId is empty");
  if (!Number.isInteger(checkpoint.step) || checkpoint.step < 0) {
    fail(`step must be a non-negative integer (received ${checkpoint.step})`);
  }
  if (!Number.isFinite(checkpoint.tokensSeen) || checkpoint.tokensSeen < 0) {
    fail(`tokensSeen must be non-negative (received ${checkpoint.tokensSeen})`);
  }
  if (!Number.isFinite(checkpoint.seed)) fail("seed is not a finite number");
  if (!Number.isFinite(checkpoint.learningRate)) fail("learningRate is not finite");
  const training = checkpoint.trainingConfig;
  if (!training) {
    fail("the training configuration snapshot is missing");
  } else {
    if (!Number.isInteger(training.batchSize) || training.batchSize < 1) fail("training snapshot: batchSize is invalid");
    if (!Number.isInteger(training.seqLen) || training.seqLen < 2) fail("training snapshot: seqLen is invalid");
    if (!Number.isInteger(training.totalSteps) || training.totalSteps < 1) {
      fail("training snapshot: totalSteps is invalid");
    }
    if (training.seed !== checkpoint.seed) {
      fail(`training snapshot seed ${training.seed} does not match the checkpoint seed ${checkpoint.seed}`);
    }
    if (training.seqLen > checkpoint.config.contextLength) {
      fail(
        `training snapshot seqLen ${training.seqLen} exceeds the checkpoint context length ${checkpoint.config.contextLength}`,
      );
    }
  }

  try {
    validateModelConfig(checkpoint.config);
  } catch (error) {
    fail(`config is invalid: ${error instanceof Error ? error.message : String(error)}`);
  }

  if (checkpoint.configFingerprint !== modelConfigFingerprint(checkpoint.config)) {
    fail(
      `config fingerprint ${checkpoint.configFingerprint} does not match the config payload (${modelConfigFingerprint(checkpoint.config)})`,
    );
  }

  // --- weights against shapes and architecture -----------------------------
  const tensors = checkpoint.weights?.tensors ?? {};
  const shapes = checkpoint.weights?.shapes ?? {};
  const names = Object.keys(tensors);
  if (names.length === 0) fail("weights payload contains no tensors");
  for (const name of names) {
    const shape = shapes[name];
    const encoded = tensors[name];
    if (!shape || !Array.isArray(shape) || shape.length === 0) {
      fail(`weights: missing shape for ${name}`);
      continue;
    }
    const expected = shape.reduce((a, b) => a * b, 1) * 4;
    if (base64ByteLength(encoded) !== expected) {
      fail(`weights: ${name} payload is ${base64ByteLength(encoded)} bytes, expected ${expected}`);
      continue;
    }
    try {
      const values = base64ToFloat32(encoded);
      if (values.length !== expected / 4) fail(`weights: ${name} decoded to ${values.length} floats`);
      for (let i = 0; i < Math.min(values.length, 32); i++) {
        if (!Number.isFinite(values[i])) {
          fail(`weights: ${name} contains a non-finite value`);
          break;
        }
      }
    } catch (error) {
      fail(`weights: ${name} is not valid base64 (${error instanceof Error ? error.message : String(error)})`);
    }
  }
  if (checkpoint.weights?.config && checkpoint.weights.config.vocabSize !== checkpoint.config.vocabSize) {
    fail("weights payload was serialised from a different vocabulary size than the checkpoint config");
  }

  // --- optimiser ----------------------------------------------------------
  const optimizer = checkpoint.optimizer;
  if (!optimizer || !Number.isInteger(optimizer.step) || optimizer.step < 0) {
    fail("optimizer state is missing or has an invalid step");
  } else {
    if (optimizer.step !== checkpoint.step) {
      fail(`optimizer step ${optimizer.step} does not match checkpoint step ${checkpoint.step}`);
    }
    for (const name of names) {
      const first = optimizer.firstMoment?.[name];
      const second = optimizer.secondMoment?.[name];
      if (!first || !second) {
        fail(`optimizer: missing Adam moments for ${name}`);
        continue;
      }
      const expected = shapes[name].reduce((a, b) => a * b, 1) * 4;
      if (base64ByteLength(first) !== expected || base64ByteLength(second) !== expected) {
        fail(`optimizer: ${name} moments have the wrong payload size`);
      }
    }
  }

  // --- rng ----------------------------------------------------------------
  if (!checkpoint.rng || !Number.isFinite(checkpoint.rng.seed) || checkpoint.rng.calls < 0) {
    fail("rng state is missing or invalid");
  }

  // --- tokenizer ----------------------------------------------------------
  const tokenizer = checkpoint.tokenizer;
  if (!tokenizer) {
    fail("tokenizer reference is missing");
  } else {
    if (!tokenizer.version?.trim()) fail("tokenizer version is empty");
    if (!tokenizer.fingerprint?.trim()) fail("tokenizer fingerprint is empty");
    if (tokenizer.vocabSize < 4) fail(`tokenizer vocabSize ${tokenizer.vocabSize} is implausible`);
    if (tokenizer.vocabSize > checkpoint.config.vocabSize) {
      fail(
        `tokenizer vocabulary (${tokenizer.vocabSize}) is larger than the model vocabulary (${checkpoint.config.vocabSize})`,
      );
    }
    const ids = tokenizer.specialTokenIds;
    if (ids) {
      const values = Object.values(ids);
      if (values.some((id) => !Number.isInteger(id) || id < 0 || id >= tokenizer.vocabSize)) {
        fail("tokenizer special token ids are outside the vocabulary");
      }
      if (checkpoint.config.specialTokenIds) {
        for (const key of ["pad", "unk", "bos", "eos"] as const) {
          if (checkpoint.config.specialTokenIds[key] !== ids[key]) {
            fail(`config and checkpoint tokenizer disagree on the ${key} token id`);
          }
        }
      }
    } else {
      fail("tokenizer special token ids are missing");
    }
    if (tokenizer.snapshot && tokenizer.snapshot.tokens.length !== tokenizer.vocabSize) {
      fail("tokenizer snapshot size does not match the recorded vocabulary size");
    }
  }

  // --- metrics and stage --------------------------------------------------
  const metrics = checkpoint.metrics;
  if (!metrics) {
    fail("metrics are missing");
  } else {
    if (checkpoint.step > 0 && !Number.isFinite(metrics.trainLoss)) {
      fail("a trained checkpoint must record a finite training loss");
    }
    if (metrics.validationLoss !== null && !Number.isFinite(metrics.validationLoss)) {
      fail("validationLoss must be a finite number or null");
    }
    if (metrics.validationPerplexity !== null && !Number.isFinite(metrics.validationPerplexity)) {
      fail("validationPerplexity must be a finite number or null");
    }
  }
  if (checkpoint.step > 0 && checkpoint.stage === "untrained") {
    fail("stage is untrained but the checkpoint has recorded optimiser steps");
  }
  if (checkpoint.step === 0 && checkpoint.stage !== "untrained") {
    fail(`stage is ${checkpoint.stage} but no optimiser step has been taken`);
  }
  if (checkpoint.optimizer && checkpoint.optimizer.step > 0 && !checkpoint.isFineTune && checkpoint.stage === "untrained") {
    fail("optimiser has steps but the checkpoint is labelled untrained");
  }

  const expectedSize = estimateCheckpointBytes(checkpoint.weights ?? { config: checkpoint.config, tensors: {}, shapes: {} });
  if (checkpoint.sizeBytes !== expectedSize) {
    fail(`sizeBytes ${checkpoint.sizeBytes} does not match the weight payload (${expectedSize})`);
  }

  return { valid: issues.length === 0, issues };
}

export function assertValidCheckpoint(checkpoint: AlphaCheckpoint): void {
  const { valid, issues } = validateCheckpoint(checkpoint);
  if (!valid) {
    throw new AlphaCheckpointError(
      `checkpoint ${checkpoint?.id ?? "(unknown)"} failed validation with ${issues.length} problem(s)`,
      issues,
    );
  }
}

export type CheckpointCompatibility = {
  compatible: boolean;
  reasons: string[];
};

/** Compare two checkpoints: same architecture, tokenizer and corpus reference. */
export function compareCheckpoints(a: AlphaCheckpoint, b: AlphaCheckpoint): CheckpointCompatibility {
  const reasons: string[] = [];
  if (a.configFingerprint !== b.configFingerprint) {
    reasons.push(`architecture differs (${a.configFingerprint} vs ${b.configFingerprint})`);
  }
  if (a.tokenizer.fingerprint !== b.tokenizer.fingerprint) {
    reasons.push(`tokenizer differs (${a.tokenizer.fingerprint} vs ${b.tokenizer.fingerprint})`);
  }
  if (a.datasetFingerprint !== b.datasetFingerprint) {
    reasons.push("corpus differs — the run would not be comparable");
  }
  return { compatible: reasons.length === 0, reasons };
}

/**
 * Prove that a checkpoint can be loaded into a given model and tokenizer.
 * This is the gate `resumeFrom` uses; it is deliberately stricter than
 * "same vocab size", because shape-compatible weights can still be meaningless.
 */
export function assertCheckpointCompatible(
  checkpoint: AlphaCheckpoint,
  target: { config: AlphaModelConfig; tokenizer: { fingerprint(): string; vocabSize: number; version: string } },
): void {
  assertValidCheckpoint(checkpoint);
  const reasons: string[] = [];
  const configReason = assertConfigMatches(checkpoint.config, target.config);
  if (configReason) reasons.push(configReason);
  if (checkpoint.tokenizer.fingerprint !== target.tokenizer.fingerprint()) {
    reasons.push(
      `checkpoint tokenizer ${checkpoint.tokenizer.fingerprint} (v${checkpoint.tokenizer.version}) does not match the workspace tokenizer ${target.tokenizer.fingerprint()} (v${target.tokenizer.version})`,
    );
  }
  if (checkpoint.tokenizer.vocabSize !== target.tokenizer.vocabSize) {
    reasons.push(
      `checkpoint vocabulary (${checkpoint.tokenizer.vocabSize}) does not match the tokenizer (${target.tokenizer.vocabSize})`,
    );
  }
  if (reasons.length > 0) {
    throw new AlphaCheckpointError(
      `checkpoint ${checkpoint.id} is not compatible with this workspace`,
      reasons,
      { checkpointId: checkpoint.id },
    );
  }
}

function assertConfigMatches(a: AlphaModelConfig, b: AlphaModelConfig): string | null {
  const keys: (keyof AlphaModelConfig)[] = [
    "vocabSize",
    "contextLength",
    "dModel",
    "nHeads",
    "nLayers",
    "dFeedForward",
    "positionalEncoding",
    "tieEmbeddings",
  ];
  const differing = keys.filter((key) => a[key] !== b[key]);
  if (differing.length === 0) return null;
  return `architecture differs on ${differing.map((key) => `${String(key)} (${String(a[key])} ≠ ${String(b[key])})`).join(", ")}`;
}

/** Compact summary for lists and the workspace UI (drops the weight payload). */
export type AlphaCheckpointSummary = Omit<
  AlphaCheckpoint,
  "weights" | "optimizer" | "rng" | "tokenizer"
> & {
  tokenizer: Omit<CheckpointTokenizerRef, "snapshot">;
  /** Tokenizer snapshot omitted; the fingerprint is enough to identify it. */
  tokenizerSnapshotIncluded: boolean;
};

export function summariseCheckpoint(checkpoint: AlphaCheckpoint): AlphaCheckpointSummary {
  const { weights: _weights, optimizer: _optimizer, rng: _rng, tokenizer, ...rest } = checkpoint;
  const { snapshot, ...tokenizerRef } = tokenizer;
  return {
    ...rest,
    tokenizer: tokenizerRef,
    tokenizerSnapshotIncluded: snapshot !== null,
  };
}

/** Convenience: a checkpoint built from a planner function, used by tests. */
export function withCheckpointId(checkpoint: AlphaCheckpoint, id: string): AlphaCheckpoint {
  if (!id.trim()) throw new AlphaValidationError("training", "checkpoint id must not be empty");
  return { ...checkpoint, id };
}
