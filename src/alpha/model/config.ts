/**
 * Alpha Model — configuration system and versioning.
 *
 * Alpha's model is *Alpha's own* transformer: no weights are downloaded, no
 * provider is called, and there is no hidden dependency on someone else's
 * checkpoint. What a config describes is an *architecture*. Whether the
 * resulting weights are untrained, trained, fine-tuned or production is a
 * separate, explicit fact (`AlphaModelStage`).
 */

import { AlphaValidationError } from "../core/errors";
import type { AlphaModelStage } from "../core/types";

export type PositionalEncodingKind = "learned" | "sinusoidal";

export type AlphaModelConfig = {
  /** Human name, e.g. "alpha-nano". */
  name: string;
  /** Architecture version. Bump whenever shapes or defaults change. */
  version: string;
  vocabSize: number;
  contextLength: number;
  dModel: number;
  nHeads: number;
  nLayers: number;
  dFeedForward: number;
  dropout: number;
  normEps: number;
  positionalEncoding: PositionalEncodingKind;
  /** Share the token embedding matrix with the output projection. */
  tieEmbeddings: boolean;
  /** Std-dev of the normal distribution used for weight initialisation. */
  initStd: number;
};

export type AlphaModelPreset = "nano" | "micro" | "small";

/**
 * Presets sized so the whole stack can actually be trained in a browser tab.
 * These are real architecture sizes, not marketing numbers.
 */
export const ALPHA_MODEL_PRESETS: Record<AlphaModelPreset, AlphaModelConfig> = {
  nano: {
    name: "alpha-nano",
    version: "0.1.0",
    vocabSize: 384,
    contextLength: 64,
    dModel: 64,
    nHeads: 4,
    nLayers: 2,
    dFeedForward: 256,
    dropout: 0.0,
    normEps: 1e-5,
    positionalEncoding: "learned",
    tieEmbeddings: true,
    initStd: 0.02,
  },
  micro: {
    name: "alpha-micro",
    version: "0.1.0",
    vocabSize: 768,
    contextLength: 96,
    dModel: 96,
    nHeads: 6,
    nLayers: 3,
    dFeedForward: 384,
    dropout: 0.05,
    normEps: 1e-5,
    positionalEncoding: "learned",
    tieEmbeddings: true,
    initStd: 0.02,
  },
  small: {
    name: "alpha-small",
    version: "0.1.0",
    vocabSize: 2048,
    contextLength: 128,
    dModel: 128,
    nHeads: 8,
    nLayers: 4,
    dFeedForward: 512,
    dropout: 0.1,
    normEps: 1e-5,
    positionalEncoding: "learned",
    tieEmbeddings: true,
    initStd: 0.02,
  },
};

export function createModelConfig(
  overrides: Partial<AlphaModelConfig> & { preset?: AlphaModelPreset } = {},
): AlphaModelConfig {
  const base = overrides.preset ? ALPHA_MODEL_PRESETS[overrides.preset] : ALPHA_MODEL_PRESETS.nano;
  const { preset: _preset, ...rest } = overrides;
  return { ...base, ...rest };
}

export function validateModelConfig(config: AlphaModelConfig): void {
  if (config.dModel % config.nHeads !== 0) {
    throw new AlphaValidationError(
      "model",
      `dModel (${config.dModel}) must be divisible by nHeads (${config.nHeads})`,
    );
  }
  if (config.nLayers < 1) {
    throw new AlphaValidationError("model", "nLayers must be at least 1");
  }
  if (config.contextLength < 2) {
    throw new AlphaValidationError("model", "contextLength must be at least 2");
  }
  if (config.vocabSize < 2) {
    throw new AlphaValidationError("model", "vocabSize must be at least 2");
  }
  if (config.dFeedForward < config.dModel) {
    throw new AlphaValidationError(
      "model",
      "dFeedForward should be at least dModel for a usable feed-forward block",
    );
  }
  if (config.dropout < 0 || config.dropout >= 1) {
    throw new AlphaValidationError("model", "dropout must be in [0, 1)");
  }
}

/** Exact parameter count for an architecture — computed, never guessed. */
export function countParameters(config: AlphaModelConfig): number {
  const c = config.dModel;
  let total = config.vocabSize * c; // token embedding
  if (config.positionalEncoding === "learned") total += config.contextLength * c;
  const perLayer =
    4 * c * c + // attention projections
    4 * c + // attention biases
    2 * c * config.dFeedForward + // mlp up/down
    config.dFeedForward + // mlp bias 1
    c + // mlp bias 2
    4 * c; // two layer norms
  total += perLayer * config.nLayers;
  total += 2 * c; // final layer norm
  if (!config.tieEmbeddings) total += config.vocabSize * c + config.vocabSize; // output projection
  return total;
}

export type ArchitectureTensorRow = { name: string; shape: string; parameters: number };

/** Parameter table used by the workspace UI and the docs. */
export function describeArchitecture(config: AlphaModelConfig): ArchitectureTensorRow[] {
  const rows: ArchitectureTensorRow[] = [];
  const c = config.dModel;
  rows.push({
    name: "token_embedding",
    shape: `[${config.vocabSize}, ${c}]`,
    parameters: config.vocabSize * c,
  });
  if (config.positionalEncoding === "learned") {
    rows.push({
      name: "position_embedding",
      shape: `[${config.contextLength}, ${c}]`,
      parameters: config.contextLength * c,
    });
  }
  for (let l = 0; l < config.nLayers; l++) {
    rows.push({ name: `layer${l}.attn.wq`, shape: `[${c}, ${c}]`, parameters: c * c });
    rows.push({ name: `layer${l}.attn.wk`, shape: `[${c}, ${c}]`, parameters: c * c });
    rows.push({ name: `layer${l}.attn.wv`, shape: `[${c}, ${c}]`, parameters: c * c });
    rows.push({ name: `layer${l}.attn.wo`, shape: `[${c}, ${c}]`, parameters: c * c });
    rows.push({ name: `layer${l}.attn.bias`, shape: `[${c}] x4`, parameters: 4 * c });
    rows.push({
      name: `layer${l}.mlp.w1`,
      shape: `[${c}, ${config.dFeedForward}]`,
      parameters: c * config.dFeedForward,
    });
    rows.push({
      name: `layer${l}.mlp.w2`,
      shape: `[${config.dFeedForward}, ${c}]`,
      parameters: config.dFeedForward * c,
    });
    rows.push({
      name: `layer${l}.mlp.bias`,
      shape: `[${config.dFeedForward}] + [${c}]`,
      parameters: config.dFeedForward + c,
    });
    rows.push({ name: `layer${l}.norm1`, shape: `[${c}] x2`, parameters: 2 * c });
    rows.push({ name: `layer${l}.norm2`, shape: `[${c}] x2`, parameters: 2 * c });
  }
  rows.push({ name: "final_norm", shape: `[${c}] x2`, parameters: 2 * c });
  if (!config.tieEmbeddings) {
    rows.push({
      name: "output_projection",
      shape: `[${c}, ${config.vocabSize}]`,
      parameters: config.vocabSize * (c + 1),
    });
  }
  return rows;
}

/**
 * A model *artifact* is architecture + weights + provenance. The stage is
 * derived from what actually happened to those weights, never asserted.
 */
export type AlphaModelArtifact = {
  id: string;
  name: string;
  version: string;
  stage: AlphaModelStage;
  config: AlphaModelConfig;
  parameterCount: number;
  createdAt: number;
  /** Checkpoint the weights came from, when the model has been trained. */
  checkpointId?: string;
  /** Tokens seen during the reported training run. */
  trainedTokens?: number;
  /** Last measured validation loss, if any. */
  validationLoss?: number;
  notes: string[];
};

export function deriveStage(input: {
  hasWeights: boolean;
  hasCheckpoint: boolean;
  trainedTokens: number;
  isFineTune: boolean;
  promoted: boolean;
}): AlphaModelStage {
  if (!input.hasWeights) return "architecture";
  if (input.promoted) return "production";
  if (!input.hasCheckpoint || input.trainedTokens === 0) return "untrained";
  if (input.isFineTune) return "fine-tuned";
  return "trained";
}

/**
 * Alpha ships with architecture only. This factory makes that explicit: the
 * returned artifact is labelled `architecture`/`untrained` until a real
 * training run produces a checkpoint.
 */
export function createModelArtifact(config: AlphaModelConfig, id: string): AlphaModelArtifact {
  validateModelConfig(config);
  return {
    id,
    name: config.name,
    version: config.version,
    stage: "architecture",
    config,
    parameterCount: countParameters(config),
    createdAt: Date.now(),
    notes: [
      "Architecture defined and instantiable.",
      "Weights are random initialisation until a training run completes.",
      "No external model weights are loaded at any point.",
    ],
  };
}
