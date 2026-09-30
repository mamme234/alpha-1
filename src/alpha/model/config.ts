/**
 * Alpha Model Configuration.
 *
 * This module owns everything about the *shape* of an Alpha model: the
 * presets, the validation rules, the parameter-count arithmetic, the
 * architecture table the UI renders, and the fingerprint that decides whether
 * a checkpoint may be loaded into a model at all.
 *
 * The parameter count is never a claim in a table. It is computed from the
 * same arithmetic the transformer actually allocates against, and the canonical
 * copy of that formula lives in `core/limits` next to the memory estimates
 * that depend on it, then is re-exported here — so the number the dashboard
 * shows and the number the memory estimate uses cannot drift apart.
 */

import { AlphaValidationError } from "../core/errors";
import {
  ALPHA_RESOURCE_LIMITS,
  assertResourceLimit,
  countParametersFromConfig,
} from "../core/limits";
import type { AlphaModelStage } from "../core/types";

/** How positions reach the model. `learned` allocates a table; `sinusoidal` does not. */
export type PositionalEncodingKind = "learned" | "sinusoidal";

/** The four ids a tokenizer reserves, recorded on the config for compatibility checks. */
export type AlphaSpecialTokenIds = {
  pad: number;
  unk: number;
  bos: number;
  eos: number;
};

export type AlphaModelConfig = {
  name: string;
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
  tieEmbeddings: boolean;
  initStd: number;
  /** Ids of pad/unk/bos/eos, when a trained tokenizer has been bound. */
  specialTokenIds?: AlphaSpecialTokenIds;
};

export type AlphaModelPreset = "nano" | "micro" | "small";

/**
 * One row of the architecture table: a tensor Alpha really allocates, its real
 * shape, and how many parameters it contributes. The rows sum to
 * `countParameters`, which is checked against the instantiated model's own
 * parameter list in the test suite.
 */
export type ArchitectureTensorRow = {
  name: string;
  shape: string;
  parameters: number;
};

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
    dropout: 0,
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

/**
 * Validate a model configuration, or throw with a message that says which
 * field is wrong. Called by the transformer constructor, by config parsing and
 * again when a checkpoint is loaded, so an invalid shape can never reach an
 * allocation.
 */
export function validateModelConfig(config: AlphaModelConfig): void {
  const fail = (message: string, details?: Record<string, unknown>): never => {
    throw new AlphaValidationError("model", message, details);
  };

  if (!config.name || typeof config.name !== "string") fail("model.name is required");
  if (!config.version || typeof config.version !== "string") fail("model.version is required");

  const positiveIntegers: Array<[string, number]> = [
    ["vocabSize", config.vocabSize],
    ["contextLength", config.contextLength],
    ["dModel", config.dModel],
    ["nHeads", config.nHeads],
    ["nLayers", config.nLayers],
    ["dFeedForward", config.dFeedForward],
  ];
  for (const [field, value] of positiveIntegers) {
    if (!Number.isInteger(value) || value <= 0) {
      fail(`model.${field} must be a positive integer (received ${value})`, { field, value });
    }
  }

  if (config.dModel % config.nHeads !== 0) {
    fail(
      `model.dModel ${config.dModel} must be divisible by nHeads ${config.nHeads} (received ${config.dModel / config.nHeads} per head)`,
      { dModel: config.dModel, nHeads: config.nHeads },
    );
  }

  if (!Number.isFinite(config.normEps) || config.normEps <= 0) {
    fail(`model.normEps must be a positive number (received ${config.normEps})`, {
      normEps: config.normEps,
    });
  }

  if (!Number.isFinite(config.initStd) || config.initStd <= 0) {
    fail(`model.initStd must be a positive number (received ${config.initStd})`, {
      initStd: config.initStd,
    });
  }

  if (config.positionalEncoding !== "learned" && config.positionalEncoding !== "sinusoidal") {
    fail(
      `model.positionalEncoding must be "learned" or "sinusoidal" (received ${String(config.positionalEncoding)})`,
      { positionalEncoding: config.positionalEncoding },
    );
  }

  if (config.specialTokenIds) {
    const ids = config.specialTokenIds;
    for (const key of ["pad", "unk", "bos", "eos"] as const) {
      const id = ids[key];
      if (!Number.isInteger(id) || id < 0 || id >= config.vocabSize) {
        fail(
          `model.specialTokenIds.${key} ${String(id)} is outside the vocabulary [0, ${config.vocabSize})`,
          { key, id, vocabSize: config.vocabSize },
        );
      }
    }
  }

  // Resource ceilings. Alpha is sized for small models on modest hardware, so
  // an oversized architecture is refused up front rather than started and
  // discovered to be impossible later.
  assertResourceLimit("maxContextLength", config.contextLength, "model");
  assertResourceLimit("maxVocabSize", config.vocabSize, "model");
  assertResourceLimit("maxLayers", config.nLayers, "model");
  assertResourceLimit("maxDModel", config.dModel, "model");
  assertResourceLimit("maxDropout", config.dropout, "model");
  assertResourceLimit("maxParameterCount", countParameters(config), "model");
}

/**
 * Build a config from a preset plus overrides and validate the result. Used by
 * the config layer and by anything that wants a shape without hand-writing one.
 */
export function createModelConfig(
  preset: AlphaModelPreset,
  overrides: Partial<AlphaModelConfig> = {},
): AlphaModelConfig {
  const base = ALPHA_MODEL_PRESETS[preset];
  if (!base) {
    throw new AlphaValidationError(
      "model",
      `unknown model preset "${String(preset)}" (expected one of ${Object.keys(ALPHA_MODEL_PRESETS).join(", ")})`,
      { preset },
    );
  }
  const config: AlphaModelConfig = { ...base, ...overrides };
  validateModelConfig(config);
  return config;
}

/**
 * Bind a trained tokenizer's special-token ids onto a model config. The result
 * records which ids the weights were trained against, which is what lets a
 * checkpoint be refused later if the tokenizer has moved underneath it.
 */
export function withSpecialTokenIds(
  config: AlphaModelConfig,
  ids: AlphaSpecialTokenIds,
): AlphaModelConfig {
  return { ...config, specialTokenIds: { pad: ids.pad, unk: ids.unk, bos: ids.bos, eos: ids.eos } };
}

/**
 * The canonical parameter count for an architecture. Re-exported from
 * `core/limits` so there is exactly one formula: the number in the dashboard,
 * the number in the memory estimate and the number the transformer allocates
 * against are the same value by construction.
 */
export const countParameters: (config: AlphaModelConfig) => number = countParametersFromConfig;

/**
 * The tensors Alpha actually allocates for this architecture, in allocation
 * order, with their real names, shapes and contribution to the parameter
 * count. The rows sum to `countParameters(config)`.
 */
export function describeArchitecture(config: AlphaModelConfig): ArchitectureTensorRow[] {
  const c = config.dModel;
  const f = config.dFeedForward;
  const rows: ArchitectureTensorRow[] = [];

  const row = (name: string, shape: string, parameters: number) => {
    rows.push({ name, shape, parameters });
  };

  row("token_embedding", `[${config.vocabSize}, ${c}]`, config.vocabSize * c);

  if (config.positionalEncoding === "learned") {
    row("position_embedding", `[${config.contextLength}, ${c}]`, config.contextLength * c);
  }

  for (let l = 0; l < config.nLayers; l++) {
    row(`layer${l}.attn.wq`, `[${c}, ${c}]`, c * c);
    row(`layer${l}.attn.wk`, `[${c}, ${c}]`, c * c);
    row(`layer${l}.attn.wv`, `[${c}, ${c}]`, c * c);
    row(`layer${l}.attn.wo`, `[${c}, ${c}]`, c * c);
    row(`layer${l}.attn.bq`, `[${c}]`, c);
    row(`layer${l}.attn.bk`, `[${c}]`, c);
    row(`layer${l}.attn.bv`, `[${c}]`, c);
    row(`layer${l}.attn.bo`, `[${c}]`, c);
    row(`layer${l}.norm1.weight`, `[${c}]`, c);
    row(`layer${l}.norm1.bias`, `[${c}]`, c);
    row(`layer${l}.mlp.w1`, `[${c}, ${f}]`, c * f);
    row(`layer${l}.mlp.b1`, `[${f}]`, f);
    row(`layer${l}.mlp.w2`, `[${f}, ${c}]`, f * c);
    row(`layer${l}.mlp.b2`, `[${c}]`, c);
    row(`layer${l}.norm2.weight`, `[${c}]`, c);
    row(`layer${l}.norm2.bias`, `[${c}]`, c);
  }

  row("final_norm.weight", `[${c}]`, c);
  row("final_norm.bias", `[${c}]`, c);

  if (!config.tieEmbeddings) {
    row("output_projection.weight", `[${c}, ${config.vocabSize}]`, c * config.vocabSize);
    row("output_projection.bias", `[${config.vocabSize}]`, config.vocabSize);
  }

  return rows;
}

/**
 * Derive the model stage from what actually exists, never from what was
 * intended. A model with weights but no checkpoint has still never been
 * trained, and only an explicitly promoted model is `production`.
 */
export function deriveStage(input: {
  hasWeights: boolean;
  hasCheckpoint: boolean;
  trainedTokens: number;
  isFineTune: boolean;
  promoted: boolean;
}): AlphaModelStage {
  if (input.promoted) return "production";
  if (input.hasCheckpoint && input.trainedTokens > 0) {
    return input.isFineTune ? "fine-tuned" : "trained";
  }
  if (input.hasWeights) return "untrained";
  return "architecture";
}

/**
 * A stable fingerprint of the architecture. Any change to a field that changes
 * the parameter layout changes this value, which is how a checkpoint is
 * refused against a config it was not trained with.
 */
export function modelConfigFingerprint(config: AlphaModelConfig): string {
  let hash = 0x811c9dc5;
  const feed = (text: string) => {
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
  };
  feed(config.name);
  feed("|");
  feed(config.version);
  feed("|");
  feed(String(config.vocabSize));
  feed(String(config.contextLength));
  feed(String(config.dModel));
  feed(String(config.nHeads));
  feed(String(config.nLayers));
  feed(String(config.dFeedForward));
  feed(config.dropout.toString());
  feed(config.normEps.toString());
  feed(config.positionalEncoding);
  feed(config.tieEmbeddings ? "tied" : "untied");
  feed(config.initStd.toString());
  if (config.specialTokenIds) {
    const ids = config.specialTokenIds;
    feed(`${ids.pad}.${ids.unk}.${ids.bos}.${ids.eos}`);
  }
  return `cfg_${hash.toString(16).padStart(8, "0")}`;
}

/** A loadable model description: the config, its real size, and where it stands. */
export type AlphaModelArtifact = {
  id: string;
  name: string;
  version: string;
  config: AlphaModelConfig;
  configFingerprint: string;
  parameterCount: number;
  stage: AlphaModelStage;
  checkpointId?: string;
  trainedTokens?: number;
  validationLoss?: number;
  createdAt: number;
  notes: string[];
};

/**
 * Create the artifact description for a freshly instantiated model. This is a
 * description, not a weight dump: `currentArtifact()` in the workspace layers
 * the stage, checkpoint and measured loss on top of it from the live state.
 */
export function createModelArtifact(config: AlphaModelConfig, id: string): AlphaModelArtifact {
  return {
    id,
    name: config.name,
    version: config.version,
    config,
    configFingerprint: modelConfigFingerprint(config),
    parameterCount: countParameters(config),
    stage: "architecture",
    createdAt: Date.now(),
    notes: [],
  };
}

/** The resource ceiling the presets are checked against, re-exported for callers. */
export { ALPHA_RESOURCE_LIMITS };
