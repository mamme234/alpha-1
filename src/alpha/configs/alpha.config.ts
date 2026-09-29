/**
 * Alpha Configs — one configuration object for the whole stack.
 *
 * Everything Alpha can be tuned by lives here: model shape, tokenizer target,
 * training loop, retrieval, memory policy, security budgets and log level. The
 * defaults are the values the workspace actually runs with; the `configs/`
 * directory holds the same shapes as JSON for anyone who prefers files.
 */

import { AlphaValidationError } from "../core/errors";
import {
  ALPHA_MODEL_PRESETS,
  type AlphaModelConfig,
  type AlphaModelPreset,
  validateModelConfig,
} from "../model/config";
import { DEFAULT_SPECIAL_TOKENS, type AlphaSpecialTokens } from "../tokenizer/bpe";
import { DEFAULT_TRAINING_CONFIG, type TrainingConfig } from "../training/trainer";
import { DEFAULT_RAG_CONFIG, type RagConfig } from "../rag/pipeline";
import { DEFAULT_MEMORY_CONFIG, type MemoryStoreConfig } from "../memory/store";
import { DEFAULT_SANDBOX } from "../security/sandbox";
import type { LogLevel } from "../observability/logger";

export type AlphaTokenizerConfig = {
  /** Target vocabulary size for a freshly trained tokenizer. */
  targetVocabSize: number;
  version: string;
  specialTokens: AlphaSpecialTokens;
};

export type AlphaSecurityConfig = {
  approvalTtlMs: number;
  sandbox: {
    maxSteps: number;
    maxGeneratedTokens: number;
    maxDurationMs: number;
    allowNetwork: boolean;
    allowFileSystem: boolean;
  };
  rateLimits: {
    inference: number;
    tool: number;
    agent: number;
    workflow: number;
  };
};

export type AlphaAutomationConfig = {
  maxAttempts: number;
  baseRetryDelayMs: number;
  historyLimit: number;
};

export type AlphaObservabilityConfig = {
  logLevel: LogLevel;
  sampleWindow: number;
  maxSpans: number;
};

export type AlphaConfig = {
  preset: AlphaModelPreset;
  model: AlphaModelConfig;
  tokenizer: AlphaTokenizerConfig;
  training: TrainingConfig;
  rag: RagConfig;
  memory: MemoryStoreConfig;
  security: AlphaSecurityConfig;
  automation: AlphaAutomationConfig;
  observability: AlphaObservabilityConfig;
};

export const DEFAULT_ALPHA_CONFIG: AlphaConfig = {
  preset: "nano",
  model: ALPHA_MODEL_PRESETS.nano,
  tokenizer: {
    targetVocabSize: 384,
    version: "0.1.0",
    specialTokens: DEFAULT_SPECIAL_TOKENS,
  },
  training: DEFAULT_TRAINING_CONFIG,
  rag: DEFAULT_RAG_CONFIG,
  memory: DEFAULT_MEMORY_CONFIG,
  security: {
    approvalTtlMs: 5 * 60_000,
    sandbox: { ...DEFAULT_SANDBOX },
    rateLimits: {
      inference: 60,
      tool: 40,
      agent: 10,
      workflow: 20,
    },
  },
  automation: {
    maxAttempts: 3,
    baseRetryDelayMs: 250,
    historyLimit: 200,
  },
  observability: {
    logLevel: "info",
    sampleWindow: 500,
    maxSpans: 4000,
  },
};

export type AlphaConfigOverrides = {
  preset?: AlphaModelPreset;
  model?: Partial<AlphaModelConfig>;
  tokenizer?: Partial<AlphaTokenizerConfig>;
  training?: Partial<TrainingConfig>;
  rag?: Partial<RagConfig>;
  memory?: Partial<MemoryStoreConfig>;
  security?: {
    approvalTtlMs?: number;
    sandbox?: Partial<AlphaSecurityConfig["sandbox"]>;
    rateLimits?: Partial<AlphaSecurityConfig["rateLimits"]>;
  };
  automation?: Partial<AlphaAutomationConfig>;
  observability?: Partial<AlphaObservabilityConfig>;
};

/** Merge overrides over the defaults (one level deep per section). */
export function createAlphaConfig(overrides: AlphaConfigOverrides = {}): AlphaConfig {
  const preset = overrides.preset ?? DEFAULT_ALPHA_CONFIG.preset;
  const base = {
    ...DEFAULT_ALPHA_CONFIG,
    preset,
    model: { ...ALPHA_MODEL_PRESETS[preset] },
  };
  const config: AlphaConfig = {
    preset,
    model: { ...base.model, ...(overrides.model ?? {}) },
    tokenizer: { ...base.tokenizer, ...(overrides.tokenizer ?? {}) },
    training: { ...base.training, ...(overrides.training ?? {}) },
    rag: { ...base.rag, ...(overrides.rag ?? {}) },
    memory: { ...base.memory, ...(overrides.memory ?? {}) },
    security: {
      approvalTtlMs: overrides.security?.approvalTtlMs ?? base.security.approvalTtlMs,
      sandbox: { ...base.security.sandbox, ...(overrides.security?.sandbox ?? {}) },
      rateLimits: { ...base.security.rateLimits, ...(overrides.security?.rateLimits ?? {}) },
    },
    automation: { ...base.automation, ...(overrides.automation ?? {}) },
    observability: { ...base.observability, ...(overrides.observability ?? {}) },
  };
  validateAlphaConfig(config);
  return config;
}

export function validateAlphaConfig(config: AlphaConfig): void {
  validateModelConfig(config.model);
  if (config.tokenizer.targetVocabSize > config.model.vocabSize) {
    throw new AlphaValidationError(
      "core",
      `tokenizer target vocab (${config.tokenizer.targetVocabSize}) exceeds the model vocab (${config.model.vocabSize})`,
    );
  }
  if (config.training.seqLen > config.model.contextLength) {
    throw new AlphaValidationError(
      "core",
      `training seqLen (${config.training.seqLen}) exceeds context length (${config.model.contextLength})`,
    );
  }
  if (config.rag.chunkTokens > config.model.contextLength) {
    throw new AlphaValidationError(
      "core",
      `rag chunkTokens (${config.rag.chunkTokens}) exceeds context length (${config.model.contextLength})`,
    );
  }
  if (config.security.approvalTtlMs <= 0) {
    throw new AlphaValidationError("core", "approvalTtlMs must be positive");
  }
}

export function serialiseAlphaConfig(config: AlphaConfig): string {
  return JSON.stringify(config, null, 2);
}

export function parseAlphaConfig(json: string): AlphaConfig {
  try {
    const parsed = JSON.parse(json) as AlphaConfigOverrides;
    return createAlphaConfig(parsed);
  } catch (error) {
    throw new AlphaValidationError("core", "configuration file is not valid JSON", {
      cause: error instanceof Error ? error.message : String(error),
    });
  }
}

/** Short human summary used in the workspace header. */
export function describeConfig(config: AlphaConfig): string {
  return `${config.model.name} v${config.model.version} · ${config.model.nLayers}L/${config.model.dModel}d · context ${config.model.contextLength} · vocab ${config.model.vocabSize}`;
}
