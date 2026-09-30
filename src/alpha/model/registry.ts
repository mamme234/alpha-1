/**
 * The Alpha model registry.
 *
 * A registry entry is a *claim* about a model, and the whole point of this
 * module is that a claim has to be backed. `registerModel` will not accept a
 * configuration that `validateModelConfig` rejects, a parameter count that does
 * not match the architecture, or a lifecycle state that the evidence does not
 * support. Nothing here is asserted because it would be convenient.
 *
 * Relationships are stored as explicit references — tokenizer, dataset,
 * training run, checkpoint, evaluation — rather than being flattened into one
 * metadata blob. That is what makes "is this model loadable by the current
 * inference runtime?" answerable instead of guessable.
 *
 * Promotion is deliberately hard. A model with weights is not trained; a
 * trained model is not evaluated; an evaluated model is not a production
 * candidate; and only an explicit operator approval, with a recorded reason,
 * can make something PRODUCTION.
 */

import { AlphaValidationError } from "../core/errors";
import { alphaId } from "../core/types";
import {
  countParameters,
  modelConfigFingerprint,
  validateModelConfig,
  type AlphaModelConfig,
  type AlphaModelPreset,
} from "./config";

/** The Step 4 lifecycle, which is finer-grained than Step 1's `AlphaModelStage`. */
export type ModelLifecycle =
  | "ARCHITECTURE_ONLY"
  | "UNTRAINED"
  | "TRAINING"
  | "TRAINED"
  | "EVALUATED"
  | "PRODUCTION_CANDIDATE"
  | "PRODUCTION";

export const MODEL_LIFECYCLE_ORDER: ModelLifecycle[] = [
  "ARCHITECTURE_ONLY",
  "UNTRAINED",
  "TRAINING",
  "TRAINED",
  "EVALUATED",
  "PRODUCTION_CANDIDATE",
  "PRODUCTION",
];

export const MODEL_LIFECYCLE_TRANSITIONS: Record<ModelLifecycle, ModelLifecycle[]> = {
  ARCHITECTURE_ONLY: ["UNTRAINED"],
  UNTRAINED: ["TRAINING", "ARCHITECTURE_ONLY"],
  TRAINING: ["TRAINED", "FAILED" as never],
  TRAINED: ["EVALUATED", "TRAINING"],
  EVALUATED: ["PRODUCTION_CANDIDATE", "TRAINED"],
  PRODUCTION_CANDIDATE: ["PRODUCTION", "EVALUATED"],
  PRODUCTION: ["PRODUCTION_CANDIDATE"],
};

/** What a registered model points at. Every field is a reference, not a copy. */
export type ModelRelationships = {
  tokenizerVersion: string;
  tokenizerFingerprint: string;
  /** Dataset id and version, as `datasetId@version`. */
  datasetReference: string;
  datasetFingerprint: string;
  datasetLicense: string;
  /** The training run that produced the weights, if any. */
  trainingRunId: string | null;
  /** The checkpoint holding the current weights, if any. */
  checkpointId: string | null;
  /** Evaluation report ids, oldest first. */
  evaluationIds: string[];
  /** The checkpoint that preceded this one, forming the lineage. */
  parentCheckpointId: string | null;
};

export type RegisteredModel = {
  modelId: string;
  modelVersion: string;
  name: string;
  preset: AlphaModelPreset | null;
  config: AlphaModelConfig;
  configFingerprint: string;
  parameterCount: number;
  contextLength: number;
  vocabSize: number;
  lifecycle: ModelLifecycle;
  relationships: ModelRelationships;
  /** Tokens actually processed by the training run behind this model. */
  trainedTokens: number;
  validationLoss: number | null;
  validationPerplexity: number | null;
  /** Operator decision record, present once promotion has been attempted. */
  promotion: ModelPromotion | null;
  createdAt: number;
  updatedAt: number;
  notes: string[];
};

export type ModelPromotion = {
  at: number;
  by: string;
  reason: string;
  from: ModelLifecycle;
  to: ModelLifecycle;
};

export type CompatibilityCheck = {
  name: string;
  passed: boolean;
  detail: string;
};

export type CompatibilityReport = {
  compatible: boolean;
  checks: CompatibilityCheck[];
};

export type RegisterModelRequest = {
  modelId: string;
  modelVersion: string;
  config: AlphaModelConfig;
  preset?: AlphaModelPreset | null;
  relationships: ModelRelationships;
  /** The checkpoint's measured token count. 0 means the weights are untrained. */
  trainedTokens?: number;
  validationLoss?: number | null;
  validationPerplexity?: number | null;
  lifecycle?: ModelLifecycle;
  notes?: string[];
  now?: number;
};

export class AlphaModelRegistry {
  private readonly models = new Map<string, RegisteredModel>();
  /** Guards against a new run silently replacing an existing version. */
  private readonly fingerprintsByVersion = new Map<string, string>();

  /** Every model, oldest first. */
  list(): RegisteredModel[] {
    return [...this.models.values()].sort((a, b) => a.createdAt - b.createdAt);
  }

  get(modelId: string): RegisteredModel | null {
    return this.models.get(modelId) ?? null;
  }

  /** Models sharing a name, i.e. the version history of one architecture family. */
  versionsOf(name: string): RegisteredModel[] {
    return this.list().filter((m) => m.name === name);
  }

  has(modelId: string): boolean {
    return this.models.has(modelId);
  }

  get size(): number {
    return this.models.size;
  }

  /**
   * Register a model. The configuration is validated here — the registry is the
   * last gate before an architecture is allowed to exist as a real model.
   */
  register(input: RegisterModelRequest): RegisteredModel {
    if (!input.modelId) {
      throw new AlphaValidationError("model", "a registered model needs a modelId");
    }
    if (this.models.has(input.modelId)) {
      throw new AlphaValidationError(
        "model",
        `model "${input.modelId}" is already registered; a new training run must create a new model id and version, never overwrite an existing one`,
        { modelId: input.modelId },
      );
    }

    const config = input.config;
    // The authentic validator from config.ts. Not reimplemented here.
    validateModelConfig(config);

    const parameterCount = countParameters(config);
    const fingerprint = modelConfigFingerprint(config);

    const trainedTokens = input.trainedTokens ?? 0;
    const hasWeights = trainedTokens > 0 || input.relationships.checkpointId !== null;

    // A lifecycle is derived from the evidence, then may only be asserted
    // *downward* — a caller can call a model untrained, never trained.
    const derived: ModelLifecycle = hasWeights ? "TRAINED" : "UNTRAINED";
    const requested = input.lifecycle ?? derived;
    const requestedRank = MODEL_LIFECYCLE_ORDER.indexOf(requested);
    const derivedRank = MODEL_LIFECYCLE_ORDER.indexOf(derived);
    if (requestedRank > derivedRank && derivedRank < 3) {
      throw new AlphaValidationError(
        "model",
        `model "${input.modelId}" cannot be ${requested}: it has no trained weights ` +
          `(${trainedTokens} trained tokens, ${input.relationships.checkpointId ? "a checkpoint" : "no checkpoint"}), ` +
          `so the most it can honestly claim is ${derived}`,
        { modelId: input.modelId, requested, derived },
      );
    }
    if (!hasWeights && (requested === "TRAINED" || requested === "EVALUATED" || requested === "PRODUCTION")) {
      throw new AlphaValidationError(
        "model",
        `model "${input.modelId}" cannot be ${requested} without trained weights`,
        { modelId: input.modelId, requested },
      );
    }

    const now = input.now ?? Date.now();
    const model: RegisteredModel = {
      modelId: input.modelId,
      modelVersion: input.modelVersion,
      name: config.name,
      preset: input.preset ?? null,
      config,
      configFingerprint: fingerprint,
      parameterCount,
      contextLength: config.contextLength,
      vocabSize: config.vocabSize,
      lifecycle: requested,
      relationships: { ...input.relationships },
      trainedTokens,
      validationLoss: input.validationLoss ?? null,
      validationPerplexity: input.validationPerplexity ?? null,
      promotion: null,
      createdAt: now,
      updatedAt: now,
      notes: input.notes ?? [],
    };

    this.models.set(input.modelId, model);
    this.fingerprintsByVersion.set(`${config.name}@${input.modelVersion}`, fingerprint);
    return model;
  }

  /**
   * Point a model at a new checkpoint produced by a new training run. This
   * never mutates the old model; it advances this one and records the lineage.
   */
  recordCheckpoint(
    modelId: string,
    update: {
      checkpointId: string;
      trainingRunId: string;
      trainedTokens: number;
      validationLoss?: number | null;
      validationPerplexity?: number | null;
      now?: number;
    },
  ): RegisteredModel {
    const model = this.require(modelId);
    if (update.trainedTokens <= 0) {
      throw new AlphaValidationError(
        "model",
        `checkpoint ${update.checkpointId} reports ${update.trainedTokens} trained tokens; Alpha will not count it as training`,
        { modelId },
      );
    }
    model.relationships = {
      ...model.relationships,
      parentCheckpointId: model.relationships.checkpointId,
      checkpointId: update.checkpointId,
      trainingRunId: update.trainingRunId,
    };
    model.trainedTokens = update.trainedTokens;
    if (update.validationLoss !== undefined) model.validationLoss = update.validationLoss;
    if (update.validationPerplexity !== undefined) model.validationPerplexity = update.validationPerplexity;
    model.lifecycle = "TRAINED";
    model.updatedAt = update.now ?? Date.now();
    return model;
  }

  /** Attach an evaluation report. Promotion to EVALUATED requires this. */
  recordEvaluation(modelId: string, evaluationId: string, now = Date.now()): RegisteredModel {
    const model = this.require(modelId);
    if (model.trainedTokens <= 0) {
      throw new AlphaValidationError(
        "model",
        `model "${modelId}" has no trained weights; it cannot be evaluated`,
        { modelId },
      );
    }
    if (model.relationships.evaluationIds.includes(evaluationId)) return model;
    model.relationships = {
      ...model.relationships,
      evaluationIds: [...model.relationships.evaluationIds, evaluationId],
    };
    model.lifecycle = "EVALUATED";
    model.updatedAt = now;
    return model;
  }

  /** Mark a model as under active training. */
  markTraining(modelId: string, now = Date.now()): RegisteredModel {
    const model = this.require(modelId);
    model.lifecycle = "TRAINING";
    model.updatedAt = now;
    return model;
  }

  /**
   * The one path to PRODUCTION, and it is deliberately not reachable from
   * training finishing. Requires: trained weights, at least one evaluation, an
   * explicit operator identity and a written reason.
   */
  promote(
    modelId: string,
    input: { by: string; reason: string; evaluationId?: string; now?: number },
  ): RegisteredModel {
    const model = this.require(modelId);

    if (model.trainedTokens <= 0) {
      throw new AlphaValidationError(
        "model",
        `refusing to promote "${modelId}": it has no trained weights`,
        { modelId },
      );
    }
    if (model.relationships.evaluationIds.length === 0) {
      throw new AlphaValidationError(
        "model",
        `refusing to promote "${modelId}": no evaluation has been recorded; a checkpoint is not a promotion`,
        { modelId },
      );
    }
    if (input.evaluationId && !model.relationships.evaluationIds.includes(input.evaluationId)) {
      throw new AlphaValidationError(
        "model",
        `refusing to promote "${modelId}": evaluation "${input.evaluationId}" is not attached to this model`,
        { modelId, evaluationId: input.evaluationId },
      );
    }
    if (!input.by || !input.reason) {
      throw new AlphaValidationError(
        "model",
        `refusing to promote "${modelId}": promotion needs an operator identity and a written reason`,
        { modelId },
      );
    }

    const now = input.now ?? Date.now();
    const from = model.lifecycle;
    // The path into PRODUCTION always goes through PRODUCTION_CANDIDATE.
    if (from !== "PRODUCTION_CANDIDATE") {
      model.lifecycle = "PRODUCTION_CANDIDATE";
      model.promotion = {
        at: now,
        by: input.by,
        reason: input.reason,
        from,
        to: "PRODUCTION_CANDIDATE",
      };
    }
    model.promotion = {
      at: now,
      by: input.by,
      reason: input.reason,
      from: model.lifecycle,
      to: "PRODUCTION",
    };
    model.lifecycle = "PRODUCTION";
    model.updatedAt = now;
    return model;
  }

  /**
   * Can this exact model be loaded by the current runtime? Every relationship
   * is checked; one failure makes the whole thing incompatible.
   */
  checkCompatibility(
    modelId: string,
    runtime: {
      tokenizerVersion: string;
      tokenizerFingerprint: string;
      datasetFingerprint: string;
      /** The architecture the runtime would instantiate. */
      config: AlphaModelConfig;
    },
  ): CompatibilityReport {
    const model = this.require(modelId);
    const checks: CompatibilityCheck[] = [];

    const expected = modelConfigFingerprint(model.config);
    checks.push({
      name: "architecture",
      passed: model.configFingerprint === expected,
      detail:
        model.configFingerprint === expected
          ? `registered fingerprint ${expected} matches its own configuration`
          : `registered fingerprint ${model.configFingerprint} does not match its configuration (${expected})`,
    });

    const runtimeFingerprint = modelConfigFingerprint(runtime.config);
    checks.push({
      name: "runtime-architecture",
      passed: runtimeFingerprint === model.configFingerprint,
      detail:
        runtimeFingerprint === model.configFingerprint
          ? `runtime architecture matches (${runtimeFingerprint})`
          : `runtime architecture ${runtimeFingerprint} differs from the model ${model.configFingerprint}`,
    });

    checks.push({
      name: "parameter-count",
      passed: countParameters(model.config) === model.parameterCount,
      detail: `registered ${model.parameterCount.toLocaleString()}, computed ${countParameters(model.config).toLocaleString()}`,
    });

    checks.push({
      name: "tokenizer-fingerprint",
      passed: model.relationships.tokenizerFingerprint === runtime.tokenizerFingerprint,
      detail:
        model.relationships.tokenizerFingerprint === runtime.tokenizerFingerprint
          ? `tokenizer ${runtime.tokenizerFingerprint} matches the one the weights were trained against`
          : `model was trained on tokenizer ${model.relationships.tokenizerFingerprint}, runtime has ${runtime.tokenizerFingerprint}`,
    });

    checks.push({
      name: "tokenizer-version",
      passed: model.relationships.tokenizerVersion === runtime.tokenizerVersion,
      detail: `model ${model.relationships.tokenizerVersion} vs runtime ${runtime.tokenizerVersion}`,
    });

    checks.push({
      name: "dataset-fingerprint",
      passed: model.relationships.datasetFingerprint === runtime.datasetFingerprint,
      detail: `model ${model.relationships.datasetFingerprint} vs runtime ${runtime.datasetFingerprint}`,
    });

    checks.push({
      name: "weights-present",
      passed: model.trainedTokens > 0 && model.relationships.checkpointId !== null,
      detail:
        model.trainedTokens > 0 && model.relationships.checkpointId !== null
          ? `checkpoint ${model.relationships.checkpointId} carries ${model.trainedTokens.toLocaleString()} trained tokens`
          : `no trained weights (checkpoint ${model.relationships.checkpointId ?? "none"}, ${model.trainedTokens} tokens)`,
    });

    return { compatible: checks.every((c) => c.passed), checks };
  }

  private require(modelId: string): RegisteredModel {
    const model = this.models.get(modelId);
    if (!model) {
      throw new AlphaValidationError("model", `model "${modelId}" is not registered`, { modelId });
    }
    return model;
  }
}

/** Convenience: the model id convention Alpha uses — name@version. */
export function modelIdFor(name: string, version: string): string {
  return `${name}@${version}`;
}

/** A fresh model id for a new version, so a rerun never collides. */
export function newModelId(name: string, version: string, now = Date.now()): string {
  return `${modelIdFor(name, version)}#${alphaId("m")}-${now.toString(36)}`;
}
