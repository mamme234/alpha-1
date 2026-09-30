/**
 * Model export and import.
 *
 * An exported Alpha model is the whole story of how a set of weights came to
 * be: the architecture, the tokenizer that defines its vocabulary, the dataset
 * version it was trained on, the run and checkpoint that produced it, and the
 * evaluation that measured it. Weights without that context cannot be
 * reproduced or trusted, so the format makes it all one object.
 *
 * Two things this format deliberately does *not* do:
 *
 *   - It does not claim a model is production. The lifecycle state travels with
 *     the artifact exactly as the registry recorded it, and importing does not
 *     promote anything.
 *   - It does not carry foreign weights. Only tensors Alpha produced, alongside
 *     the config that describes them.
 */

import { AlphaValidationError } from "../core/errors";
import { alphaId } from "../core/types";
import { countParameters, modelConfigFingerprint, validateModelConfig, type AlphaModelConfig } from "./config";
import { AlphaTransformer, type SerializedWeights } from "./transformer";
import type { ModelLifecycle, RegisteredModel } from "./registry";
import { ALPHA_CHECKPOINT_FORMAT_VERSION, type AlphaCheckpoint } from "../training/checkpoint";

export const ALPHA_MODEL_EXPORT_VERSION = "1.0.0";

export type AlphaModelExport = {
  format: "alpha.model-export";
  formatVersion: string;
  createdAt: number;

  modelId: string;
  modelVersion: string;
  name: string;
  /** The lifecycle as recorded. Import never changes it. */
  lifecycle: ModelLifecycle;
  /** The registry entry this artifact was produced from, minus the weights. */
  model: RegisteredModel;
  config: AlphaModelConfig;
  configFingerprint: string;
  parameterCount: number;

  /** Weights, exactly as Alpha serialises them. */
  weights: SerializedWeights;

  /** The checkpoint the weights came from, or null for an untrained export. */
  checkpoint: AlphaCheckpoint | null;

  /**
   * Everything needed to reproduce the run, flattened so the artifact is
   * readable without the registry.
   */
  provenance: {
    tokenizerVersion: string;
    tokenizerFingerprint: string;
    datasetReference: string;
    datasetFingerprint: string;
    datasetLicense: string;
    trainingRunId: string | null;
    checkpointId: string | null;
    parentCheckpointId: string | null;
    seed: number | null;
    trainedTokens: number;
    validationLoss: number | null;
    validationPerplexity: number | null;
    evaluationIds: string[];
    /** Reproduces a run: config + data + tokenizer + seed + optimisation. */
    reproducibility: {
      configFingerprint: string;
      datasetFingerprint: string;
      tokenizerFingerprint: string;
      seed: number | null;
      optimizer: string | null;
      trainingConfig: unknown;
    };
  };

  /** A plain-language note, generated from what is actually known. */
  note: string;
};

export type ExportModelInput = {
  model: RegisteredModel;
  /** Serialised weights from the live model. */
  weights: SerializedWeights;
  checkpoint?: AlphaCheckpoint | null;
  seed?: number | null;
  optimizer?: string | null;
  trainingConfig?: unknown;
  now?: number;
};

/**
 * Build the export artifact. The parameter count is recomputed from the config
 * rather than copied, and the config is validated before anything is written.
 */
export function exportModel(input: ExportModelInput): AlphaModelExport {
  const { model } = input;

  // The authentic validator. An export that cannot be instantiated is refused.
  validateModelConfig(model.config);

  const parameterCount = countParameters(model.config);
  const fingerprint = modelConfigFingerprint(model.config);
  if (parameterCount !== model.parameterCount) {
    throw new AlphaValidationError(
      "model",
      `refusing to export "${model.modelId}": registered parameter count ${model.parameterCount.toLocaleString()} does not match the architecture (${parameterCount.toLocaleString()})`,
      { modelId: model.modelId },
    );
  }

  const hasWeights = model.trainedTokens > 0;
  const note = hasWeights
    ? `Alpha ${model.name}@${model.modelVersion} — ${parameterCount.toLocaleString()} parameters, ` +
      `${model.trainedTokens.toLocaleString()} training tokens, lifecycle ${model.lifecycle}. ` +
      `Trained only on ${model.relationships.datasetReference} (${model.relationships.datasetLicense}). ` +
      `No external model weights are present in this artifact.`
    : `Alpha ${model.name}@${model.modelVersion} — ${parameterCount.toLocaleString()} parameters, ` +
      `NO trained weights. Lifecycle ${model.lifecycle}. These weights are random initialisation; ` +
      `this artifact must not be presented as a capable model.`;

  return {
    format: "alpha.model-export",
    formatVersion: ALPHA_MODEL_EXPORT_VERSION,
    createdAt: input.now ?? Date.now(),
    modelId: model.modelId,
    modelVersion: model.modelVersion,
    name: model.name,
    lifecycle: model.lifecycle,
    model,
    config: model.config,
    configFingerprint: fingerprint,
    parameterCount,
    weights: input.weights,
    checkpoint: input.checkpoint ?? null,
    provenance: {
      tokenizerVersion: model.relationships.tokenizerVersion,
      tokenizerFingerprint: model.relationships.tokenizerFingerprint,
      datasetReference: model.relationships.datasetReference,
      datasetFingerprint: model.relationships.datasetFingerprint,
      datasetLicense: model.relationships.datasetLicense,
      trainingRunId: model.relationships.trainingRunId,
      checkpointId: model.relationships.checkpointId,
      parentCheckpointId: model.relationships.parentCheckpointId,
      seed: input.seed ?? null,
      trainedTokens: model.trainedTokens,
      validationLoss: model.validationLoss,
      validationPerplexity: model.validationPerplexity,
      evaluationIds: [...model.relationships.evaluationIds],
      reproducibility: {
        configFingerprint: fingerprint,
        datasetFingerprint: model.relationships.datasetFingerprint,
        tokenizerFingerprint: model.relationships.tokenizerFingerprint,
        seed: input.seed ?? null,
        optimizer: input.optimizer ?? null,
        trainingConfig: input.trainingConfig ?? null,
      },
    },
    note,
  };
}

export type ImportResult = {
  model: AlphaTransformer;
  config: AlphaModelConfig;
  parameterCount: number;
  /** Warnings raised while loading; an import is refused for hard failures. */
  warnings: string[];
};

/**
 * Rebuild a model from an export artifact. Every consistency claim in the
 * artifact is re-checked against the payload — a hand-edited or corrupted
 * artifact is refused rather than half-loaded.
 */
export function importModel(artifact: AlphaModelExport): ImportResult {
  if (artifact?.format !== "alpha.model-export") {
    throw new AlphaValidationError("model", "not an Alpha model export artifact");
  }
  if (artifact.formatVersion !== ALPHA_MODEL_EXPORT_VERSION) {
    throw new AlphaValidationError(
      "model",
      `export format ${artifact.formatVersion} is not the version Alpha reads (${ALPHA_MODEL_EXPORT_VERSION})`,
      { expected: ALPHA_MODEL_EXPORT_VERSION, received: artifact.formatVersion },
    );
  }

  validateModelConfig(artifact.config);

  const fingerprint = modelConfigFingerprint(artifact.config);
  if (fingerprint !== artifact.configFingerprint) {
    throw new AlphaValidationError(
      "model",
      `artifact is inconsistent: its config fingerprint (${artifact.configFingerprint}) does not match its own configuration (${fingerprint})`,
    );
  }

  const parameterCount = countParameters(artifact.config);
  if (parameterCount !== artifact.parameterCount) {
    throw new AlphaValidationError(
      "model",
      `artifact is inconsistent: it claims ${artifact.parameterCount.toLocaleString()} parameters but its architecture has ${parameterCount.toLocaleString()}`,
    );
  }

  const model = new AlphaTransformer(artifact.config);
  const warnings: string[] = [];

  const shapeMismatch = (() => {
    for (const [name, shape] of Object.entries(artifact.weights.shapes ?? {})) {
      const expected = model.parameterMap().get(name)?.shape;
      if (expected && expected.join(",") !== shape.join(",")) {
        return `${name} is ${shape.join("x")} in the artifact but ${expected.join("x")} in the architecture`;
      }
    }
    return null;
  })();
  if (shapeMismatch) {
    throw new AlphaValidationError(
      "model",
      `artifact weights do not match the architecture: ${shapeMismatch}`,
    );
  }

  model.loadWeights(artifact.weights);

  if (artifact.provenance.trainedTokens > 0 && !artifact.provenance.checkpointId) {
    warnings.push(
      `artifact claims ${artifact.provenance.trainedTokens.toLocaleString()} trained tokens but carries no checkpoint id; treat its training provenance as unverified`,
    );
  }
  if (artifact.provenance.trainedTokens === 0) {
    warnings.push(
      "artifact carries no trained weights — these are random initialisation values",
    );
  }
  if (
    artifact.checkpoint &&
    artifact.checkpoint.configFingerprint !== fingerprint
  ) {
    throw new AlphaValidationError(
      "model",
      `artifact's checkpoint was trained against a different architecture (${artifact.checkpoint.configFingerprint}) than the artifact declares (${fingerprint})`,
    );
  }
  if (
    artifact.checkpoint &&
    artifact.checkpoint.formatVersion !== ALPHA_CHECKPOINT_FORMAT_VERSION
  ) {
    warnings.push(
      `artifact's checkpoint is format ${artifact.checkpoint.formatVersion}; Alpha writes ${ALPHA_CHECKPOINT_FORMAT_VERSION}`,
    );
  }

  return { model, config: artifact.config, parameterCount, warnings };
}

/** Serialise an export to JSON. Weights are base64 inside `SerializedWeights`. */
export function serialiseModelExport(artifact: AlphaModelExport): string {
  return JSON.stringify(artifact);
}

/** Parse a JSON export, refusing anything that is not a well-formed artifact. */
export function parseModelExport(json: string): AlphaModelExport {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch (error) {
    throw new AlphaValidationError(
      "model",
      `model export is not valid JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  const artifact = parsed as AlphaModelExport;
  if (!artifact || typeof artifact !== "object" || typeof artifact.format !== "string") {
    throw new AlphaValidationError("model", "model export is missing its format marker");
  }
  if (!artifact.weights || typeof artifact.weights !== "object") {
    throw new AlphaValidationError("model", "model export carries no weights");
  }
  if (!artifact.config || typeof artifact.config !== "object") {
    throw new AlphaValidationError("model", "model export carries no configuration");
  }
  if (!artifact.provenance || typeof artifact.provenance !== "object") {
    throw new AlphaValidationError("model", "model export carries no provenance record");
  }
  return artifact;
}

/** Convenience id for an export, for logs and manifests. */
export function exportId(artifact: AlphaModelExport): string {
  return `${alphaId("exp")}_${artifact.modelId.replace(/[^a-zA-Z0-9@._-]/g, "_")}`;
}
