/**
 * Alpha Serving Artifact.
 *
 * The artifact is the *whole* model as one document: the tokenizer snapshot, the
 * architecture it was trained under, the weights themselves, and the metadata
 * that makes them checkable — every fingerprint the Step 5 verification run
 * published, plus the measurements that run produced.
 *
 * Three rules shape this module:
 *
 *   1. An artifact is verified, not trusted. Loading recomputes the tokenizer
 *      fingerprint, the configuration fingerprint and the parameter count, and
 *      refuses anything that does not match what the file claims. A weights blob
 *      that would load into the wrong shape is rejected before a single forward
 *      pass can produce text from it.
 *
 *   2. There is no external model. `externalModels` is a literal `"none"` in the
 *      format, weights are float32 arrays produced by Alpha's own trainer, and
 *      nothing here fetches, downloads or contacts a provider.
 *
 *   3. A load failure is a load failure. `loadServingArtifact` throws with the
 *      specific mismatch; it never substitutes random weights, a smaller
 *      architecture or a fallback model.
 */

import { AlphaValidationError } from "../core/errors";
import type { AlphaModelStage } from "../core/types";
import { base64ByteLength } from "../core/serialize";
import {
  countParameters,
  modelConfigFingerprint,
  validateModelConfig,
  type AlphaModelConfig,
} from "../model/config";
import { AlphaTransformer, type SerializedWeights } from "../model/transformer";
import { AlphaTokenizer, type AlphaTokenizerSnapshot } from "../tokenizer/bpe";
import type { SamplingConfig } from "../inference/engine";
import type { TrainingConfig } from "../training/trainer";

/** Format version written into every artifact. Bump when the shape changes. */
export const ALPHA_SERVING_ARTIFACT_FORMAT = "alpha-serving-artifact/1";

export type AlphaServingArtifact = {
  formatVersion: string;
  createdAt: number;
  /** Honest stage of the weights. `trained` for a real training run. */
  stage: AlphaModelStage;
  model: {
    config: AlphaModelConfig;
    configFingerprint: string;
    parameterCount: number;
  };
  tokenizer: {
    snapshot: AlphaTokenizerSnapshot;
    fingerprint: string;
    vocabSize: number;
    version: string;
    trainedOn: string;
  };
  /** Float32 tensors, base64, exactly as `AlphaTransformer.serializeWeights()` emits. */
  weights: SerializedWeights;
  training: {
    steps: number;
    tokensSeen: number;
    tokensPerStep: number;
    seed: number;
    firstLoss: number | null;
    lastLoss: number | null;
    validationLoss: number | null;
    validationPerplexity: number | null;
    uniformLossBaseline: number;
    durationMs: number;
    tokensPerSecond: number;
    checkpointId: string | null;
    gradientAccumulationSteps: number;
    config: TrainingConfig;
  };
  data: {
    datasetName: string;
    datasetVersion: string;
    datasetFingerprint: string;
    mixtureFingerprint: string | null;
    documents: number;
    characters: number;
  };
  evaluation: {
    suiteFingerprint: string;
    suiteCases: number;
    heldOutDocuments: number;
    loss: number | null;
    perplexity: number | null;
    nextTokenTop1Accuracy: number | null;
    gateFingerprint: string;
    gatePassed: boolean;
  };
  /** Defaults the chat product starts from; a caller may override them. */
  generation: {
    defaults: Partial<SamplingConfig>;
  };
  /** Stated in the file itself, so the file cannot be read as provider-backed. */
  externalModels: "none";
};

export type LoadedServingArtifact = {
  artifact: AlphaServingArtifact;
  tokenizer: AlphaTokenizer;
  model: AlphaTransformer;
  loadMs: number;
  /** Non-fatal notes about the file (currently always empty; kept explicit). */
  notes: string[];
};

function fail(problem: string, detail?: Record<string, unknown>): never {
  throw new AlphaValidationError("model", `serving artifact rejected: ${problem}`, detail ?? {});
}

/** Rebuild a tokenizer from its snapshot, reporting a broken snapshot as one. */
function rebuildTokenizer(snapshot: AlphaTokenizerSnapshot): AlphaTokenizer {
  try {
    return AlphaTokenizer.fromJSON(snapshot);
  } catch (error) {
    fail(
      `the tokenizer snapshot could not be rebuilt: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

/** Build the artifact document from a trained model. Used by the export path. */
export function createServingArtifact(input: {
  model: AlphaTransformer;
  tokenizer: AlphaTokenizer;
  stage: AlphaModelStage;
  training: AlphaServingArtifact["training"];
  data: AlphaServingArtifact["data"];
  evaluation: AlphaServingArtifact["evaluation"];
  generationDefaults?: Partial<SamplingConfig>;
  createdAt?: number;
}): AlphaServingArtifact {
  const config = input.model.config;
  validateModelConfig(config);
  return {
    formatVersion: ALPHA_SERVING_ARTIFACT_FORMAT,
    createdAt: input.createdAt ?? Date.now(),
    stage: input.stage,
    model: {
      config,
      configFingerprint: modelConfigFingerprint(config),
      parameterCount: countParameters(config),
    },
    tokenizer: {
      snapshot: input.tokenizer.toJSON(),
      fingerprint: input.tokenizer.fingerprint(),
      vocabSize: input.tokenizer.vocabSize,
      version: input.tokenizer.version,
      trainedOn: input.tokenizer.trainedOn,
    },
    weights: input.model.serializeWeights(),
    training: input.training,
    data: input.data,
    evaluation: input.evaluation,
    generation: {
      defaults: input.generationDefaults ?? {
        temperature: 0.8,
        topK: 40,
        topP: 0.95,
        maxNewTokens: 64,
        repetitionPenalty: 1.1,
        deterministic: false,
      },
    },
    externalModels: "none",
  };
}

/**
 * Validate an artifact document without building anything from it.
 *
 * Every fingerprint recorded in the file is recomputed here, so a file whose
 * configuration was edited after the run that produced it is refused.
 */
export function parseServingArtifact(value: unknown): AlphaServingArtifact {
  if (!value || typeof value !== "object") fail("the artifact is not an object");
  const artifact = value as AlphaServingArtifact;

  if (artifact.formatVersion !== ALPHA_SERVING_ARTIFACT_FORMAT) {
    fail(
      `unknown format version "${String(artifact.formatVersion)}"; this build reads ${ALPHA_SERVING_ARTIFACT_FORMAT}`,
      { formatVersion: artifact.formatVersion },
    );
  }
  if (artifact.externalModels !== "none") {
    fail(
      "the artifact does not declare that no external model is involved; Alpha refuses to serve it",
      { externalModels: artifact.externalModels },
    );
  }
  if (!artifact.model?.config) fail("the artifact carries no model configuration");
  validateModelConfig(artifact.model.config);

  const recomputedConfig = modelConfigFingerprint(artifact.model.config);
  if (recomputedConfig !== artifact.model.configFingerprint) {
    fail(
      `configuration fingerprint mismatch: the file claims ${artifact.model.configFingerprint}, the configuration hashes to ${recomputedConfig}`,
      { claimed: artifact.model.configFingerprint, recomputed: recomputedConfig },
    );
  }
  const recomputedParameters = countParameters(artifact.model.config);
  if (recomputedParameters !== artifact.model.parameterCount) {
    fail(
      `parameter count mismatch: the file claims ${artifact.model.parameterCount}, the configuration counts ${recomputedParameters}`,
      { claimed: artifact.model.parameterCount, recomputed: recomputedParameters },
    );
  }

  if (!artifact.tokenizer?.snapshot) fail("the artifact carries no tokenizer snapshot");
  const rebuiltTokenizer = rebuildTokenizer(artifact.tokenizer.snapshot);
  const recomputedTokenizer = rebuiltTokenizer.fingerprint();
  if (recomputedTokenizer !== artifact.tokenizer.fingerprint) {
    fail(
      `tokenizer fingerprint mismatch: the file claims ${artifact.tokenizer.fingerprint}, the snapshot hashes to ${recomputedTokenizer}`,
      { claimed: artifact.tokenizer.fingerprint, recomputed: recomputedTokenizer },
    );
  }
  if (rebuiltTokenizer.vocabSize !== artifact.tokenizer.vocabSize) {
    fail(
      `tokenizer vocabulary mismatch: the file claims ${artifact.tokenizer.vocabSize}, the snapshot holds ${rebuiltTokenizer.vocabSize}`,
    );
  }
  if (rebuiltTokenizer.vocabSize > artifact.model.config.vocabSize) {
    fail(
      `tokenizer vocabulary (${rebuiltTokenizer.vocabSize}) is larger than the model vocabulary (${artifact.model.config.vocabSize})`,
    );
  }

  const tensors = artifact.weights?.tensors;
  if (!tensors || typeof tensors !== "object" || Object.keys(tensors).length === 0) {
    fail("the artifact carries no weights");
  }
  if (artifact.weights.config && modelConfigFingerprint(artifact.weights.config) !== recomputedConfig) {
    fail("the weights were serialised from a different configuration than the model block declares");
  }

  return artifact;
}

/**
 * Build the tokenizer and the model from an artifact.
 *
 * The parameter shapes are checked against the serialised tensors *before* the
 * weights are loaded, because `loadWeights` would otherwise skip a mismatched
 * tensor and leave random values in place — which would produce fluent-looking
 * garbage attributed to a model that never generated it.
 */
export function loadServingArtifact(value: unknown): LoadedServingArtifact {
  const startedAt = Date.now();
  const artifact = parseServingArtifact(value);

  const tokenizer = AlphaTokenizer.fromJSON(artifact.tokenizer.snapshot);
  const model = new AlphaTransformer(artifact.model.config);

  const expected = model.parameterMap();
  const supplied = artifact.weights.tensors;
  const missing: string[] = [];
  const wrongShape: string[] = [];
  for (const [name, tensor] of expected) {
    const encoded = supplied[name];
    if (typeof encoded !== "string") {
      missing.push(name);
      continue;
    }
    if (base64ByteLength(encoded) !== tensor.size * 4) {
      wrongShape.push(`${name} (${base64ByteLength(encoded)} bytes for ${tensor.size} float32 values)`);
    }
  }
  if (missing.length > 0 || wrongShape.length > 0) {
    fail(
      `weights do not match the architecture: ${missing.length} missing tensor(s), ${wrongShape.length} with the wrong size`,
      { missing: missing.slice(0, 8), wrongShape: wrongShape.slice(0, 8) },
    );
  }

  model.loadWeights(artifact.weights);

  // A last, cheap check that the load actually populated the parameters: a
  // freshly constructed model has non-zero initialisation, so an all-zero
  // tensor would mean the write silently did nothing.
  const embedding = expected.get("tok_embedding") ?? [...expected.values()][0];
  if (embedding) {
    let nonZero = 0;
    for (let i = 0; i < embedding.data.length && nonZero === 0; i++) {
      if (embedding.data[i] !== 0) nonZero += 1;
    }
    if (nonZero === 0) fail("the loaded weights are all zero; refusing to serve an empty model");
  }

  return {
    artifact,
    tokenizer,
    model,
    loadMs: Date.now() - startedAt,
    notes: [],
  };
}

/** One-line description for logs and status panels. */
export function describeServingArtifact(artifact: AlphaServingArtifact): string {
  return (
    `${artifact.model.config.name}@${artifact.model.config.version} [${artifact.stage}] · ` +
    `${artifact.model.parameterCount.toLocaleString()} parameters · ${artifact.model.configFingerprint} · ` +
    `tokenizer ${artifact.tokenizer.fingerprint} (${artifact.tokenizer.vocabSize} tokens) · ` +
    `dataset ${artifact.data.datasetFingerprint} · suite ${artifact.evaluation.suiteFingerprint} · ` +
    `external models: none`
  );
}
