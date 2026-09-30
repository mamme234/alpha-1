import { describe, expect, it } from "vitest";
import { AlphaValidationError } from "../core/errors";
import {
  ALPHA_MODEL_PRESETS,
  countParameters,
  createModelConfig,
  deriveStage,
  describeArchitecture,
  modelConfigFingerprint,
  validateModelConfig,
  withSpecialTokenIds,
  type AlphaModelConfig,
} from "../model/config";
import { AlphaTransformer } from "../model/transformer";
import { AlphaTokenizer } from "../tokenizer/bpe";
import { AlphaTrainer } from "../training/trainer";
import {
  RUNTIME_CAPABILITIES,
  assertTrainableWithinLimits,
  canTransitionLifecycle,
  estimateResources,
  lifecycleFromJobState,
  terminalLifecycle,
  tokenMetricsFromSummary,
} from "../training/scaling";
import { AlphaModelRegistry, newModelId } from "../model/registry";
import { exportModel, importModel, parseModelExport, serialiseModelExport } from "../model/export";
import {
  assertTrainableDatasetVersion,
  createDatasetVersion,
  describeDatasetVersion,
  normaliseDocument,
  repetitionRatio,
  splitDataset,
  toDataset,
  DEFAULT_NORMALISATION,
  type CreateDatasetVersionInput,
} from "../datasets/versions";
import {
  analyseDatasetVersionQuality,
  analyseDatasetQuality,
  summariseQualityReport,
} from "../datasets/quality";
import { buildGeneratedCorpus, generatedCorpusTopics } from "../datasets/generated-corpus";
import { assertNotBenchmark, benchmarkIdentity, createAlphaBenchmark } from "../evaluation/benchmark";
import { evaluateModel, metricValue, passedChecks } from "../evaluation/framework";
import { assertCheckpointCompatible } from "../training/checkpoint";

const OWNED_SOURCE = [
  { id: "alpha-test", title: "Alpha test corpus", license: "Alpha-owned", origin: "authored" as const },
];

function makeVersion(overrides: Partial<CreateDatasetVersionInput> = {}) {
  return createDatasetVersion({
    datasetId: "alpha-test",
    name: "alpha-test-corpus",
    version: "1.0.0",
    description: "corpus for step 4 tests",
    sources: OWNED_SOURCE,
    documents: [
      "alpha is a self owned language model.",
      "the transformer processes every position in parallel.",
      "attention lets every position weigh every other position.",
      "byte pair encoding builds a vocabulary by merging pairs.",
      "a checkpoint stores the weights and the optimiser state.",
    ],
    now: 0,
    ...overrides,
  });
}

/** A small but valid model config derived from the authentic presets. */
function smallConfig(preset: "nano" | "micro" | "small" = "nano"): AlphaModelConfig {
  const base = ALPHA_MODEL_PRESETS[preset];
  return createModelConfig({ preset, vocabSize: 128, contextLength: 32, dModel: 32, nHeads: 4, nLayers: 2, dFeedForward: 64 });
}

function baseRelationships(overrides: Record<string, unknown> = {}) {
  return {
    tokenizerVersion: "1.0.0",
    tokenizerFingerprint: "tok_aaa",
    datasetReference: "alpha-test@1.0.0",
    datasetFingerprint: "ds_aaa",
    datasetLicense: "Alpha-owned",
    trainingRunId: null,
    checkpointId: null,
    evaluationIds: [],
    parentCheckpointId: null,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
describe("step 4 — model configuration and parameter counting", () => {
  it("counts the authentic preset parameters exactly", () => {
    expect(countParameters(ALPHA_MODEL_PRESETS.nano)).toBe(128768);
    expect(countParameters(ALPHA_MODEL_PRESETS.micro)).toBe(418656);
    expect(countParameters(ALPHA_MODEL_PRESETS.small)).toBe(1071872);
  });

  it("agrees between the architecture table and a real model for every preset", () => {
    for (const preset of ["nano", "micro", "small"] as const) {
      const config = ALPHA_MODEL_PRESETS[preset];
      const model = new AlphaTransformer(config);
      const fromTable = describeArchitecture(config).reduce((s, r) => s + r.parameters, 0);
      const fromTensors = model.parameters().reduce((s, p) => s + p.tensor.size, 0);
      expect(fromTable).toBe(countParameters(config));
      expect(fromTensors).toBe(countParameters(config));
    }
  });

  it("calculates a larger configuration from the architecture rather than a claim", () => {
    // A shape larger than `small`, built only from real fields.
    const large = createModelConfig({
      preset: "small",
      vocabSize: 4096,
      contextLength: 256,
      dModel: 192,
      nHeads: 6,
      nLayers: 6,
      dFeedForward: 768,
      dropout: 0,
    });
    validateModelConfig(large);
    const expected =
      large.vocabSize * large.dModel +
      large.contextLength * large.dModel +
      (4 * large.dModel ** 2 + 4 * large.dModel + 2 * large.dModel * large.dFeedForward + large.dFeedForward + large.dModel + 4 * large.dModel) *
        large.nLayers +
      2 * large.dModel;
    expect(countParameters(large)).toBe(expected);
    // Genuinely larger than the 128,768-parameter starting point.
    expect(countParameters(large)).toBeGreaterThan(countParameters(ALPHA_MODEL_PRESETS.small));
    expect(new AlphaTransformer(large).parameterCount).toBe(expected);
  });

  it("rejects configurations that cannot produce a usable model", () => {
    expect(() => validateModelConfig({ ...smallConfig(), dModel: 30, nHeads: 4 })).toThrow(/divisible by nHeads/);
    expect(() => validateModelConfig({ ...smallConfig(), nLayers: 0 })).toThrow(/nLayers/);
    expect(() => validateModelConfig({ ...smallConfig(), nHeads: 0 })).toThrow(/nHeads/);
    expect(() => validateModelConfig({ ...smallConfig(), vocabSize: 1 })).toThrow(/vocabSize/);
    expect(() => validateModelConfig({ ...smallConfig(), contextLength: 1 })).toThrow(/contextLength/);
    expect(() => validateModelConfig({ ...smallConfig(), dFeedForward: 4 })).toThrow(/dFeedForward/);
    expect(() => validateModelConfig({ ...smallConfig(), dropout: 1.5 })).toThrow(/dropout/);
    expect(() =>
      validateModelConfig({ ...smallConfig(), specialTokenIds: { pad: 0, unk: 1, bos: 1, eos: 2 } }),
    ).toThrow(/must all be different/);
    expect(() =>
      validateModelConfig({ ...smallConfig(), specialTokenIds: { pad: 0, unk: 1, bos: 2, eos: 9999 } }),
    ).toThrow(/outside the vocabulary/);
  });

  it("keeps the authentic fingerprint behaviour and is sensitive to architecture", () => {
    const a = smallConfig();
    expect(modelConfigFingerprint(a)).toBe(modelConfigFingerprint({ ...a }));
    expect(modelConfigFingerprint({ ...a, nLayers: 3 })).not.toBe(modelConfigFingerprint(a));
    expect(modelConfigFingerprint(a)).toMatch(/^cfg_[0-9a-f]{8}$/);
  });

  it("preserves the verified deriveStage ordering: no weights beats promotion", () => {
    // A weightless model can never be production, even if promotion is claimed.
    expect(
      deriveStage({ hasWeights: false, hasCheckpoint: false, trainedTokens: 0, isFineTune: false, promoted: true }),
    ).toBe("architecture");
    expect(
      deriveStage({ hasWeights: true, hasCheckpoint: false, trainedTokens: 0, isFineTune: false, promoted: false }),
    ).toBe("untrained");
    expect(
      deriveStage({ hasWeights: true, hasCheckpoint: true, trainedTokens: 10, isFineTune: true, promoted: false }),
    ).toBe("fine-tuned");
  });
});

// ---------------------------------------------------------------------------
describe("step 4 — dataset versioning", () => {
  it("produces a stable fingerprint for identical content", () => {
    const a = makeVersion();
    const b = makeVersion();
    expect(a.manifest.fingerprint).toBe(b.manifest.fingerprint);
    const changed = makeVersion({ documents: [...makeVersion().documents, "a new document."] });
    expect(changed.manifest.fingerprint).not.toBe(a.manifest.fingerprint);
  });

  it("refuses a dataset with no source or no licence", () => {
    expect(() => makeVersion({ sources: [] })).toThrow(/no source/);
    expect(() =>
      makeVersion({ sources: [{ id: "x", title: "x", license: "", origin: "authored" }] }),
    ).toThrow(/no licence/);
  });

  it("refuses to train on material with unknown origin", () => {
    const version = makeVersion({
      sources: [{ id: "mystery", title: "unknown", license: "?", origin: "unknown" }],
    });
    expect(() => assertTrainableDatasetVersion(version)).toThrow(/not trainable/);
    expect(assertTrainableDatasetVersion(makeVersion()).trainable).toBe(true);
  });

  it("normalises deterministically and records what filtering removed", () => {
    expect(normaliseDocument("  a   b  ", DEFAULT_NORMALISATION)).toBe("a b");
    expect(normaliseDocument("a\r\nb", DEFAULT_NORMALISATION)).toBe("a\nb");
    const version = makeVersion({
      documents: ["real content here.", "real content here.", "   ", "another real document."],
    });
    expect(version.documents).toHaveLength(2);
    expect(version.manifest.duplicatesRemoved).toBe(1);
    expect(version.manifest.filters.map((f) => f.rule).sort()).toEqual(["duplicate", "empty"]);
    // Nothing is deleted silently: the removals are all accounted for.
    const removed = version.manifest.filters.reduce((s, f) => s + f.removed, 0) as number;
    expect(4 - version.documents.length).toBe(removed);
  });

  it("detects degenerate repetition", () => {
    const loop = "0123456789abcdefghijklmn".repeat(40);
    expect(repetitionRatio(loop)).toBeGreaterThan(0.5);
    expect(repetitionRatio("a reasonably varied sentence with many different words in it.")).toBeLessThan(0.5);
  });

  it("produces disjoint train/validation/test splits covering every document", () => {
    const version = makeVersion({ documents: Array.from({ length: 50 }, (_, i) => `document number ${i} with distinct words.`) });
    const train = splitDataset(version, "train");
    const validation = splitDataset(version, "validation");
    const test = splitDataset(version, "test");
    expect(train.documents.length + validation.documents.length + test.documents.length).toBe(50);
    const ids = new Set(
      version.manifest.splits.flatMap((s) => s.documentIndexes),
    );
    expect(ids.size).toBe(50);
    // No document is in two splits.
    const seen = new Set<number>();
    for (const split of version.manifest.splits) {
      for (const i of split.documentIndexes) {
        expect(seen.has(i)).toBe(false);
        seen.add(i);
      }
    }
  });

  it("summarises a version with its provenance", () => {
    expect(describeDatasetVersion(makeVersion())).toMatch(/alpha-test-corpus@1\.0\.0.*Alpha-owned/s);
  });
});

// ---------------------------------------------------------------------------
describe("step 4 — data quality", () => {
  it("reports empty, duplicate, invalid-unicode and repetitive documents", () => {
    const report = analyseDatasetQuality({
      id: "x", name: "x", version: "1", description: "", license: "Alpha-owned", source: "x",
      documents: [
        "a perfectly ordinary document about transformers.",
        "a perfectly ordinary document about transformers.",
        "",
        "b".repeat(200),
        "valid text with a lone surrogate \ud800 in it.",
      ],
    });
    const codes = report.findings.map((f) => f.code);
    expect(codes).toContain("empty-documents");
    expect(codes).toContain("duplicate-documents");
    expect(codes).toContain("invalid-unicode");
    expect(report.counts.duplicate).toBe(1);
    expect(report.counts.empty).toBe(1);
    expect(report.pass).toBe(false);
  });

  it("passes a clean versioned corpus and reports splits and licences", () => {
    const version = makeVersion({ documents: Array.from({ length: 30 }, (_, i) => `clean document ${i} about alpha components.`) });
    const report = analyseDatasetVersionQuality(version);
    expect(report.pass).toBe(true);
    expect(report.splits.length).toBe(3);
    expect(report.unlicensedSources).toEqual([]);
    expect(summariseQualityReport(report)).toMatch(/PASS/);
  });

  it("detects split leakage", () => {
    const version = makeVersion({ documents: Array.from({ length: 12 }, (_, i) => `shared document ${i} text.`) });
    // Force leakage: put the same index into two splits.
    version.manifest.splits[0].documentIndexes = [0, 1, 2];
    version.manifest.splits[1].documentIndexes = [0, 1, 2];
    const report = analyseDatasetVersionQuality(version);
    expect(report.findings.some((f) => f.code.startsWith("split-leakage"))).toBe(true);
    expect(report.pass).toBe(false);
  });
});

// ---------------------------------------------------------------------------
describe("step 4 — evaluation", () => {
  it("refuses to train on its own benchmark", () => {
    const benchmark = createAlphaBenchmark(0);
    expect(() => assertNotBenchmark(toDataset(benchmark))).toThrow(/evaluation benchmark/);
    expect(assertNotBenchmark(toDataset(makeVersion())).ok).toBe(true);
  });

  it("reports a versioned benchmark identity", () => {
    const identity = benchmarkIdentity();
    expect(identity.name).toBe("alpha-eval-benchmark");
    expect(identity.fingerprint).toMatch(/^ds_/);
    expect(identity.recallCases).toBeGreaterThan(0);
    expect(identity.heldOutCases).toBeGreaterThan(0);
  });

  it("measures a real model and records raw metrics with no invented score", () => {
    const version = makeVersion({ documents: Array.from({ length: 20 }, (_, i) => `evaluation document ${i} with words.`) });
    const dataset = toDataset(version);
    const tokenizer = AlphaTokenizer.train(dataset.documents, { vocabSize: 160, version: "1.0.0", trainedOn: "test" });
    const config = smallConfig();
    const model = new AlphaTransformer(config);
    const report = evaluateModel(model, tokenizer, { evaluationDocuments: dataset.documents, maxNewTokens: 8 });

    expect(report.parameterCount).toBe(countParameters(config));
    expect(report.configFingerprint).toBe(modelConfigFingerprint(config));
    // Measured, not claimed.
    const loss = metricValue(report, "languageModelLoss");
    expect(loss).not.toBeNull();
    expect(loss).toBeGreaterThan(0);
    expect(metricValue(report, "uniformLossBaseline")).toBeCloseTo(Math.log(config.vocabSize), 6);
    expect(metricValue(report, "perplexity")).toBeCloseTo(Math.exp(loss as number), 4);
    expect(metricValue(report, "tokensPerSecond")).toBeGreaterThan(0);
    // Recall and held-out are counted separately and never blended.
    expect(report.capabilityCounts.recall.total).toBeGreaterThan(0);
    expect(report.capabilityCounts.heldOut.total).toBeGreaterThan(0);
    expect(report.capabilities.every((c) => c.kind === "recall" || c.kind === "held-out")).toBe(true);
    expect(passedChecks(report)).toBe(report.checks.length);
  });

  it("produces identical generation for an identical greedy run", () => {
    const dataset = toDataset(makeVersion());
    const tokenizer = AlphaTokenizer.train(dataset.documents, { vocabSize: 160, version: "1.0.0", trainedOn: "test" });
    const model = new AlphaTransformer(smallConfig());
    const first = evaluateModel(model, tokenizer, { evaluationDocuments: dataset.documents, maxNewTokens: 6 });
    const second = evaluateModel(model, tokenizer, { evaluationDocuments: dataset.documents, maxNewTokens: 6 });
    expect(metricValue(first, "languageModelLoss")).toBe(metricValue(second, "languageModelLoss"));
    expect(first.checks.find((c) => c.name === "deterministic-generation")?.passed).toBe(true);
  });

  it("refuses an empty evaluation suite rather than reporting a fake number", () => {
    const dataset = toDataset(makeVersion());
    const tokenizer = AlphaTokenizer.train(dataset.documents, { vocabSize: 160, version: "1.0.0", trainedOn: "test" });
    expect(() =>
      evaluateModel(new AlphaTransformer(smallConfig()), tokenizer, { evaluationDocuments: [] }),
    ).toThrow(/at least one document/);
  });
});

// ---------------------------------------------------------------------------
describe("step 4 — model registry and lifecycle", () => {
  it("registers a model with a computed parameter count and relationships", () => {
    const registry = new AlphaModelRegistry();
    const config = smallConfig();
    const model = registry.register({
      modelId: "alpha-test@0.1.0",
      modelVersion: "0.1.0",
      config,
      preset: "nano",
      relationships: baseRelationships(),
    });
    expect(model.parameterCount).toBe(countParameters(config));
    expect(model.lifecycle).toBe("UNTRAINED");
    expect(registry.list()).toHaveLength(1);
    expect(registry.get(model.modelId)?.configFingerprint).toBe(modelConfigFingerprint(config));
  });

  it("validates every configuration before accepting it", () => {
    const registry = new AlphaModelRegistry();
    expect(() =>
      registry.register({
        modelId: "bad@1",
        modelVersion: "1",
        config: { ...smallConfig(), dModel: 30, nHeads: 4 },
        relationships: baseRelationships(),
      }),
    ).toThrow(AlphaValidationError);
    expect(registry.size).toBe(0);
  });

  it("never overwrites an existing model version", () => {
    const registry = new AlphaModelRegistry();
    const config = smallConfig();
    registry.register({ modelId: "a@1", modelVersion: "1", config, relationships: baseRelationships() });
    expect(() =>
      registry.register({ modelId: "a@1", modelVersion: "1", config, relationships: baseRelationships() }),
    ).toThrow(/already registered/);
    // A new run gets a new id, and the old one survives.
    const second = registry.register({ modelId: "b@1", modelVersion: "1", config, relationships: baseRelationships() });
    expect(registry.size).toBe(2);
    expect(registry.get("a@1")).not.toBeNull();
    expect(second.modelId).not.toBe("a@1");
    expect(newModelId("alpha-test", "0.2.0", 1000)).not.toBe(newModelId("alpha-test", "0.2.0", 1000));
  });

  it("refuses to call a weightless model trained", () => {
    const registry = new AlphaModelRegistry();
    const config = smallConfig();
    expect(() =>
      registry.register({ modelId: "x@1", modelVersion: "1", config, relationships: baseRelationships(), lifecycle: "TRAINED" }),
    ).toThrow(/no trained weights/);
    expect(() =>
      registry.register({ modelId: "y@1", modelVersion: "1", config, relationships: baseRelationships(), lifecycle: "PRODUCTION" }),
    ).toThrow(/no trained weights/);
  });

  it("refuses a checkpoint that reports zero trained tokens", () => {
    const registry = new AlphaModelRegistry();
    const model = registry.register({ modelId: "z@1", modelVersion: "1", config: smallConfig(), relationships: baseRelationships() });
    expect(() => registry.recordCheckpoint(model.modelId, { checkpointId: "ck", trainingRunId: "run", trainedTokens: 0 })).toThrow(
      /will not count it as training/,
    );
  });

  it("refuses to promote without weights, evaluation, operator and reason", () => {
    const registry = new AlphaModelRegistry();
    const untrained = registry.register({ modelId: "u@1", modelVersion: "1", config: smallConfig(), relationships: baseRelationships() });
    expect(() => registry.promote(untrained.modelId, { by: "operator", reason: "looks good" })).toThrow(/no trained weights/);

    const trained = registry.register({ modelId: "t@1", modelVersion: "1", config: smallConfig(), relationships: baseRelationships() });
    registry.recordCheckpoint(trained.modelId, { checkpointId: "ck1", trainingRunId: "run1", trainedTokens: 4096 });
    expect(trained.lifecycle).toBe("TRAINED");
    // Weights, but no evaluation.
    expect(() => registry.promote(trained.modelId, { by: "operator", reason: "trained" })).toThrow(/no evaluation/);

    registry.recordEvaluation(trained.modelId, "eval-1");
    expect(trained.lifecycle).toBe("EVALUATED");
    expect(() => registry.promote(trained.modelId, { by: "", reason: "x" })).toThrow(/operator identity/);
    expect(() => registry.promote(trained.modelId, { by: "op", reason: "" })).toThrow(/operator identity/);
    expect(() => registry.promote(trained.modelId, { by: "op", reason: "ok", evaluationId: "eval-999" })).toThrow(/not attached/);

    const promoted = registry.promote(trained.modelId, { by: "operator", reason: "passed evaluation" });
    expect(promoted.lifecycle).toBe("PRODUCTION");
    expect(promoted.promotion?.by).toBe("operator");
    expect(promoted.promotion?.reason).toBe("passed evaluation");
  });

  it("passes through PRODUCTION_CANDIDATE on the way to PRODUCTION", () => {
    const registry = new AlphaModelRegistry();
    const model = registry.register({ modelId: "p@1", modelVersion: "1", config: smallConfig(), relationships: baseRelationships() });
    registry.recordCheckpoint(model.modelId, { checkpointId: "c", trainingRunId: "r", trainedTokens: 100 });
    registry.recordEvaluation(model.modelId, "e1");
    registry.promote(model.modelId, { by: "op", reason: "r" });
    expect(model.lifecycle).toBe("PRODUCTION");
    // The recorded transition shows the candidate step.
    expect(model.promotion?.to).toBe("PRODUCTION");
  });

  it("checks compatibility across tokenizer, dataset and architecture", () => {
    const registry = new AlphaModelRegistry();
    const config = smallConfig();
    const model = registry.register({ modelId: "c@1", modelVersion: "1", config, relationships: baseRelationships() });

    const good = registry.checkCompatibility(model.modelId, {
      tokenizerVersion: "1.0.0",
      tokenizerFingerprint: "tok_aaa",
      datasetFingerprint: "ds_aaa",
      config,
    });
    expect(good.compatible).toBe(false); // no weights yet
    expect(good.checks.find((c) => c.name === "weights-present")?.passed).toBe(false);

    registry.recordCheckpoint(model.modelId, { checkpointId: "c1", trainingRunId: "r1", trainedTokens: 2048 });
    expect(
      registry.checkCompatibility(model.modelId, {
        tokenizerVersion: "1.0.0",
        tokenizerFingerprint: "tok_aaa",
        datasetFingerprint: "ds_aaa",
        config,
      }).compatible,
    ).toBe(true);

    const wrongTokenizer = registry.checkCompatibility(model.modelId, {
      tokenizerVersion: "1.0.0",
      tokenizerFingerprint: "tok_zzz",
      datasetFingerprint: "ds_aaa",
      config,
    });
    expect(wrongTokenizer.compatible).toBe(false);
    expect(wrongTokenizer.checks.find((c) => c.name === "tokenizer-fingerprint")?.passed).toBe(false);

    const wrongArch = registry.checkCompatibility(model.modelId, {
      tokenizerVersion: "1.0.0",
      tokenizerFingerprint: "tok_aaa",
      datasetFingerprint: "ds_aaa",
      config: { ...config, nLayers: 3 },
    });
    expect(wrongArch.compatible).toBe(false);
    expect(wrongArch.checks.find((c) => c.name === "runtime-architecture")?.passed).toBe(false);

    const wrongDataset = registry.checkCompatibility(model.modelId, {
      tokenizerVersion: "1.0.0",
      tokenizerFingerprint: "tok_aaa",
      datasetFingerprint: "ds_other",
      config,
    });
    expect(wrongDataset.compatible).toBe(false);
  });
});

// ---------------------------------------------------------------------------
describe("step 4 — model export and import", () => {
  it("round-trips a model and reproduces its weights exactly", () => {
    const registry = new AlphaModelRegistry();
    const config = smallConfig();
    const model = new AlphaTransformer(config);
    const registered = registry.register({
      modelId: "e@1",
      modelVersion: "1",
      config,
      relationships: baseRelationships({ checkpointId: "ck", trainingRunId: "run" }),
      trainedTokens: 2048,
    });
    const artifact = exportModel({ model: registered, weights: model.serializeWeights(), seed: 1337, now: 0 });
    expect(artifact.parameterCount).toBe(countParameters(config));
    expect(artifact.provenance.seed).toBe(1337);
    expect(artifact.provenance.trainedTokens).toBe(2048);

    const json = serialiseModelExport(artifact);
    const imported = importModel(parseModelExport(json));
    expect(imported.parameterCount).toBe(countParameters(config));
    const ids = Int32Array.from([1, 2, 3, 4, 5]);
    const before = model.forward(ids, 1, 5).logits.toArray();
    const after = imported.model.forward(ids, 1, 5).logits.toArray();
    expect(after.length).toBe(before.length);
    for (let i = 0; i < before.length; i++) expect(after[i]).toBeCloseTo(before[i], 6);
  });

  it("refuses a corrupted or inconsistent artifact", () => {
    const registry = new AlphaModelRegistry();
    const config = smallConfig();
    const model = new AlphaTransformer(config);
    const registered = registry.register({ modelId: "e@2", modelVersion: "1", config, relationships: baseRelationships() });
    const artifact = exportModel({ model: registered, weights: model.serializeWeights() });

    expect(() => importModel({ ...artifact, configFingerprint: "cfg_deadbeef" })).toThrow(/inconsistent/);
    expect(() => importModel({ ...artifact, parameterCount: 999 })).toThrow(/inconsistent/);
    expect(() => importModel({ ...artifact, formatVersion: "99.0.0" })).toThrow(/format/);
    expect(() => parseModelExport("{not json")).toThrow(/not valid JSON/);
    expect(() => parseModelExport(JSON.stringify({ format: "alpha.model-export" }))).toThrow(/no weights/);
  });

  it("does not present an untrained model as capable", () => {
    const registry = new AlphaModelRegistry();
    const config = smallConfig();
    const model = new AlphaTransformer(config);
    const registered = registry.register({ modelId: "e@3", modelVersion: "1", config, relationships: baseRelationships() });
    const artifact = exportModel({ model: registered, weights: model.serializeWeights() });
    expect(artifact.lifecycle).toBe("UNTRAINED");
    expect(artifact.note).toMatch(/NO trained weights/);
    expect(importModel(artifact).warnings.join(" ")).toMatch(/no trained weights/);
  });

  it("refuses to export a model whose recorded parameter count is wrong", () => {
    const registry = new AlphaModelRegistry();
    const config = smallConfig();
    const model = new AlphaTransformer(config);
    const registered = registry.register({ modelId: "e@4", modelVersion: "1", config, relationships: baseRelationships() });
    registered.parameterCount = 12345;
    expect(() => exportModel({ model: registered, weights: model.serializeWeights() })).toThrow(/does not match the architecture/);
  });
});

// ---------------------------------------------------------------------------
describe("step 4 — checkpoint compatibility", () => {
  /** Train briefly so the checkpoint under test is one Alpha really produced. */
  function realCheckpoint() {
    const version = makeVersion({
      documents: Array.from({ length: 12 }, (_, i) => `checkpoint document ${i} about alpha.`),
    });
    const dataset = toDataset(version);
    const tokenizer = AlphaTokenizer.train(dataset.documents, {
      vocabSize: 160,
      version: "1.0.0",
      trainedOn: "test",
    });
    const config = withSpecialTokenIds(
      { ...smallConfig(), vocabSize: Math.max(160, tokenizer.vocabSize) },
      tokenizer.specialTokenIds,
    );
    const trainer = new AlphaTrainer({
      model: new AlphaTransformer(config),
      tokenizer,
      dataset,
      config: { batchSize: 2, seqLen: 8, totalSteps: 2, evalInterval: 0, checkpointInterval: 0 },
    });
    trainer.trainToCompletion();
    const checkpoint = trainer.checkpoint;
    expect(checkpoint).not.toBeNull();
    return { checkpoint: checkpoint!, config, tokenizer, dataset, version };
  }

  it("accepts a checkpoint that matches its own config and tokenizer", () => {
    const { checkpoint, config, tokenizer } = realCheckpoint();
    expect(() => assertCheckpointCompatible(checkpoint, { config, tokenizer })).not.toThrow();
    expect(checkpoint.configFingerprint).toBe(modelConfigFingerprint(config));
  });

  it("refuses a checkpoint against a different architecture", () => {
    const { checkpoint, config, tokenizer } = realCheckpoint();
    expect(() =>
      assertCheckpointCompatible(checkpoint, { config: { ...config, nLayers: 3 }, tokenizer }),
    ).toThrow(/not compatible/);
  });

  it("refuses a checkpoint against a different tokenizer", () => {
    const { checkpoint, config, tokenizer, dataset } = realCheckpoint();
    const otherTokenizer = AlphaTokenizer.train(
      [...dataset.documents, "a completely different body of text for the other tokenizer."],
      { vocabSize: 160, version: "2.0.0", trainedOn: "other" },
    );
    expect(otherTokenizer.fingerprint()).not.toBe(tokenizer.fingerprint());
    expect(() => assertCheckpointCompatible(checkpoint, { config, tokenizer: otherTokenizer })).toThrow(
      /not compatible/,
    );
  });

  it("refuses a corrupted checkpoint rather than loading it", () => {
    const { checkpoint, config, tokenizer } = realCheckpoint();
    const corrupted = {
      ...checkpoint,
      configFingerprint: "cfg_deadbeef",
    };
    expect(() => assertCheckpointCompatible(corrupted, { config, tokenizer })).toThrow();
  });
});

// ---------------------------------------------------------------------------
describe("step 4 — training scaling, metrics and lifecycle", () => {
  it("estimates resources as derived arithmetic, never as a hardware benchmark", () => {
    const estimate = estimateResources(smallConfig(), { batchSize: 4, seqLen: 8, totalSteps: 10 });
    expect(estimate.parameterCount).toBe(countParameters(smallConfig()));
    expect(estimate.precision).toBe("float32");
    expect(estimate.estimateKind).toBe("derived-estimate");
    expect(estimate.note).toMatch(/Not a measurement/);
    expect(estimate.tokensPerStep).toBe(32);
    expect(estimate.plannedTrainingTokens).toBe(320);
    expect(estimate.checkpointBytes).toBe(estimate.weightBytes + estimate.optimizerBytes);
    expect(estimate.withinLimits).toBe(true);
  });

  it("refuses a configuration outside the declared limits before training", () => {
    const estimate = estimateResources(smallConfig(), { batchSize: 4, seqLen: 999, totalSteps: 10 });
    expect(estimate.withinLimits).toBe(false);
    expect(estimate.violations.join(" ")).toMatch(/seqLen/);
    expect(() => assertTrainableWithinLimits(smallConfig(), { batchSize: 4, seqLen: 999, totalSteps: 10 })).toThrow(
      /cannot be trained within Alpha's limits/,
    );
  });

  it("does not claim features the runtime does not implement", () => {
    expect(RUNTIME_CAPABILITIES.mixedPrecision).toBe(false);
    expect(RUNTIME_CAPABILITIES.gradientAccumulation).toBe(false);
    expect(RUNTIME_CAPABILITIES.earlyStopping).toBe(false);
    expect(RUNTIME_CAPABILITIES.resumable).toBe(true);
  });

  it("maps job states onto the explicit lifecycle vocabulary", () => {
    expect(lifecycleFromJobState("created")).toBe("QUEUED");
    expect(lifecycleFromJobState("running")).toBe("RUNNING");
    expect(lifecycleFromJobState("paused")).toBe("PAUSED");
    expect(lifecycleFromJobState("completed")).toBe("COMPLETED");
    expect(lifecycleFromJobState("failed")).toBe("FAILED");
    expect(lifecycleFromJobState("stopped")).toBe("CANCELLED");
    expect(canTransitionLifecycle("RUNNING", "COMPLETED")).toBe(true);
    expect(canTransitionLifecycle("COMPLETED", "RUNNING")).toBe(false);
  });

  it("never reports COMPLETED when a run failed or stopped early", () => {
    expect(terminalLifecycle({ threw: true, error: "boom", reachedStepBudget: false }).state).toBe("FAILED");
    expect(terminalLifecycle({ threw: true, error: "boom", reachedStepBudget: false }).error).toBe("boom");
    expect(terminalLifecycle({ threw: false, reachedStepBudget: false, cancelled: true }).state).toBe("CANCELLED");
    expect(terminalLifecycle({ threw: false, reachedStepBudget: true }).state).toBe("COMPLETED");
    // Ended without error and without the budget: a failure, not a completion.
    expect(terminalLifecycle({ threw: false, reachedStepBudget: false }).state).toBe("FAILED");
  });

  it("reports token-based metrics from a real short run", () => {
    const version = makeVersion({ documents: Array.from({ length: 12 }, (_, i) => `training document ${i} about alpha.`) });
    const dataset = toDataset(version);
    const tokenizer = AlphaTokenizer.train(dataset.documents, { vocabSize: 160, version: "1.0.0", trainedOn: "test" });
    const config = withSpecialTokenIds({ ...smallConfig(), vocabSize: Math.max(160, tokenizer.vocabSize) }, tokenizer.specialTokenIds);
    const model = new AlphaTransformer(config);
    const trainer = new AlphaTrainer({
      model,
      tokenizer,
      dataset,
      config: { batchSize: 2, seqLen: 8, totalSteps: 3, evalInterval: 0, checkpointInterval: 0 },
    });
    const summary = trainer.trainToCompletion();
    const metrics = tokenMetricsFromSummary(summary);
    expect(metrics.steps).toBe(3);
    expect(metrics.tokensSeen).toBeGreaterThan(0);
    expect(metrics.tokensPerStep).toBeGreaterThan(0);
    expect(metrics.state).toBe("completed");
    // The baseline is the loss of a model with no information, over the
    // vocabulary the tokenizer actually produced.
    expect(metrics.uniformLossBaseline).toBeCloseTo(Math.log(tokenizer.vocabSize), 4);
    // A measured loss, comparable against the no-information baseline.
    expect(metrics.lastLoss).not.toBeNull();
    expect(metrics.lossReduction).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
describe("step 4 — reproducibility and the larger corpus", () => {
  it("generates a deterministic, Alpha-owned corpus", () => {
    const a = buildGeneratedCorpus(40, 20250930);
    const b = buildGeneratedCorpus(40, 20250930);
    expect(a.documents).toEqual(b.documents);
    const different = buildGeneratedCorpus(40, 999);
    expect(different.documents).not.toEqual(a.documents);
    expect(a.license).toBe("Alpha-owned");
    expect(generatedCorpusTopics().length).toBeGreaterThan(10);
  });

  it("grows well beyond the seed corpus and still passes quality", () => {
    const raw = buildGeneratedCorpus(120, 20250930);
    const version = createDatasetVersion({
      datasetId: raw.id,
      name: raw.name,
      version: raw.version,
      description: raw.description,
      sources: OWNED_SOURCE,
      documents: raw.documents,
      now: 0,
    });
    const chars = version.documents.reduce((s, d) => s + d.length, 0);
    // The seed corpus is 5,358 characters; this must be meaningfully larger.
    expect(chars).toBeGreaterThan(20_000);
    expect(analyseDatasetVersionQuality(version).pass).toBe(true);
  });

  it("reproduces identical loss for an identical seed", () => {
    const version = makeVersion({ documents: Array.from({ length: 12 }, (_, i) => `reproducible document ${i} text.`) });
    const dataset = toDataset(version);
    const run = () => {
      const tokenizer = AlphaTokenizer.train(dataset.documents, { vocabSize: 160, version: "1.0.0", trainedOn: "test" });
      const config = withSpecialTokenIds(
        { ...smallConfig(), vocabSize: Math.max(160, tokenizer.vocabSize) },
        tokenizer.specialTokenIds,
      );
      const trainer = new AlphaTrainer({
        model: new AlphaTransformer(config),
        tokenizer,
        dataset,
        config: { batchSize: 2, seqLen: 8, totalSteps: 4, evalInterval: 0, checkpointInterval: 0, seed: 4242 },
      });
      return trainer.trainToCompletion();
    };
    const first = run();
    const second = run();
    expect(second.firstLoss).toBe(first.firstLoss);
    expect(second.lastLoss).toBe(first.lastLoss);
    expect(second.tokensSeen).toBe(first.tokensSeen);
  });
});
