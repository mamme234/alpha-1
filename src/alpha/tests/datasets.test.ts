import { describe, expect, it } from "vitest";
import {
  assertValidDataset,
  createDataset,
  datasetFingerprint,
  datasetStats,
  splitDocuments,
  validateDataset,
  type AlphaDataset,
} from "../datasets/types";
import {
  BatchSampler,
  DocumentBatchSampler,
  corpusReport,
  countTrainingExamples,
  encodeCorpus,
  padSequences,
} from "../datasets/corpus";
import { ALPHA_SEED_CORPUS, seedCorpusSlice } from "../datasets/seed-corpus";
import { crossEntropy, setGradEnabled } from "../core/tensor";
import { buildModel, buildTokenizer } from "./helpers";

function datasetWith(documents: string[], overrides: Partial<AlphaDataset> = {}): AlphaDataset {
  return {
    id: "ds_test",
    name: "test-corpus",
    version: "1.0.0",
    description: "unit-test corpus",
    license: "CC0-1.0",
    source: "authored for the test suite",
    documents,
    ...overrides,
  };
}

/** Independent softmax + NLL, used to prove the padded loss is masked correctly. */
function meanNllOver(
  logits: Float32Array,
  rows: number,
  vocab: number,
  targets: Int32Array,
  ignore: number,
): { loss: number; counted: number } {
  let sum = 0;
  let counted = 0;
  for (let i = 0; i < rows; i++) {
    const offset = i * vocab;
    const target = targets[i];
    if (target === ignore) continue;
    let max = -Infinity;
    for (let j = 0; j < vocab; j++) max = Math.max(max, logits[offset + j]);
    let total = 0;
    for (let j = 0; j < vocab; j++) total += Math.exp(logits[offset + j] - max);
    const prob = Math.exp(logits[offset + target] - max) / total;
    sum += -Math.log(Math.max(prob, 1e-12));
    counted++;
  }
  return { loss: counted > 0 ? sum / counted : Number.NaN, counted };
}

describe("alpha dataset pipeline", () => {
  it("reports malformed documents instead of silently dropping them", () => {
    const dataset = datasetWith([
      "a perfectly good document with plenty of characters",
      "   ",
      "",
      "another good document that also has enough characters to train on",
    ]);
    const validation = validateDataset(dataset);
    expect(validation.valid).toBe(false);
    expect(validation.issues.map((issue) => issue.index)).toContain(1);
    expect(validation.issues.map((issue) => issue.index)).toContain(2);
    expect(() => assertValidDataset(dataset)).toThrow(/failed validation/);
    expect(() => assertValidDataset(dataset)).toThrow(/#1/);
    expect(() => assertValidDataset(dataset)).toThrow(/whitespace/);
  });

  it("refuses a corpus whose provenance is incomplete", () => {
    expect(validateDataset(datasetWith(["some training text that is long enough to pass"], { license: "" })).issues.map((i) => i.problem).join(" ")).toMatch(
      /licence/,
    );
    expect(validateDataset(datasetWith(["some training text that is long enough to pass"], { name: "" })).valid).toBe(false);
    expect(() =>
      createDataset({ name: "x", version: "1", description: "", license: "CC0", source: "", documents: [] }),
    ).toThrow();
  });

  it("accepts the bundled seed corpus and counts its contents", () => {
    const validation = validateDataset(ALPHA_SEED_CORPUS);
    expect(validation.valid).toBe(true);
    expect(validation.issues).toEqual([]);
    const stats = datasetStats(ALPHA_SEED_CORPUS);
    expect(stats.documents).toBe(20);
    expect(stats.characters).toBeGreaterThan(1000);
    expect(stats.emptyDocuments).toBe(0);
    expect(stats.shortestDocument).toBeGreaterThan(0);
    expect(stats.longestDocument).toBeGreaterThanOrEqual(stats.shortestDocument);
    expect(datasetFingerprint(ALPHA_SEED_CORPUS)).toMatch(/^ds_[0-9a-f]{8}$/);
  });

  it("splits train and validation deterministically with both sides populated", () => {
    const dataset = seedCorpusSlice(10);
    const first = splitDocuments(dataset, 0.2);
    const second = splitDocuments(dataset, 0.2);
    expect(first.train.documents).toEqual(second.train.documents);
    expect(first.validation.documents).toEqual(second.validation.documents);
    expect(first.validation.documents.length).toBeGreaterThan(0);
    expect(first.train.documents.length).toBeGreaterThan(0);
    expect(first.train.documents.length + first.validation.documents.length).toBe(10);
    // Held-out documents are strided, not the last N.
    expect(first.validation.documents).toContain(dataset.documents[0]);
    expect(() => splitDocuments(dataset, 0)).toThrow(/validationFraction/);
    expect(() => splitDocuments(dataset, 1)).toThrow(/validationFraction/);
  });

  it("counts training examples from the token stream", () => {
    expect(countTrainingExamples(100, 32)).toBe(3);
    expect(countTrainingExamples(31, 32)).toBe(0);
    expect(countTrainingExamples(0, 32)).toBe(0);
  });

  it("encodes the corpus and counts unknown characters rather than dropping them", () => {
    const tokenizer = buildTokenizer();
    const dataset = datasetWith([
      ...seedCorpusSlice(4).documents,
      "alpha ✦ ✦ unicode characters outside the trained alphabet",
    ]);
    const corpus = encodeCorpus(dataset, tokenizer, { validationFraction: 0.25 });
    expect(corpus.stats.documents).toBe(5);
    expect(corpus.stats.trainTokens).toBeGreaterThan(0);
    expect(corpus.stats.validationTokens).toBeGreaterThan(0);
    expect(corpus.stats.totalTokens).toBe(corpus.tokenIds.length);
    expect(corpus.stats.vocabularySize).toBe(tokenizer.vocabSize);
    expect(corpus.stats.unknownTokens).toBeGreaterThan(0);
    expect(corpus.trainDocuments.length + corpus.validationDocuments.length).toBe(5);
    expect(corpus.datasetFingerprint).toBe(datasetFingerprint(dataset));
  });

  it("exposes every statistic a run needs in one report", () => {
    const tokenizer = buildTokenizer();
    const report = corpusReport(seedCorpusSlice(10), tokenizer, {
      seqLen: 24,
      batchSize: 4,
      validationFraction: 0.2,
    });
    expect(report.name).toBe("alpha-seed");
    expect(report.documents).toBe(10);
    expect(report.characters).toBeGreaterThan(0);
    expect(report.tokens).toBe(report.trainTokens + report.validationTokens);
    expect(report.vocabularySize).toBe(tokenizer.vocabSize);
    expect(report.sequenceLength).toBe(24);
    expect(report.batchSize).toBe(4);
    expect(report.trainExamples).toBe(countTrainingExamples(report.trainTokens, 24));
    expect(report.validationExamples).toBeGreaterThan(0);
    expect(report.limits.maxDocuments).toBeGreaterThan(0);
    expect(report.padded).toBe(false);
    expect(report.paddingTokensPerBatch).toBeNull();
  });

  it("pads variable-length documents and keeps the padding out of the loss", () => {
    const tokenizer = buildTokenizer();
    const shift = 4;
    const documents = [
      Int32Array.from([shift + 1, shift + 2, shift + 3]),
      Int32Array.from([shift + 4, shift + 5, shift + 6, shift + 7, shift + 8]),
    ];
    const batch = padSequences(documents, { padId: tokenizer.padId, seqLen: 5 });
    expect(batch.batch).toBe(2);
    expect(batch.seqLen).toBe(5);
    // First document only fills two positions; the rest is padding.
    expect(batch.lossMask[0]).toBe(1);
    expect(batch.lossMask[1]).toBe(1);
    expect(batch.lossMask[2]).toBe(0);
    expect(batch.attentionMask[2]).toBe(0);
    expect(batch.target[2]).toBe(tokenizer.padId);
    expect(batch.paddingTokens).toBe(4); // 3 in row 0, 1 in row 1
    expect(Array.from(batch.lossMask).reduce((a, b) => a + b, 0)).toBe(6);

    // The trainer's loss ignores pad targets, so padded positions contribute
    // nothing at all: compare the model's own loss against an independent NLL
    // computed over the real targets only.
    const model = buildModel(tokenizer);
    setGradEnabled(false);
    try {
      const logits = model.forward(batch.input, batch.batch, batch.seqLen, { training: false }).logits;
      const result = crossEntropy(logits, batch.target, tokenizer.padId);
      const independent = meanNllOver(
        logits.data,
        batch.batch * batch.seqLen,
        model.config.vocabSize,
        batch.target,
        tokenizer.padId,
      );
      expect(result.tokens).toBe(6);
      expect(result.loss).toBeCloseTo(independent.loss, 6);
    } finally {
      setGradEnabled(true);
    }
  });

  it("batches documents with a sampler that reports its padding", () => {
    const tokenizer = buildTokenizer();
    const corpus = encodeCorpus(seedCorpusSlice(8), tokenizer, { validationFraction: 0.25 });
    const sampler = new DocumentBatchSampler(corpus.trainDocuments, {
      batchSize: 3,
      seqLen: 16,
      seed: 4,
      padId: tokenizer.padId,
    });
    const batch = sampler.next();
    expect(batch.batch).toBe(3);
    expect(batch.seqLen).toBe(16);
    expect(batch.paddingTokens).toBeGreaterThanOrEqual(0);
    expect(batch.input.length).toBe(48);
    const swept = [...sampler.sequentialBatches()];
    expect(swept.length).toBeGreaterThan(0);
    for (const item of swept) {
      expect(item.paddingTokens).toBe(
        Array.from(item.lossMask).filter((value) => value === 0).length,
      );
    }
  });

  it("reports padding in the corpus report when documents are batched", () => {
    const tokenizer = buildTokenizer();
    const report = corpusReport(seedCorpusSlice(8), tokenizer, {
      seqLen: 32,
      batchSize: 4,
      validationFraction: 0.25,
      padded: true,
    });
    expect(report.padded).toBe(true);
    expect(report.paddingTokensPerBatch).not.toBeNull();
    expect(report.trainExamples).toBeGreaterThan(0);
  });

  it("refuses batch and sequence sizes beyond Alpha's resource limits", () => {
    const tokenizer = buildTokenizer();
    const corpus = encodeCorpus(seedCorpusSlice(4), tokenizer);
    expect(() => new BatchSampler(corpus.trainIds, { batchSize: 5000, seqLen: 8 })).toThrow(
      /exceeds Alpha's configured limit/,
    );
    expect(() => new BatchSampler(corpus.trainIds, { batchSize: 2, seqLen: 100_000 })).toThrow(
      /exceeds Alpha's configured limit/,
    );
    expect(() =>
      corpusReport(seedCorpusSlice(4), tokenizer, { seqLen: 8, batchSize: 5000 }),
    ).toThrow(/exceeds Alpha's configured limit/);
  });

  it("uses the whole window in window mode, so no padding is needed", () => {
    const tokenizer = buildTokenizer();
    const corpus = encodeCorpus(seedCorpusSlice(8), tokenizer);
    const sampler = new BatchSampler(corpus.trainIds, { batchSize: 2, seqLen: 16, seed: 9 });
    const batch = sampler.next();
    expect(batch.paddingTokens).toBe(0);
    expect(Array.from(batch.lossMask).every((value) => value === 1)).toBe(true);
    expect(Array.from(batch.attentionMask).every((value) => value === 1)).toBe(true);
    // Targets are the inputs shifted by one position.
    for (let row = 0; row < batch.batch; row++) {
      for (let t = 0; t < batch.seqLen - 1; t++) {
        expect(batch.target[row * batch.seqLen + t]).toBe(batch.input[row * batch.seqLen + t + 1]);
      }
    }
  });

  it("counts only real targets when a padded batch is used for training", () => {
    const tokenizer = buildTokenizer();
    const model = buildModel(tokenizer);
    const ids = Int32Array.from(tokenizer.encode("alpha trains its own model"));
    const padded = padSequences([ids, ids.slice(0, 6)], { padId: tokenizer.padId, seqLen: 8 });
    setGradEnabled(false);
    try {
      const logits = model.forward(padded.input, padded.batch, padded.seqLen, { training: false }).logits;
      const result = crossEntropy(logits, padded.target, tokenizer.padId);
      expect(result.tokens).toBe(Array.from(padded.lossMask).reduce((a, b) => a + b, 0));
      expect(result.tokens).toBeLessThan(padded.input.length);
    } finally {
      setGradEnabled(true);
    }
  });
});
