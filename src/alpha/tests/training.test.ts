import { describe, expect, it } from "vitest";
import { AlphaTokenizer } from "../tokenizer/bpe";
import { AlphaTransformer } from "../model/transformer";
import { ALPHA_MODEL_PRESETS } from "../model/config";
import { AlphaTrainer } from "../training/trainer";
import { defaultSchedule, learningRateAt } from "../training/schedule";
import { AdamW } from "../training/optimizer";
import { parseCheckpoint, checkpointToJson } from "../training/checkpoint";
import { encodeCorpus, BatchSampler } from "../datasets/corpus";
import { seedCorpusSlice } from "../datasets/seed-corpus";
import { Tensor, backward, crossEntropy } from "../core/tensor";

function buildTinySetup() {
  const tokenizer = AlphaTokenizer.train(seedCorpusSlice(10).documents, {
    vocabSize: 220,
    minPairFrequency: 1,
  });
  const config = {
    ...ALPHA_MODEL_PRESETS.nano,
    vocabSize: Math.max(tokenizer.vocabSize, 16),
    contextLength: 32,
    dModel: 48,
    nHeads: 4,
    nLayers: 2,
    dFeedForward: 96,
  };
  const model = new AlphaTransformer(config);
  return { tokenizer, config, model };
}

describe("alpha training engine", () => {
  it("encodes a corpus into train and validation token streams", () => {
    const { tokenizer } = buildTinySetup();
    const corpus = encodeCorpus(seedCorpusSlice(10), tokenizer, { validationFraction: 0.2 });
    expect(corpus.stats.trainTokens).toBeGreaterThan(0);
    expect(corpus.stats.validationTokens).toBeGreaterThan(0);
    expect(corpus.tokenIds.length).toBe(corpus.stats.trainTokens + corpus.stats.validationTokens);
    expect(corpus.license).toContain("CC0");
  });

  it("samples batches whose targets are the inputs shifted by one", () => {
    const { tokenizer } = buildTinySetup();
    const corpus = encodeCorpus(seedCorpusSlice(10), tokenizer);
    const sampler = new BatchSampler(corpus.trainIds, { batchSize: 2, seqLen: 8, seed: 3 });
    const batch = sampler.next();
    expect(batch.input.length).toBe(16);
    expect(batch.batch).toBe(2);
    expect(batch.seqLen).toBe(8);
    for (let b = 0; b < batch.batch; b++) {
      for (let t = 0; t < batch.seqLen - 1; t++) {
        expect(batch.target[b * batch.seqLen + t]).toBe(batch.input[b * batch.seqLen + t + 1]);
      }
    }
  });

  it("reduces loss below the uniform baseline on the seed corpus", () => {
    const { tokenizer, model } = buildTinySetup();
    const trainer = new AlphaTrainer({
      model,
      tokenizer,
      dataset: seedCorpusSlice(10),
      config: {
        batchSize: 4,
        seqLen: 24,
        totalSteps: 40,
        learningRate: 5e-3,
        warmupSteps: 5,
        minFactor: 0.2,
        evalInterval: 0,
        evalBatches: 4,
        checkpointInterval: 0,
        seed: 11,
      },
    });

    const before = trainer.evaluate({ maxBatches: 4 });
    const summary = trainer.trainToCompletion();
    const after = trainer.evaluate({ maxBatches: 4 });

    expect(trainer.step).toBe(40);
    expect(trainer.history.length).toBe(40);
    expect(summary.uniformLossBaseline).toBeCloseTo(Math.log(tokenizer.vocabSize), 5);
    expect(before.loss).toBeGreaterThan(0);
    // The model must actually learn: validation loss moves toward the baseline
    // and the final training loss beats a uniform predictor.
    expect(after.loss).toBeLessThan(before.loss);
    expect(summary.lastLoss).not.toBeNull();
    expect(summary.lastLoss!).toBeLessThan(summary.uniformLossBaseline);
    expect(summary.tokensSeen).toBe(40 * 4 * 24);
  }, 300_000);

  it("builds a checkpoint that round-trips and resumes training", () => {
    const { tokenizer, model } = buildTinySetup();
    const trainer = new AlphaTrainer({
      model,
      tokenizer,
      dataset: seedCorpusSlice(6),
      config: {
        batchSize: 2,
        seqLen: 16,
        totalSteps: 6,
        learningRate: 4e-3,
        warmupSteps: 2,
        evalInterval: 0,
        checkpointInterval: 0,
        evalBatches: 2,
      },
    });
    trainer.trainToCompletion();
    const checkpoint = trainer.buildCheckpoint();

    expect(checkpoint.step).toBe(6);
    expect(checkpoint.stage).toBe("trained");
    expect(checkpoint.tokensSeen).toBe(6 * 2 * 16);
    expect(checkpoint.metrics.validationLoss).not.toBeNull();
    expect(checkpoint.sizeBytes).toBeGreaterThan(0);
    expect(checkpoint.notes.join(" ")).toContain("No external model weights");

    const restored = parseCheckpoint(checkpointToJson(checkpoint));
    expect(restored.step).toBe(6);
    const freshModel = new AlphaTransformer(model.config);
    const resumed = new AlphaTrainer({
      model: freshModel,
      tokenizer,
      dataset: seedCorpusSlice(6),
      config: {
        batchSize: 2,
        seqLen: 16,
        totalSteps: 10,
        learningRate: 4e-3,
        warmupSteps: 2,
        evalInterval: 0,
        checkpointInterval: 0,
        evalBatches: 2,
      },
    });
    expect(resumed.step).toBe(0);
    resumed.resumeFrom(restored);
    expect(resumed.step).toBe(6);
    expect(resumed.optimizer.step).toBe(6);
    const summary = resumed.trainToCompletion();
    expect(summary.steps).toBe(4);
    expect(resumed.step).toBe(10);
  }, 300_000);

  it("schedules the learning rate with warmup and decay", () => {
    const schedule = defaultSchedule(100, 1e-3);
    expect(learningRateAt(0, schedule)).toBe(0);
    expect(learningRateAt(5, schedule)).toBeCloseTo(5e-4, 6); // halfway through warmup
    expect(learningRateAt(100, schedule)).toBeCloseTo(1e-4, 6); // floor at minFactor
    const mid = learningRateAt(50, schedule);
    expect(mid).toBeGreaterThan(1e-4);
    expect(mid).toBeLessThan(1e-3);
  });

  it("applies gradient clipping in AdamW", () => {
    const tensor = Tensor.from(
      [
        [1, 1],
        [1, 1],
      ],
      true,
    );
    const optimizer = new AdamW([{ name: "layer0.mlp.w1", tensor }], {
      learningRate: 0.1,
      gradClipNorm: 1,
      weightDecay: 0,
    });
    tensor.grad = new Float32Array([100, 100, 100, 100]);
    const report = optimizer.stepWithSchedule(0.1);
    expect(report.gradNorm).toBeCloseTo(200, 3);
    expect(report.clippedGradNorm).toBeCloseTo(1, 6);
    // Each parameter moves by at most learningRate * 1 per step.
    expect(Math.abs(tensor.data[0] - 1)).toBeLessThanOrEqual(0.11);
  });

  it("trains through Alpha's own autodiff on a single batch", () => {
    const { tokenizer, model, config } = buildTinySetup();
    const ids = Int32Array.from(tokenizer.encode("alpha trains its own model"));
    const slice = ids.slice(0, Math.min(ids.length - 1, 16));
    const input = slice;
    const target = Int32Array.from(ids.slice(1, 1 + slice.length));
    const optimizer = new AdamW(model.parameters(), { learningRate: 1e-2, weightDecay: 0, gradClipNorm: 1 });

    const lossAt = () => {
      const forward = model.forward(input, 1, input.length, { training: false });
      return crossEntropy(forward.logits, target, tokenizer.padId).loss;
    };
    const before = lossAt();
    for (let step = 0; step < 12; step++) {
      const forward = model.forward(input, 1, input.length, { training: false });
      const result = crossEntropy(forward.logits, target, tokenizer.padId);
      backward(result.tensor);
      optimizer.stepWithSchedule(1e-2);
      optimizer.zeroGrad();
    }
    const after = lossAt();
    expect(config.vocabSize).toBeGreaterThanOrEqual(tokenizer.vocabSize);
    expect(after).toBeLessThan(before);
  }, 300_000);
});
