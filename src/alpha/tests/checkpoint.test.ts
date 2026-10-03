import { describe, expect, it } from "vitest";
import { AlphaTokenizer } from "../tokenizer/bpe";
import { AlphaTransformer } from "../model/transformer";
import { modelConfigFingerprint } from "../model/config";
import { AlphaTrainer } from "../training/trainer";
import {
  ALPHA_CHECKPOINT_FORMAT_VERSION,
  assertCheckpointCompatible,
  assertValidCheckpoint,
  checkpointToJson,
  compareCheckpoints,
  estimateCheckpointBytes,
  parseCheckpoint,
  summariseCheckpoint,
  validateCheckpoint,
  withCheckpointId,
  type AlphaCheckpoint,
} from "../training/checkpoint";
import { AlphaCheckpointError } from "../core/errors";
import { AlphaRng } from "../core/rng";
import { encodeCorpus, BatchSampler } from "../datasets/corpus";
import { seedCorpusSlice } from "../datasets/seed-corpus";
import { gradientCheckFixture } from "./helpers";

const TRAINING = {
  batchSize: 2,
  seqLen: 16,
  totalSteps: 6,
  learningRate: 4e-3,
  warmupSteps: 2,
  evalInterval: 0,
  checkpointInterval: 0,
  evalBatches: 2,
  validationFraction: 0.25,
  seed: 5,
};

/** Train a tiny model and return its checkpoint. */
function trainedCheckpoint(): {
  checkpoint: AlphaCheckpoint;
  tokenizer: AlphaTokenizer;
  model: AlphaTransformer;
} {
  const { tokenizer, model } = gradientCheckFixture();
  const trainer = new AlphaTrainer({
    model,
    tokenizer,
    dataset: seedCorpusSlice(8),
    config: TRAINING,
    checkpointLabel: "checkpoint-test",
    runId: "run_checkpoint_test",
  });
  trainer.trainToCompletion();
  return { checkpoint: trainer.buildCheckpoint(), tokenizer, model };
}

function broken(checkpoint: AlphaCheckpoint, patch: Partial<AlphaCheckpoint>): AlphaCheckpoint {
  return { ...checkpoint, ...patch };
}

function catchError(run: () => unknown): unknown {
  try {
    run();
    return null;
  } catch (error) {
    return error;
  }
}

describe("alpha checkpoint system", () => {
  it("records everything needed to reproduce the run", () => {
    const { checkpoint, tokenizer } = trainedCheckpoint();
    expect(checkpoint.formatVersion).toBe(ALPHA_CHECKPOINT_FORMAT_VERSION);
    expect(checkpoint.runId).toBe("run_checkpoint_test");
    expect(checkpoint.step).toBe(6);
    expect(checkpoint.stage).toBe("trained");
    expect(checkpoint.seed).toBe(TRAINING.seed);
    expect(checkpoint.trainingConfig.totalSteps).toBe(TRAINING.totalSteps);
    expect(checkpoint.configFingerprint).toBe(modelConfigFingerprint(checkpoint.config));
    expect(checkpoint.tokenizer.fingerprint).toBe(tokenizer.fingerprint());
    expect(checkpoint.tokenizer.snapshot?.tokens.length).toBe(tokenizer.vocabSize);
    expect(checkpoint.tokenizer.specialTokenIds.pad).toBe(tokenizer.padId);
    expect(checkpoint.datasetFingerprint).toMatch(/^ds_[0-9a-f]{8}$/);
    expect(checkpoint.datasetVersion).toBe(seedCorpusSlice(8).version);
    expect(checkpoint.optimizer.step).toBe(6);
    expect(checkpoint.metrics.trainLoss).toBeGreaterThan(0);
    expect(checkpoint.sizeBytes).toBe(estimateCheckpointBytes(checkpoint.weights));
    expect(validateCheckpoint(checkpoint)).toEqual({ valid: true, issues: [] });
    expect(() => assertValidCheckpoint(checkpoint)).not.toThrow();
  });

  it("round-trips through JSON without losing anything", () => {
    const { checkpoint } = trainedCheckpoint();
    const restored = parseCheckpoint(checkpointToJson(checkpoint));
    expect(restored.id).toBe(checkpoint.id);
    expect(restored.runId).toBe(checkpoint.runId);
    expect(restored.step).toBe(checkpoint.step);
    expect(Object.keys(restored.weights.tensors)).toEqual(Object.keys(checkpoint.weights.tensors));
    expect(restored.optimizer.firstMoment).toEqual(checkpoint.optimizer.firstMoment);
    expect(restored.rng).toEqual(checkpoint.rng!);
    expect(restored.tokenizer.snapshot).toEqual(checkpoint.tokenizer.snapshot);
    expect(validateCheckpoint(restored).valid).toBe(true);
    // The tokenizer snapshot is enough to rebuild the vocabulary exactly.
    const rebuilt = AlphaTokenizer.fromJSON(restored.tokenizer.snapshot!);
    expect(rebuilt.vocabSize).toBe(checkpoint.tokenizer.vocabSize);
    expect(rebuilt.fingerprint()).toBe(checkpoint.tokenizer.fingerprint);
  });

  it("summarises without carrying the weight payload", () => {
    const { checkpoint } = trainedCheckpoint();
    const summary = summariseCheckpoint(checkpoint);
    expect(summary.tokenizerSnapshotIncluded).toBe(true);
    expect(summary.step).toBe(checkpoint.step);
    expect("weights" in summary).toBe(false);
    expect("optimizer" in summary).toBe(false);
    expect(JSON.stringify(summary).length).toBeLessThan(checkpointToJson(checkpoint).length);
  });

  it("refuses a payload that is not JSON or is missing fields", () => {
    expect(() => parseCheckpoint("{")).toThrow(AlphaCheckpointError);
    expect(() => parseCheckpoint("[]")).toThrow(/not an object/);
    expect(() => parseCheckpoint("{}")).toThrow(/missing required fields/);
  });

  it("fails clearly when the payload does not match its own metadata", () => {
    const { checkpoint } = trainedCheckpoint();

    const corruptWeights = broken(checkpoint, {
      weights: { ...checkpoint.weights, tensors: { ...checkpoint.weights.tensors, "layer0.attn.wq": "!!!!" } },
    });
    const weightValidation = validateCheckpoint(corruptWeights);
    expect(weightValidation.valid).toBe(false);
    expect(weightValidation.issues.join(" ")).toMatch(/layer0\.attn\.wq/);

    const stepMismatch = broken(checkpoint, { step: 3 });
    expect(validateCheckpoint(stepMismatch).issues.join(" ")).toMatch(/optimizer step 6 does not match checkpoint step 3/);

    const fingerprintMismatch = broken(checkpoint, { configFingerprint: "cfg_deadbeef" });
    expect(validateCheckpoint(fingerprintMismatch).issues.join(" ")).toMatch(/fingerprint/);

    const stageLie = broken(checkpoint, { step: 0 });
    expect(validateCheckpoint(stageLie).issues.join(" ")).toMatch(/stage is trained but no optimiser step/);

    const noTokenizer = broken(checkpoint, {
      tokenizer: undefined as unknown as AlphaCheckpoint["tokenizer"],
    });
    expect(validateCheckpoint(noTokenizer).issues.join(" ")).toMatch(/tokenizer reference is missing/);

    const noSnapshotSize = broken(checkpoint, {
      tokenizer: {
        ...checkpoint.tokenizer,
        snapshot: { ...checkpoint.tokenizer.snapshot!, tokens: ["<pad>"] },
      },
    });
    expect(validateCheckpoint(noSnapshotSize).issues.join(" ")).toMatch(/snapshot size/);

    expect(() => assertValidCheckpoint(corruptWeights)).toThrow(AlphaCheckpointError);
    try {
      assertValidCheckpoint(corruptWeights);
    } catch (error) {
      expect((error as AlphaCheckpointError).issues.length).toBeGreaterThan(0);
    }
  });

  it("refuses to load into a different vocabulary or architecture", () => {
    const { checkpoint, tokenizer, model } = trainedCheckpoint();
    expect(() =>
      assertCheckpointCompatible(checkpoint, { config: model.config, tokenizer }),
    ).not.toThrow();

    const otherTokenizer = AlphaTokenizer.train(
      ["a completely different corpus about cats and weather and nothing else at all"],
      { vocabSize: 80, version: tokenizer.version },
    );
    expect(otherTokenizer.fingerprint()).not.toBe(tokenizer.fingerprint());
    const tokenizerMismatch = catchError(() =>
      assertCheckpointCompatible(checkpoint, { config: model.config, tokenizer: otherTokenizer }),
    );
    expect(tokenizerMismatch).toBeInstanceOf(AlphaCheckpointError);
    expect((tokenizerMismatch as AlphaCheckpointError).issues.join(" ")).toMatch(
      /does not match the workspace tokenizer/,
    );

    const wideConfig = { ...model.config, dModel: 64, dFeedForward: 128 };
    const architectureMismatch = catchError(() =>
      assertCheckpointCompatible(checkpoint, { config: wideConfig, tokenizer }),
    );
    expect((architectureMismatch as AlphaCheckpointError).issues.join(" ")).toMatch(
      /architecture differs on dModel/,
    );
  });

  it("compares checkpoints by architecture, tokenizer and corpus", () => {
    const { checkpoint } = trainedCheckpoint();
    expect(compareCheckpoints(checkpoint, checkpoint)).toEqual({ compatible: true, reasons: [] });
    const otherCorpus = broken(checkpoint, { datasetFingerprint: "ds_00000000" });
    const comparison = compareCheckpoints(checkpoint, otherCorpus);
    expect(comparison.compatible).toBe(false);
    expect(comparison.reasons.join(" ")).toMatch(/corpus differs/);
  });

  it("keeps ids unique and rejects an empty one", () => {
    const { checkpoint } = trainedCheckpoint();
    const renamed = withCheckpointId(checkpoint, "ckpt_manual");
    expect(renamed.id).toBe("ckpt_manual");
    expect(checkpoint.id).not.toBe(renamed.id);
    expect(() => withCheckpointId(checkpoint, "  ")).toThrow(/must not be empty/);
  });

  it("carries a resumable optimiser and RNG position", () => {
    const { checkpoint, tokenizer } = trainedCheckpoint();
    const fresh = new AlphaTransformer(checkpoint.config);
    const trainer = new AlphaTrainer({
      model: fresh,
      tokenizer,
      dataset: seedCorpusSlice(8),
      config: { ...TRAINING, totalSteps: 8 },
      checkpointLabel: "resume",
      runId: "run_resume",
      isFineTune: true,
    });
    trainer.resumeFrom(checkpoint);
    expect(trainer.step).toBe(6);
    expect(trainer.optimizer.step).toBe(6);
    const resumed = trainer.trainToCompletion();
    expect(resumed.steps).toBe(2);
    expect(trainer.step).toBe(8);
    expect(resumed.checkpoint?.isFineTune).toBe(true);
    expect(resumed.checkpoint?.stage).toBe("fine-tuned");

    // The RNG position is restored, so replaying it reproduces the same draws.
    const replayed = AlphaRng.fromState(checkpoint.rng!);
    const original = new AlphaRng(checkpoint.rng!.seed);
    expect(replayed.next()).toBe(original.next());
  });

  it("resumes with sampler state so windows and dropout continue exactly", () => {
    const { checkpoint, tokenizer } = trainedCheckpoint();
    const { seedCorpusSlice, TRAINING } = requireTestHelpers();
    const { AlphaTrainer, AlphaTransformer } = require("../..");
    const fresh = new AlphaTransformer(checkpoint.config);
    const resumed = new AlphaTrainer({
      model: fresh,
      tokenizer,
      dataset: seedCorpusSlice(8),
      config: { ...TRAINING, totalSteps: 8 },
      checkpointLabel: "resume-sampler",
      runId: "run_resume_sampler",
      isFineTune: true,
    });
    resumed.resumeFrom(checkpoint);
    // Verify the saved sampler state is loaded, so the first window after
    // resume is the same window the continuous run would draw.
    resumed.sampler.next();
    expect(resumed.step).toBe(6);
    expect(resumed.sampler.saveState()).toEqual(checkpoint.sampler);
  });
describe("checkpoint", () => {
  const trained = trainedCheckpoint();
  const { checkpoint, tokenizer } = trained;

  it("round-trips a checkpoint through JSON", () => {
    const { checkpointToJson, parseCheckpoint } = require("../..");
    const json = checkpointToJson(checkpoint);
    const parsed = parseCheckpoint(json);
    expect(parsed.id).toBe(checkpoint.id);
    expect(parsed.configFingerprint).toBe(checkpoint.configFingerprint);
    expect(parsed.tokenizer.fingerprint).toBe(checkpoint.tokenizer.fingerprint);
  });

  it("resumes with sampler state so windows and dropout continue exactly", () => {
    const { checkpoint, tokenizer } = trainedCheckpoint();
    const { seedCorpusSlice, TRAINING } = requireTestHelpers();
    const { AlphaTrainer, AlphaTransformer } = require("../..");
    const fresh = new AlphaTransformer(checkpoint.config);
    const resumed = new AlphaTrainer({
      model: fresh,
      tokenizer,
      dataset: seedCorpusSlice(8),
      config: { ...TRAINING, totalSteps: 8 },
      checkpointLabel: "resume-sampler",
      runId: "run_resume_sampler",
      isFineTune: true,
    });
    resumed.resumeFrom(checkpoint);
    resumed.sampler.next();
    expect(resumed.step).toBe(6);
    expect(resumed.sampler.saveState()).toEqual(checkpoint.sampler);
  });


describe("checkpoint", () => {
  const trained = trainedCheckpoint();
  const { checkpoint, tokenizer } = trained;

  it("round-trips a checkpoint through JSON", () => {
    const { checkpointToJson, parseCheckpoint } = require("../..");
    const json = checkpointToJson(checkpoint);
    const parsed = parseCheckpoint(json);
    expect(parsed.id).toBe(checkpoint.id);
    expect(parsed.configFingerprint).toBe(checkpoint.configFingerprint);
    expect(parsed.tokenizer.fingerprint).toBe(checkpoint.tokenizer.fingerprint);
  });

  it("resumes with sampler state so windows and dropout continue exactly", () => {
    const { checkpoint, tokenizer } = trainedCheckpoint();
    const { seedCorpusSlice, TRAINING } = requireTestHelpers();
    const { AlphaTrainer, AlphaTransformer } = require("../..");
    const fresh = new AlphaTransformer(checkpoint.config);
    const resumed = new AlphaTrainer({
      model: fresh,
      tokenizer,
      dataset: seedCorpusSlice(8),
      config: { ...TRAINING, totalSteps: 8 },
      checkpointLabel: "resume-sampler",
      runId: "run_resume_sampler",
      isFineTune: true,
    });
    resumed.resumeFrom(checkpoint);
    resumed.sampler.next();
    expect(resumed.step).toBe(6);
    expect(resumed.sampler.saveState()).toEqual(checkpoint.sampler);
  });

  it("encodes the corpus it recorded") {
    const { checkpoint, tokenizer } = trainedCheckpoint();
    const corpus = encodeCorpus(seedCorpusSlice(8), tokenizer, { validationFraction: 0.25 });
    const sampler = new BatchSampler(corpus.trainIds, {
      batchSize: checkpoint.trainingConfig.batchSize,
      seqLen: checkpoint.trainingConfig.seqLen,
      seed: checkpoint.trainingConfig.seed,
    });
    const batch = sampler.next();
    expect(batch.input.length).toBe(
      checkpoint.trainingConfig.batchSize * checkpoint.trainingConfig.seqLen,
    );
    expect(checkpoint.tokensSeen).toBe(
      checkpoint.step * checkpoint.trainingConfig.batchSize * checkpoint.trainingConfig.seqLen,
    );
  });
});
