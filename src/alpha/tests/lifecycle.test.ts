import { describe, expect, it } from "vitest";
import { formatLifecycleReport, runAlphaTrainingLifecycle } from "../training/lifecycle";
import { verifyAlphaModel, type VerificationReport } from "../training/verify";
import { buildModel, buildTokenizer } from "./helpers";
import { seedCorpusSlice } from "../datasets/seed-corpus";

const SMALL_RUN = {
  batchSize: 4,
  seqLen: 16,
  totalSteps: 20,
  warmupSteps: 3,
  learningRate: 3e-3,
  evalInterval: 10,
  evalBatches: 2,
  checkpointInterval: 10,
  validationFraction: 0.2,
  seed: 4242,
};

describe("alpha lifecycle: corpus → tokens → transformer → loss → gradients → AdamW → checkpoint → reload → resume → generated tokens", () => {
  it("runs the whole pipeline for real and records what happened", () => {
    const report = runAlphaTrainingLifecycle({
      preset: "nano",
      training: SMALL_RUN,
      runVerification: true,
      prompt: "Alpha is a self owned",
    });

    expect(report.ok).toBe(true);
    expect(report.stages.map((stage) => stage.name)).toEqual([
      "initialise",
      "corpus",
      "train",
      "checkpoint",
      "reload",
      "resume",
      "inference",
      "verification",
    ]);
    expect(report.stages.every((stage) => stage.ok)).toBe(true);

    // --- corpus: real counts, both splits populated -----------------------
    expect(report.corpus.documents).toBeGreaterThan(0);
    expect(report.corpus.characters).toBeGreaterThan(0);
    expect(report.corpus.tokens).toBe(report.corpus.trainTokens + report.corpus.validationTokens);
    expect(report.corpus.trainTokens).toBeGreaterThan(0);
    expect(report.corpus.validationTokens).toBeGreaterThan(0);
    expect(report.corpus.vocabularySize).toBe(report.tokenizer.vocabSize);
    expect(report.corpus.trainExamples).toBeGreaterThan(0);
    expect(report.corpus.validationExamples).toBeGreaterThan(0);
    expect(report.corpus.sequenceLength).toBe(SMALL_RUN.seqLen);
    expect(report.corpus.batchSize).toBe(SMALL_RUN.batchSize);

    // --- training: real steps, real loss, real updates --------------------
    expect(report.training.steps).toBe(SMALL_RUN.totalSteps);
    expect(report.training.tokensSeen).toBe(
      SMALL_RUN.totalSteps * SMALL_RUN.batchSize * SMALL_RUN.seqLen,
    );
    expect(report.training.uniformLoss).toBeCloseTo(Math.log(report.tokenizer.vocabSize), 6);
    expect(Number.isFinite(report.training.firstLoss)).toBe(true);
    expect(Number.isFinite(report.training.lastLoss)).toBe(true);
    expect(report.training.lastLoss).toBeLessThan(report.training.uniformLoss);
    expect(report.training.bestLoss).toBeLessThanOrEqual(report.training.firstLoss);
    expect(report.training.meanGradNorm).toBeGreaterThan(0);
    expect(report.stages.find((stage) => stage.name === "train")?.data.parametersMoved).toBe(true);

    // --- checkpoint: valid, self-describing, with the recipe recorded -----
    expect(report.checkpoint.step).toBe(SMALL_RUN.totalSteps);
    expect(report.checkpoint.valid).toBe(true);
    expect(report.checkpoint.validationIssues).toEqual([]);
    expect(report.checkpoint.stage).toBe("trained");
    expect(report.checkpoint.payloadBytes).toBeGreaterThan(0);
    expect(report.checkpoint.jsonBytes).toBeGreaterThan(report.checkpoint.payloadBytes);
    expect(report.checkpoint.tokenizerFingerprint).toBe(report.tokenizer.fingerprint);

    // --- reload: byte-identical weights -----------------------------------
    expect(report.reload.compatible).toBe(true);
    expect(report.reload.maxAbsoluteWeightDifference).toBe(0);

    // --- resume: continues the run, does not restart it -------------------
    expect(report.resume.resumedAtStep).toBe(report.checkpoint.step);
    expect(report.resume.finalStep).toBeGreaterThan(report.checkpoint.step);
    expect(report.resume.stepsThisResume).toBe(2);
    expect(report.resume.optimizerStep).toBe(report.resume.finalStep);

    // --- generation: tokens from Alpha's own weights ----------------------
    expect(report.generation.generatedTokens).toBeGreaterThan(0);
    expect(report.generation.tokenIds.length).toBe(report.generation.generatedTokens);
    expect(report.generation.deterministic).toBe(true);
    expect(report.generation.decodedFromIdsMatches).toBe(true);
    expect(report.generation.stopReason).not.toBe("");
    expect(report.generation.modelStage).toBe("trained");

    // --- verification A–I -------------------------------------------------
    const verification = report.verification as VerificationReport;
    expect(verification).not.toBeNull();
    expect(verification.checks.map((check) => check.id)).toEqual(["A", "B", "C", "D", "E", "F", "G", "H", "I"]);
    expect(verification.checks.filter((check) => !check.passed)).toEqual([]);
    expect(verification.passed).toBe(true);
    const gradientCheck = verification.checks.find((check) => check.id === "C")!;
    expect(gradientCheck.data.gradientCheckPassed).toBe(true);
    expect(Number(gradientCheck.data.nonZeroEntries)).toBeGreaterThan(0);
    const reloadCheck = verification.checks.find((check) => check.id === "G")!;
    expect(reloadCheck.data.maxAbsoluteDifference).toBe(0);

    // The report is a plain JSON document — it can be recorded as evidence.
    const json = JSON.parse(JSON.stringify(report)) as typeof report;
    expect(json.training.lastLoss).toBe(report.training.lastLoss);
    const formatted = formatLifecycleReport(report);
    expect(formatted).toContain("ALPHA LIFECYCLE REPORT");
    expect(formatted).toContain("Loss");
    expect(formatted).toContain("Verification PASSED");
  }, 600_000);

  it("is reproducible: the same configuration produces the same losses and tokens", () => {
    const options = {
      preset: "nano" as const,
      training: { ...SMALL_RUN, totalSteps: 8, checkpointInterval: 0, evalInterval: 0 },
      prompt: "the tokenizer learns merges",
    };
    const first = runAlphaTrainingLifecycle(options);
    const second = runAlphaTrainingLifecycle(options);
    expect(first.ok).toBe(true);
    expect(second.ok).toBe(true);
    expect(second.training.firstLoss).toBe(first.training.firstLoss);
    expect(second.training.lastLoss).toBe(first.training.lastLoss);
    expect(second.generation.tokenIds).toEqual(first.generation.tokenIds);
    expect(second.checkpoint.id).not.toBe(first.checkpoint.id); // ids are unique per run
    expect(second.checkpoint.tokenizerFingerprint).toBe(first.checkpoint.tokenizerFingerprint);
  }, 600_000);

  it("reports verification honestly, including the checks it can fail", () => {
    const tokenizer = buildTokenizer();
    const report = verifyAlphaModel({
      model: buildModel(tokenizer),
      tokenizer,
      dataset: seedCorpusSlice(4),
      training: { batchSize: 2, seqLen: 8, totalSteps: 3, evalInterval: 0, checkpointInterval: 0, validationFraction: 0.5 },
    });
    expect(report.checks.length).toBe(9);
    expect(report.model.parameterCount).toBeGreaterThan(0);
    expect(report.model.configFingerprint).toMatch(/^cfg_[0-9a-f]{8}$/);
    expect(report.tokenizer.fingerprint).toBe(tokenizer.fingerprint());
    expect(report.dataset.fingerprint).toMatch(/^ds_[0-9a-f]{8}$/);
    // A tiny run usually does not beat the uniform baseline; the report says so
    // rather than pretending otherwise.
    if ((report.training.lastLoss ?? 0) >= report.training.uniformLoss) {
      expect(report.notes.join(" ")).toMatch(/uniform baseline/);
    }
    expect(report.generation.deterministic).toBe(true);
    expect(report.durationMs).toBeGreaterThanOrEqual(0);
  }, 600_000);
});
