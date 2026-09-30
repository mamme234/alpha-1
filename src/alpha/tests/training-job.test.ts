import { describe, expect, it } from "vitest";
import {
  TRAINING_JOB_TRANSITIONS,
  canTransitionJob,
  createTrainingJob,
  failJob,
  jobProgress,
  recordJobCheckpoint,
  recordJobEvaluation,
  recordJobStep,
  summariseJob,
  trainingJobStateLabel,
  transitionJob,
  validateTrainingJob,
  type AlphaTrainingJob,
} from "../training/job";
import { createTrainingConfig, type TrainingEvent, type TrainingSummary } from "../training/trainer";
import { seedCorpusSlice } from "../datasets/seed-corpus";
import { buildStack } from "./helpers";

const CONFIG = createTrainingConfig({ batchSize: 2, seqLen: 24, totalSteps: 4, evalInterval: 0, checkpointInterval: 0 });

function newJob(): AlphaTrainingJob {
  return createTrainingJob({
    modelName: "alpha-nano",
    modelVersion: "0.1.0",
    tokenizerVersion: "test",
    tokenizerFingerprint: "tok_test",
    datasetName: seedCorpusSlice(4).name,
    datasetVersion: seedCorpusSlice(4).version,
    datasetFingerprint: "ds_test",
    datasetLicense: "CC0-1.0",
    corpusTokens: 1000,
    corpusDocuments: 4,
    config: CONFIG,
    seed: CONFIG.seed,
  });
}

function metric(step: number, loss: number) {
  return {
    step,
    loss,
    perplexity: Math.exp(loss),
    learningRate: 1e-3,
    gradNorm: 0.5,
    updateNorm: 0.01,
    tokensSeen: step * CONFIG.batchSize * CONFIG.seqLen,
    paddingTokens: 0,
    elapsedMs: 1,
  };
}

describe("alpha training job", () => {
  it("starts with a unique run id and a configuration snapshot", () => {
    const a = newJob();
    const b = newJob();
    expect(a.id).not.toBe(b.id);
    expect(a.state).toBe("created");
    expect(trainingJobStateLabel(a.state)).toBe("START");
    expect(a.config).toEqual(CONFIG);
    expect(a.seed).toBe(CONFIG.seed);
    expect(a.totalSteps).toBe(4);
    expect(a.startedAt).toBeNull();
    expect(validateTrainingJob(a)).toEqual({ valid: true, issues: [] });
  });

  it("enforces the lifecycle: no completing a run that never started", () => {
    const job = newJob();
    expect(canTransitionJob("created", "running")).toBe(true);
    expect(canTransitionJob("completed", "running")).toBe(false);
    expect(TRAINING_JOB_TRANSITIONS.completed).toEqual([]);
    expect(() => transitionJob(job, "paused")).toThrow(/cannot move from created to paused/);

    const running = transitionJob(job, "running");
    expect(running.state).toBe("running");
    expect(running.startedAt).not.toBeNull();

    const paused = transitionJob(running, "paused");
    expect(paused.state).toBe("paused");
    const resumed = transitionJob(paused, "running");
    expect(resumed.resumes).toBe(1);

    const completed = transitionJob({ ...resumed, step: 4 }, "completed");
    expect(completed.completedAt).not.toBeNull();
    expect(() => transitionJob(completed, "running")).toThrow(/cannot move from completed to running/);
  });

  it("records steps, evaluations and checkpoints against the run", () => {
    let job = transitionJob(newJob(), "running");
    job = recordJobStep(job, metric(1, 4.2));
    job = recordJobStep(job, metric(2, 3.8));
    job = recordJobStep(job, metric(3, 4.4));
    expect(job.step).toBe(3);
    expect(job.trainLoss).toBeCloseTo(4.4, 6);
    expect(job.bestLoss).toBeCloseTo(3.8, 6);
    expect(job.tokensSeen).toBe(3 * CONFIG.batchSize * CONFIG.seqLen);
    expect(job.epochs).toBeCloseTo(job.tokensSeen / 1000, 2);
    expect(jobProgress(job)).toBeCloseTo(0.75, 6);

    job = recordJobEvaluation(job, { step: 3, loss: 4.1, perplexity: 60 });
    expect(job.validationLoss).toBeCloseTo(4.1, 6);

    job = recordJobCheckpoint(job, {
      id: "ckpt_1",
      step: 3,
      tokensSeen: job.tokensSeen,
      metrics: { trainLoss: 4.4, validationLoss: 4.1, validationPerplexity: 60, uniformLoss: 5.4 },
      runId: job.id,
    } as never);
    expect(job.checkpointIds).toEqual(["ckpt_1"]);
    expect(job.lastCheckpointId).toBe("ckpt_1");
    // Recording the same checkpoint twice does not duplicate the reference.
    job = recordJobCheckpoint(job, {
      id: "ckpt_1",
      step: 3,
      tokensSeen: job.tokensSeen,
      metrics: { trainLoss: 4.4, validationLoss: 4.1, validationPerplexity: 60, uniformLoss: 5.4 },
      runId: job.id,
    } as never);
    expect(job.checkpointIds).toEqual(["ckpt_1"]);
    expect(summariseJob(job)).toContain(job.id);
    expect(summariseJob(job)).toContain("1 checkpoint(s)");
    expect(() => recordJobStep(transitionJob(job, "paused"), metric(4, 4))).toThrow(/not running/);
  });

  it("marks a failed run with its error and refuses to call it complete", () => {
    const failed = failJob(transitionJob(newJob(), "running"), "out of memory");
    expect(failed.state).toBe("failed");
    expect(failed.error).toBe("out of memory");
    expect(validateTrainingJob(failed).valid).toBe(true);
    expect(validateTrainingJob({ ...failed, error: null }).issues.join(" ")).toMatch(/must record its error/);
    expect(validateTrainingJob({ ...failed, step: 2, totalSteps: 4, state: "completed" }).issues.join(" ")).toMatch(
      /must have reached 4 steps/,
    );
  });

  it("runs a real training job to completion in the workspace", async () => {
    const { workspace } = await buildStack();
    const events: TrainingEvent[] = [];
    const iterator = workspace.train({ totalSteps: 4 });
    let next = await iterator.next();
    while (!next.done) {
      events.push(next.value);
      next = await iterator.next();
    }
    const summary = next.value as TrainingSummary;

    expect(summary.state).toBe("completed");
    expect(summary.steps).toBe(4);
    const job = workspace.trainingJob;
    expect(job).not.toBeNull();
    expect(job!.state).toBe("completed");
    expect(job!.step).toBe(4);
    expect(job!.tokensSeen).toBe(4 * job!.config.batchSize * job!.config.seqLen);
    expect(job!.trainLoss).not.toBeNull();
    expect(job!.checkpointIds.length).toBeGreaterThanOrEqual(1);
    expect(job!.lastCheckpointId).toBe(workspace.currentCheckpoint?.id);
    expect(validateTrainingJob(job!).valid).toBe(true);
    // The run record is part of the snapshot the UI reads.
    expect(workspace.snapshot().training.job?.id).toBe(job!.id);
    expect(workspace.snapshot().training.jobSummary).toContain("COMPLETED");
    expect(events.some((event) => event.type === "job")).toBe(true);
    expect(events.filter((event) => event.type === "checkpoint").length).toBeGreaterThanOrEqual(1);
  }, 60_000);

  it("pauses a real run at a step boundary and resumes the same run", async () => {
    const { workspace } = await buildStack();
    const iterator = workspace.train({ totalSteps: 8, batchSize: 2, seqLen: 24 });
    let next = await iterator.next();
    let steps = 0;
    while (!next.done) {
      if (next.value.type === "step" && ++steps === 3) {
        // Pause takes effect at the next boundary.
        expect(workspace.pauseTraining()).not.toBeNull();
      }
      next = await iterator.next();
    }
    const paused = next.value as TrainingSummary;
    expect(paused.state).toBe("paused");
    expect(paused.steps).toBe(3);
    expect(workspace.trainingJob?.state).toBe("paused");
    expect(workspace.trainingJob?.step).toBe(3);

    // A pause writes a checkpoint, so the run survives losing the process.
    const checkpoint = workspace.currentCheckpoint;
    expect(checkpoint).not.toBeNull();
    expect(checkpoint!.step).toBe(3);

    workspace.resumeTraining();
    const second = workspace.train();
    let resumedNext = await second.next();
    while (!resumedNext.done) resumedNext = await second.next();
    const resumed = resumedNext.value as TrainingSummary;
    expect(resumed.state).toBe("completed");
    expect(workspace.trainingJob?.state).toBe("completed");
    expect(workspace.trainingJob?.step).toBe(8);
    expect(workspace.trainingJob?.resumes).toBe(1);
    expect(workspace.trainingJob?.id).toBe(paused.jobId);
    expect(workspace.trainingJob!.checkpointIds.length).toBeGreaterThanOrEqual(2);
  }, 60_000);

  it("stops a run for good and starts a new one afterwards", async () => {
    const { workspace } = await buildStack();
    const iterator = workspace.train({ totalSteps: 500, batchSize: 2, seqLen: 24 });
    let next = await iterator.next();
    let steps = 0;
    while (!next.done) {
      if (next.value.type === "step" && ++steps === 2) workspace.stopTraining();
      next = await iterator.next();
    }
    const stopped = next.value as TrainingSummary;
    expect(stopped.state).toBe("stopped");
    expect(workspace.trainingJob?.state).toBe("stopped");
    expect(workspace.trainingJob?.completedAt).not.toBeNull();
    // A stop still preserves the work as a checkpoint...
    expect(workspace.currentCheckpoint).not.toBeNull();
    const preserved = workspace.currentCheckpoint!.id;

    // ...but a stopped run is not resumable: training again starts a new run
    // record that continues from those weights.
    const previousId = workspace.trainingJob!.id;
    const fresh = workspace.train({ totalSteps: 2, batchSize: 2, seqLen: 24 });
    let freshNext = await fresh.next();
    while (!freshNext.done) freshNext = await fresh.next();
    expect((freshNext.value as TrainingSummary).state).toBe("completed");
    expect(workspace.trainingJob?.id).not.toBe(previousId);
    expect(workspace.trainingJob?.resumedFromCheckpointId).toBe(preserved);
    expect(workspace.trainingJob?.step).toBe(2);
  }, 60_000);
});
