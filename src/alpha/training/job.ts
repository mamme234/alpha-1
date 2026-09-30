/**
 * Alpha Training Engine — training jobs.
 *
 * A training *run* is a first-class record, not an anonymous loop. It has a run
 * id, the exact recipe it was started with (model version, tokenizer version,
 * dataset version, configuration snapshot, seed), a lifecycle state, the metrics
 * it reached and the checkpoints it produced.
 *
 * The state machine is pure and total: every transition is either allowed or
 * rejected with a message, so a paused run can never be "completed" without
 * resuming, and a failed run can be resumed explicitly rather than silently
 * restarted from scratch.
 */

import { AlphaValidationError } from "../core/errors";
import { alphaId } from "../core/types";
import type { TrainingConfig, TrainingMetricPoint } from "./trainer";
import type { AlphaCheckpoint } from "./checkpoint";

export type TrainingJobState =
  | "created"
  | "running"
  | "paused"
  | "completed"
  | "failed"
  | "stopped";

export const TRAINING_JOB_STATES: TrainingJobState[] = [
  "created",
  "running",
  "paused",
  "completed",
  "failed",
  "stopped",
];

export const TRAINING_JOB_TRANSITIONS: Record<TrainingJobState, TrainingJobState[]> = {
  created: ["running", "failed", "stopped"],
  running: ["paused", "completed", "failed", "stopped"],
  // A paused or failed run can be resumed; it can also be abandoned.
  paused: ["running", "stopped", "failed"],
  failed: ["running", "stopped"],
  completed: [],
  stopped: [],
};

export function canTransitionJob(from: TrainingJobState, to: TrainingJobState): boolean {
  return TRAINING_JOB_TRANSITIONS[from].includes(to);
}

export type TrainingJobStateLabel = "START" | "RUNNING" | "PAUSED" | "COMPLETED" | "FAILED" | "STOPPED";

export function trainingJobStateLabel(state: TrainingJobState): TrainingJobStateLabel {
  switch (state) {
    case "created":
      return "START";
    case "running":
      return "RUNNING";
    case "paused":
      return "PAUSED";
    case "completed":
      return "COMPLETED";
    case "failed":
      return "FAILED";
    case "stopped":
      return "STOPPED";
  }
}

export type AlphaTrainingJob = {
  /** Unique run id — quoted in every checkpoint and log line from the run. */
  id: string;
  state: TrainingJobState;
  modelName: string;
  modelVersion: string;
  tokenizerVersion: string;
  tokenizerFingerprint: string;
  datasetName: string;
  datasetVersion: string;
  datasetFingerprint: string;
  datasetLicense: string;
  /** Trained tokens in the corpus, used to report epochs honestly. */
  corpusTokens: number;
  corpusDocuments: number;
  /** The configuration snapshot the run was started with. */
  config: TrainingConfig;
  seed: number;
  createdAt: number;
  startedAt: number | null;
  updatedAt: number;
  completedAt: number | null;
  step: number;
  totalSteps: number;
  /** Epochs of the corpus consumed, when the sampler is epoch-based. */
  epochs: number | null;
  tokensSeen: number;
  trainLoss: number | null;
  bestLoss: number | null;
  validationLoss: number | null;
  learningRate: number | null;
  /** Checkpoints produced by this run, oldest first. */
  checkpointIds: string[];
  lastCheckpointId: string | null;
  /** Checkpoint this run continued from, when it is a resumed run. */
  resumedFromCheckpointId: string | null;
  /** How many times the run has been resumed. */
  resumes: number;
  error: string | null;
  notes: string[];
};

export type CreateTrainingJobInput = {
  modelName: string;
  modelVersion: string;
  tokenizerVersion: string;
  tokenizerFingerprint: string;
  datasetName: string;
  datasetVersion: string;
  datasetFingerprint: string;
  datasetLicense: string;
  corpusTokens: number;
  corpusDocuments: number;
  config: TrainingConfig;
  seed?: number;
  /** Continue an existing run rather than creating a new one. */
  resumedFromCheckpointId?: string | null;
  id?: string;
  now?: number;
};

export function createTrainingJob(input: CreateTrainingJobInput): AlphaTrainingJob {
  const now = input.now ?? Date.now();
  const resumed = input.resumedFromCheckpointId ?? null;
  return {
    id: input.id ?? alphaId("run"),
    state: "created",
    modelName: input.modelName,
    modelVersion: input.modelVersion,
    tokenizerVersion: input.tokenizerVersion,
    tokenizerFingerprint: input.tokenizerFingerprint,
    datasetName: input.datasetName,
    datasetVersion: input.datasetVersion,
    datasetFingerprint: input.datasetFingerprint,
    datasetLicense: input.datasetLicense,
    corpusTokens: input.corpusTokens,
    corpusDocuments: input.corpusDocuments,
    config: { ...input.config },
    seed: input.seed ?? input.config.seed,
    createdAt: now,
    startedAt: null,
    updatedAt: now,
    completedAt: null,
    step: 0,
    totalSteps: input.config.totalSteps,
    epochs: null,
    tokensSeen: 0,
    trainLoss: null,
    bestLoss: null,
    validationLoss: null,
    learningRate: null,
    checkpointIds: [],
    lastCheckpointId: null,
    resumedFromCheckpointId: resumed,
    // A record that already continues another run has one resume behind it.
    resumes: resumed ? 1 : 0,
    error: null,
    notes: resumed
      ? [`Continues run from checkpoint ${resumed}; it does not start a new model.`]
      : ["Fresh run: weights start from the deterministic initialisation for this seed."],
  };
}

/** Move a job to a new state, rejecting an illegal transition. */
export function transitionJob(
  job: AlphaTrainingJob,
  to: TrainingJobState,
  patch: Partial<AlphaTrainingJob> = {},
  now = Date.now(),
): AlphaTrainingJob {
  if (!canTransitionJob(job.state, to)) {
    throw new AlphaValidationError(
      "training",
      `training run ${job.id} cannot move from ${job.state} to ${to} (allowed: ${TRAINING_JOB_TRANSITIONS[job.state].join(", ") || "none"})`,
      { from: job.state, to },
    );
  }
  const resumed = to === "running" && (job.state === "paused" || job.state === "failed");
  return {
    ...job,
    ...patch,
    state: to,
    updatedAt: now,
    startedAt: to === "running" && job.startedAt === null ? now : job.startedAt,
    completedAt: to === "completed" || to === "stopped" || to === "failed" ? now : job.completedAt,
    resumes: resumed ? job.resumes + 1 : job.resumes,
  };
}

/** Record one optimiser step on the job. */
export function recordJobStep(
  job: AlphaTrainingJob,
  point: TrainingMetricPoint,
  now = Date.now(),
): AlphaTrainingJob {
  if (job.state !== "running") {
    throw new AlphaValidationError("training", `training run ${job.id} is ${job.state}, not running`);
  }
  const tokens = Math.max(job.tokensSeen, point.tokensSeen);
  return {
    ...job,
    step: point.step,
    tokensSeen: tokens,
    trainLoss: point.loss,
    bestLoss: job.bestLoss === null ? point.loss : Math.min(job.bestLoss, point.loss),
    learningRate: point.learningRate,
    // Epochs are passes over the corpus, computed from tokens actually seen.
    epochs:
      job.corpusTokens > 0 ? Number((tokens / job.corpusTokens).toFixed(2)) : null,
    updatedAt: now,
  };
}

export function recordJobEvaluation(
  job: AlphaTrainingJob,
  evaluation: { step: number; loss: number; perplexity: number },
  now = Date.now(),
): AlphaTrainingJob {
  return {
    ...job,
    validationLoss: Number.isFinite(evaluation.loss) ? evaluation.loss : job.validationLoss,
    updatedAt: now,
  };
}

export function recordJobCheckpoint(
  job: AlphaTrainingJob,
  checkpoint: AlphaCheckpoint,
  now = Date.now(),
): AlphaTrainingJob {
  return {
    ...job,
    step: Math.max(job.step, checkpoint.step),
    tokensSeen: Math.max(job.tokensSeen, checkpoint.tokensSeen),
    trainLoss: Number.isFinite(checkpoint.metrics.trainLoss) ? checkpoint.metrics.trainLoss : job.trainLoss,
    validationLoss: checkpoint.metrics.validationLoss ?? job.validationLoss,
    checkpointIds: job.checkpointIds.includes(checkpoint.id)
      ? job.checkpointIds
      : [...job.checkpointIds, checkpoint.id],
    lastCheckpointId: checkpoint.id,
    updatedAt: now,
  };
}

export function failJob(job: AlphaTrainingJob, message: string, now = Date.now()): AlphaTrainingJob {
  return transitionJob(job, "failed", { error: message }, now);
}

export function jobProgress(job: AlphaTrainingJob): number {
  if (job.totalSteps <= 0) return 0;
  return Math.min(1, job.step / job.totalSteps);
}

export function summariseJob(job: AlphaTrainingJob): string {
  const loss = job.trainLoss === null ? "—" : job.trainLoss.toFixed(4);
  const validation = job.validationLoss === null ? "—" : job.validationLoss.toFixed(4);
  return `${job.id} · ${trainingJobStateLabel(job.state)} · step ${job.step}/${job.totalSteps} · train ${loss} · val ${validation} · ${job.checkpointIds.length} checkpoint(s)`;
}

/**
 * Validate a job record, e.g. one loaded back from storage. Returns every
 * problem found so a corrupt record can be reported rather than trusted.
 */
export function validateTrainingJob(job: AlphaTrainingJob): { valid: boolean; issues: string[] } {
  const issues: string[] = [];
  if (!job.id?.trim()) issues.push("run id is empty");
  if (!TRAINING_JOB_STATES.includes(job.state)) issues.push(`unknown state ${job.state}`);
  if (!Number.isInteger(job.step) || job.step < 0) issues.push("step must be a non-negative integer");
  if (!Number.isInteger(job.totalSteps) || job.totalSteps < 1) issues.push("totalSteps must be at least 1");
  if (job.state === "completed" && job.step < job.totalSteps) {
    issues.push(`a completed run must have reached ${job.totalSteps} steps (it recorded ${job.step})`);
  }
  if (job.state === "failed" && !job.error) issues.push("a failed run must record its error");
  if (job.trainLoss !== null && !Number.isFinite(job.trainLoss)) issues.push("trainLoss is not finite");
  if (!Number.isFinite(job.seed)) issues.push("seed is not finite");
  if (!job.config) issues.push("configuration snapshot is missing");
  return { valid: issues.length === 0, issues };
}
