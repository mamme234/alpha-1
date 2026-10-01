/**
 * Experiment records.
 *
 * An experiment is the answer to "what exactly was run, and what did it
 * produce". Without one, a capability claim cannot be checked: there is no way
 * to know whether two numbers came from the same seed, the same corpus, or the
 * same tokenizer — and if they did not, they are not comparable.
 *
 * The record is deliberately complete and deliberately boring. It captures the
 * identity of everything that varies (model, tokenizer, dataset, seed, config),
 * the measurements that came out (losses, perplexity, token counts, timing, raw
 * capability results), and the artifact that can be reloaded (checkpoint).
 *
 * Two rules:
 *
 *   1. **Nothing is invented.** A field that has no measured value is `null`,
 *      never a plausible-looking default. `failureReason` is required when the
 *      status is failed, so a failed run is recorded as a failure rather than
 *      quietly absent from the list.
 *
 *   2. **Reproducible.** `describeReproduction` returns the exact inputs needed
 *      to re-run the experiment. If it cannot be re-run, the record is not an
 *      experiment, it is a note.
 */

import { AlphaValidationError } from "../core/errors";
import type { AlphaModelConfig } from "../model/config";
import type { TrainingConfig, TrainingSummary, EarlyStoppingReport } from "./trainer";
import type { AlphaCheckpoint } from "./checkpoint";
import type { CapabilityReport } from "../evaluation/runner";

export type ExperimentStatus = "running" | "completed" | "failed" | "aborted";

export const EXPERIMENT_STATUSES: ExperimentStatus[] = [
  "running",
  "completed",
  "failed",
  "aborted",
];

export type ExperimentMeasurements = {
  steps: number;
  tokensSeen: number;
  tokensPerStep: number;
  firstLoss: number | null;
  lastLoss: number | null;
  bestLoss: number | null;
  validationLoss: number | null;
  validationPerplexity: number | null;
  uniformLossBaseline: number;
  durationMs: number;
  tokensPerSecond: number;
  checkpointCount: number;
  gradientAccumulationSteps: number;
  earlyStopping: EarlyStoppingReport | null;
  /** Steps never taken because early stopping ended the run. */
  stepsSkippedByEarlyStopping: number;
};

export type Experiment = {
  experimentId: string;
  /** Human label, e.g. "step5 improved-corpus micro". */
  label: string;
  status: ExperimentStatus;
  /** Present only when `status` is failed or aborted. */
  failureReason: string | null;
  createdAt: number;
  /** Epoch ms the run ended, or null while it is still running. */
  endedAt: number | null;

  // --- identity of what varied ---------------------------------------------
  parent: {
    /** Parent model id and version, when this run continued from one. */
    modelId: string | null;
    modelVersion: string | null;
    /** Checkpoint the run started from, when it was a continuation. */
    checkpointId: string | null;
    parameterCount: number | null;
  };
  candidate: {
    modelId: string;
    modelVersion: string;
    preset: string;
    config: AlphaModelConfig;
    configFingerprint: string;
    parameterCount: number;
  };
  dataset: {
    id: string;
    name: string;
    version: string;
    fingerprint: string;
    license: string;
    documents: number;
    characters: number;
    trainDocuments: number;
    validationDocuments: number;
    testDocuments: number;
    mixtureFingerprint: string | null;
  };
  tokenizer: {
    version: string;
    fingerprint: string;
    vocabSize: number;
    trainedOn: string;
    /** What the tokenizer measurement said, when it was measured. */
    measured: {
      tokensPerCharacter: number | null;
      vocabularyCoverage: number | null;
      unknownTokenShare: number | null;
      retrained: boolean;
      reason: string | null;
    } | null;
  };
  /** The training configuration exactly as run. */
  trainingConfig: TrainingConfig;
  seed: number;

  // --- what happened --------------------------------------------------------
  measurements: ExperimentMeasurements | null;
  /** Raw capability results against the frozen suite. Null if not yet evaluated. */
  capability: CapabilityReport | null;
  /** Fingerprint of the evaluation suite used. Required once capability exists. */
  evaluationSuiteFingerprint: string | null;
  checkpoint: {
    id: string;
    step: number;
    sizeBytes: number;
    createdAt: number;
  } | null;
};

export type CreateExperimentInput = {
  experimentId: string;
  label: string;
  parent?: Partial<Experiment["parent"]>;
  candidate: {
    modelId: string;
    modelVersion: string;
    preset: string;
    config: AlphaModelConfig;
    configFingerprint: string;
    parameterCount: number;
  };
  dataset: Experiment["dataset"];
  tokenizer: Experiment["tokenizer"];
  trainingConfig: TrainingConfig;
  seed: number;
  now?: number;
};

/** Open an experiment record. Status starts as `running`. */
export function createExperiment(input: CreateExperimentInput): Experiment {
  if (!input.experimentId) {
    throw new AlphaValidationError("training", "an experiment needs an experimentId");
  }
  if (!input.label) {
    throw new AlphaValidationError("training", `experiment ${input.experimentId} needs a label`);
  }
  if (!input.dataset.fingerprint) {
    throw new AlphaValidationError(
      "training",
      `experiment ${input.experimentId} cannot be recorded without a dataset fingerprint; a result with no dataset identity cannot be reproduced`,
      { experimentId: input.experimentId },
    );
  }
  if (!input.tokenizer.fingerprint) {
    throw new AlphaValidationError(
      "training",
      `experiment ${input.experimentId} cannot be recorded without a tokenizer fingerprint`,
      { experimentId: input.experimentId },
    );
  }
  return {
    experimentId: input.experimentId,
    label: input.label,
    status: "running",
    failureReason: null,
    createdAt: input.now ?? Date.now(),
    endedAt: null,
    parent: {
      modelId: null,
      modelVersion: null,
      checkpointId: null,
      parameterCount: null,
      ...input.parent,
    },
    candidate: { ...input.candidate },
    dataset: { ...input.dataset },
    tokenizer: { ...input.tokenizer },
    trainingConfig: { ...input.trainingConfig },
    seed: input.seed,
    measurements: null,
    capability: null,
    evaluationSuiteFingerprint: null,
    checkpoint: null,
  };
}

/** Derive measurements from a real `TrainingSummary`. Every field is measured. */
export function recordTraining(experiment: Experiment, summary: TrainingSummary): Experiment {
  if (experiment.status !== "running") {
    throw new AlphaValidationError(
      "training",
      `experiment ${experiment.experimentId} is ${experiment.status} and cannot record another training run`,
      { experimentId: experiment.experimentId, status: experiment.status },
    );
  }
  experiment.measurements = {
    steps: summary.steps,
    tokensSeen: summary.tokensSeen,
    tokensPerStep: summary.tokensPerStep,
    firstLoss: summary.firstLoss,
    lastLoss: summary.lastLoss,
    bestLoss: summary.bestLoss,
    validationLoss: summary.validationLoss,
    validationPerplexity: summary.validationPerplexity,
    uniformLossBaseline: summary.uniformLossBaseline,
    durationMs: summary.durationMs,
    tokensPerSecond: summary.throughputTokensPerSecond,
    checkpointCount: summary.checkpointCount,
    gradientAccumulationSteps: summary.gradientAccumulationSteps,
    earlyStopping: summary.earlyStopping ? { ...summary.earlyStopping } : null,
    stepsSkippedByEarlyStopping: summary.stepsSkippedByEarlyStopping,
  };
  if (summary.checkpoint) {
    experiment.checkpoint = {
      id: summary.checkpoint.id,
      step: summary.checkpoint.step,
      sizeBytes: summary.checkpoint.sizeBytes,
      createdAt: summary.checkpoint.createdAt,
    };
  }
  return experiment;
}

/**
 * Attach capability results.
 *
 * The suite fingerprint is recorded with them so a later comparison can prove
 * both runs were measured against the *same* evaluation — which is the only way
 * a comparison means anything.
 */
export function recordCapability(
  experiment: Experiment,
  report: CapabilityReport,
  suiteFingerprint: string,
): Experiment {
  experiment.capability = report;
  experiment.evaluationSuiteFingerprint = suiteFingerprint;
  return experiment;
}

/** Close an experiment successfully. */
export function completeExperiment(experiment: Experiment, now = Date.now()): Experiment {
  if (experiment.status !== "running") {
    throw new AlphaValidationError(
      "training",
      `experiment ${experiment.experimentId} is ${experiment.status}, not running`,
    );
  }
  experiment.status = "completed";
  experiment.endedAt = now;
  return experiment;
}

/**
 * Fail an experiment, recording why.
 *
 * A failed run keeps every measurement it did produce. Hiding partial results
 * from failed experiments is how a record starts to flatter the model.
 */
export function failExperiment(
  experiment: Experiment,
  reason: string,
  now = Date.now(),
): Experiment {
  if (!reason || reason.trim().length === 0) {
    throw new AlphaValidationError(
      "training",
      `experiment ${experiment.experimentId} failed and no reason was given; a failure without a reason cannot be audited`,
      { experimentId: experiment.experimentId },
    );
  }
  experiment.status = "failed";
  experiment.failureReason = reason;
  experiment.endedAt = now;
  return experiment;
}

/** Abort an experiment (operator decision, not an error). */
export function abortExperiment(
  experiment: Experiment,
  reason: string,
  now = Date.now(),
): Experiment {
  if (!reason) {
    throw new AlphaValidationError("training", "aborting an experiment requires a reason");
  }
  experiment.status = "aborted";
  experiment.failureReason = reason;
  experiment.endedAt = now;
  return experiment;
}

/**
 * The inputs needed to reproduce this experiment exactly.
 *
 * Returns a string rather than an object because the point is that a human can
 * read it and re-run it.
 */
export function describeReproduction(experiment: Experiment): string {
  const cfg = experiment.trainingConfig;
  return [
    `experiment:  ${experiment.experimentId} (${experiment.label})`,
    `model:       ${experiment.candidate.modelId} [${experiment.candidate.preset}] ` +
      `${experiment.candidate.parameterCount.toLocaleString()} params, config ${experiment.candidate.configFingerprint}`,
    `dataset:     ${experiment.dataset.name}@${experiment.dataset.version} ` +
      `fp ${experiment.dataset.fingerprint} (${experiment.dataset.documents} docs, ${experiment.dataset.characters.toLocaleString()} chars)`,
    `tokenizer:   v${experiment.tokenizer.version} fp ${experiment.tokenizer.fingerprint} ` +
      `(${experiment.tokenizer.vocabSize} tokens, trained on ${experiment.tokenizer.trainedOn})`,
    `seed:        ${experiment.seed}`,
    `training:    batch ${cfg.batchSize} x seq ${cfg.seqLen} x accum ${cfg.gradientAccumulationSteps} ` +
      `for ${cfg.totalSteps} steps · lr ${cfg.learningRate} ${cfg.schedule} ` +
      `warmup ${cfg.warmupSteps} · wd ${cfg.weightDecay} · clip ${cfg.gradClipNorm} · ` +
      `batchMode ${cfg.batchMode} · eval every ${cfg.evalInterval} · checkpoint every ${cfg.checkpointInterval}`,
    cfg.earlyStopping
      ? `early stop:  ${cfg.earlyStopping.monitor}, patience ${cfg.earlyStopping.patience}, ` +
        `minDelta ${cfg.earlyStopping.minDelta}, minSteps ${cfg.earlyStopping.minSteps}`
      : "early stop: disabled",
    `evaluation:  suite ${experiment.evaluationSuiteFingerprint ?? "not yet evaluated"}`,
    experiment.checkpoint
      ? `checkpoint:  ${experiment.checkpoint.id} at step ${experiment.checkpoint.step}`
      : "checkpoint: none",
  ].join("\n");
}

/** One-line summary: status and the headline measurements, no verdict. */
export function summariseExperiment(experiment: Experiment): string {
  const m = experiment.measurements;
  const loss = m?.lastLoss == null ? "n/a" : m.lastLoss.toFixed(4);
  const val = m?.validationLoss == null ? "n/a" : m.validationLoss.toFixed(4);
  const ppl = m?.validationPerplexity == null ? "n/a" : m.validationPerplexity.toFixed(2);
  const base = ` ${experiment.experimentId} [${experiment.status}] ${experiment.label} — ` +
    `${experiment.candidate.parameterCount.toLocaleString()} params · ` +
    `${m ? m.tokensSeen.toLocaleString() : 0} tokens · train ${loss} · val ${val} · ppl ${ppl}`;
  return experiment.failureReason ? `${base} · FAILED: ${experiment.failureReason}` : base;
}

/**
 * Compare two experiments' capability reports on the measurements they share.
 *
 * Measurements only. No aggregation, no weighting, no winner: each row says what
 * moved and in which direction, and a reader decides what that means. Returns
 * `null` for a row when either side did not measure it, because a comparison
 * against a missing number is not a comparison.
 */
export type CapabilityComparisonRow = {
  measurement: string;
  /** Lower values are better for this measurement. */
  lowerIsBetter: boolean;
  baseline: number | null;
  candidate: number | null;
  delta: number | null;
  /** Relative change of candidate against baseline. null if baseline is 0/null. */
  relative: number | null;
  improved: boolean | null;
  unit: string;
};

export function compareCapability(
  baseline: CapabilityReport,
  candidate: CapabilityReport,
): CapabilityComparisonRow[] {
  const rows: Array<{
    measurement: string;
    lowerIsBetter: boolean;
    unit: string;
    get: (r: CapabilityReport) => number | null;
  }> = [
    {
      measurement: "languageModeling.loss",
      lowerIsBetter: true,
      unit: "nats/token",
      get: (r) => r.languageModeling.loss,
    },
    {
      measurement: "languageModeling.perplexity",
      lowerIsBetter: true,
      unit: "exp(loss)",
      get: (r) => r.languageModeling.perplexity,
    },
    {
      measurement: "languageModeling.nextTokenTop1Accuracy",
      lowerIsBetter: false,
      unit: "fraction",
      get: (r) => r.languageModeling.nextTokenTop1Accuracy,
    },
    {
      measurement: "completion.meanContinuationNll",
      lowerIsBetter: true,
      unit: "nats/token",
      get: (r) => r.categories.find((c) => c.category === "completion")?.meanContinuationNll ?? null,
    },
    {
      measurement: "completion.meanContinuationTop1",
      lowerIsBetter: false,
      unit: "fraction",
      get: (r) =>
        r.categories.find((c) => c.category === "completion")?.meanContinuationTop1 ?? null,
    },
    {
      measurement: "question-answering.meanContinuationNll",
      lowerIsBetter: true,
      unit: "nats/token",
      get: (r) =>
        r.categories.find((c) => c.category === "question-answering")?.meanContinuationNll ?? null,
    },
    {
      measurement: "question-answering.meanContinuationTop1",
      lowerIsBetter: false,
      unit: "fraction",
      get: (r) =>
        r.categories.find((c) => c.category === "question-answering")?.meanContinuationTop1 ??
        null,
    },
    {
      measurement: "instruction-following.meanContinuationNll",
      lowerIsBetter: true,
      unit: "nats/token",
      get: (r) =>
        r.categories.find((c) => c.category === "instruction-following")?.meanContinuationNll ??
        null,
    },
    {
      measurement: "summarization.meanContinuationNll",
      lowerIsBetter: true,
      unit: "nats/token",
      get: (r) =>
        r.categories.find((c) => c.category === "summarization")?.meanContinuationNll ?? null,
    },
    {
      measurement: "context-retention.meanContinuationNll",
      lowerIsBetter: true,
      unit: "nats/token",
      get: (r) =>
        r.categories.find((c) => c.category === "context-retention")?.meanContinuationNll ?? null,
    },
    {
      measurement: "context-retention.meanContinuationTop1",
      lowerIsBetter: false,
      unit: "fraction",
      get: (r) =>
        r.categories.find((c) => c.category === "context-retention")?.meanContinuationTop1 ??
        null,
    },
    {
      measurement: "structured-output.passRate",
      lowerIsBetter: false,
      unit: "fraction",
      get: (r) =>
        r.categories.find((c) => c.category === "structured-output")?.formatPassed
          ? (r.categories.find((c) => c.category === "structured-output")!.formatPassed!.value /
              Math.max(1, r.categories.find((c) => c.category === "structured-output")!.formatPassed!.total))
          : null,
    },
    {
      measurement: "generation-quality.meanRepetitionRatio",
      lowerIsBetter: true,
      unit: "fraction",
      get: (r) =>
        r.categories.find((c) => c.category === "generation-quality")?.meanRepetitionRatio ?? null,
    },
    {
      measurement: "generation-quality.meanDistinctTrigramRatio",
      lowerIsBetter: false,
      unit: "fraction",
      get: (r) =>
        r.categories.find((c) => c.category === "generation-quality")
          ?.meanDistinctTrigramRatio ?? null,
    },
    {
      measurement: "confusions.detected",
      lowerIsBetter: true,
      unit: "count",
      get: (r) => r.counts.confusion.confusions,
    },
  ];

  return rows.map((row) => {
    const a = row.get(baseline);
    const b = row.get(candidate);
    const delta = a === null || b === null ? null : b - a;
    const relative = delta === null || a === null || a === 0 ? null : delta / Math.abs(a);
    const improved = delta === null ? null : row.lowerIsBetter ? delta < 0 : delta > 0;
    return {
      measurement: row.measurement,
      lowerIsBetter: row.lowerIsBetter,
      baseline: a,
      candidate: b,
      delta,
      relative,
      improved,
      unit: row.unit,
    };
  });
}

/** A compact comparison table for a report. */
export function describeComparison(rows: CapabilityComparisonRow[]): string {
  const head = `measurement${" ".repeat(16)}baseline${" ".repeat(10)}candidate${" ".repeat(10)}delta      direction`;
  const lines = [head, "-".repeat(head.length)];
  for (const row of rows) {
    const fmt = (v: number | null) =>
      v === null ? "n/a".padEnd(9) : Math.abs(v) >= 100 ? v.toFixed(2).padEnd(9) : v.toFixed(5).padEnd(9);
    const direction =
      row.improved === null ? "not comparable" : row.improved ? "improved" : "worse/same";
    lines.push(
      `${row.measurement.padEnd(30)}${fmt(row.baseline)}${fmt(row.candidate)}${fmt(row.delta)}   ${direction}`,
    );
  }
  return lines.join("\n");
}
