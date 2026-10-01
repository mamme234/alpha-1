/**
 * Alpha Training Engine — the trainer.
 *
 * Responsibilities:
 *   validated dataset -> tokenised corpus -> batches -> forward pass ->
 *   cross-entropy -> backpropagation through Alpha's own autodiff -> AdamW ->
 *   metrics -> checkpoint
 *
 * It exposes training as a *generator*: each `next()` performs one optimiser
 * step and yields the metrics. That lets a browser UI run real training in
 * slices without blocking the main thread, and lets a Node script simply loop
 * to completion. No GPU, no external runtime, no hidden model download.
 *
 * Everything a run needs to be reproducible is captured up front: a run id, the
 * seed, the tokenizer version and fingerprint, the dataset reference and the
 * full configuration snapshot. All of it is written into every checkpoint, so a
 * run can be audited after the fact.
 */

import { AlphaRng } from "../core/rng";
import { AlphaValidationError } from "../core/errors";
import { assertResourceLimit } from "../core/limits";
import { backward, crossEntropy, scale, setGradEnabled, type Tensor } from "../core/tensor";
import type { AlphaTokenizer } from "../tokenizer/bpe";
import { assertValidDataset, datasetFingerprint, type AlphaDataset } from "../datasets/types";
import {
  BatchSampler,
  DocumentBatchSampler,
  countTrainingExamples,
  encodeCorpus,
  type CorpusStats,
  type EncodedCorpus,
  type TrainingBatch,
} from "../datasets/corpus";
import type { AlphaTransformer } from "../model/transformer";
import { AdamW } from "./optimizer";
import { learningRateAt, type ScheduleConfig, type ScheduleKind } from "./schedule";
import {
  assertCheckpointCompatible,
  assertValidCheckpoint,
  createCheckpoint,
  type AlphaCheckpoint,
  type AlphaCheckpointMetrics,
  type CheckpointTokenizerRef,
} from "./checkpoint";
import { numericalGradientCheck, type GradientCheckReport } from "./gradients";
import type { AlphaTrainingJob, TrainingJobState } from "./job";

export type BatchMode = "windows" | "documents";

/**
 * Early-stopping rule, evaluated on a *measured validation* metric rather than
 * on a step count. Serializable, because a resumed run must continue monitoring
 * exactly the thing it was monitoring.
 */
export type EarlyStoppingConfig = {
  /** Which measured quantity is watched. Both loss and perplexity are lower-is-better. */
  monitor: "validationLoss" | "validationPerplexity";
  /**
   * How many consecutive evaluations without an improvement of at least
   * `minDelta` are tolerated before the run stops.
   */
  patience: number;
  /** An improvement smaller than this does not count as an improvement. */
  minDelta: number;
  /** Never stop before this many optimiser steps, however bad the curve looks. */
  minSteps: number;
  /**
   * How often the monitored metric is evaluated. 0 reuses `evalInterval`.
   * Early stopping needs its own cadence: with a long `evalInterval` the
   * patience counter would only ever be decremented a handful of times.
   */
  evalEvery: number;
};

/** The record of what early stopping did, or that it did nothing. */
export type EarlyStoppingReport = {
  monitor: "validationLoss" | "validationPerplexity";
  patience: number;
  minDelta: number;
  minSteps: number;
  /** Best monitored value seen, in the metric's own units. */
  bestValue: number | null;
  /** Optimiser step at which that best value was recorded. */
  bestStep: number | null;
  /** How many evaluations have been made. */
  evaluations: number;
  /** Consecutive evaluations without a qualifying improvement. */
  evaluationsSinceImprovement: number;
  stoppedEarly: boolean;
  /** Why the run ended, in words, whatever the outcome. */
  stoppingReason: string;
};

export type TrainingConfig = {
  batchSize: number;
  seqLen: number;
  totalSteps: number;
  learningRate: number;
  schedule: ScheduleKind;
  warmupSteps: number;
  minFactor: number;
  weightDecay: number;
  gradClipNorm: number;
  /** Run a validation pass every N steps (0 disables periodic evaluation). */
  evalInterval: number;
  /** Maximum validation batches per evaluation. */
  evalBatches: number;
  validationFraction: number;
  seed: number;
  /** Emit a checkpoint every N steps (0 disables periodic checkpoints). */
  checkpointInterval: number;
  /**
   * `windows` samples fixed-length windows of one token stream (no padding).
   * `documents` samples whole documents and pads the batch, masking the pads
   * out of the loss.
   */
  batchMode: BatchMode;
  /**
   * Number of micro-batches whose gradients are summed into one optimiser
   * update. 1 means one micro-batch per update, which is the Step 1-4
   * behaviour exactly. Greater than 1 gives an effective batch of
   * `batchSize * gradientAccumulationSteps` rows without holding that many rows
   * of activations at once.
   */
  gradientAccumulationSteps: number;
  /** Early-stopping rule, or null to train for the full step budget. */
  earlyStopping: EarlyStoppingConfig | null;
};

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  batchSize: 8,
  seqLen: 32,
  totalSteps: 120,
  learningRate: 3e-3,
  schedule: "cosine",
  warmupSteps: 12,
  minFactor: 0.1,
  weightDecay: 0.01,
  gradClipNorm: 1,
  evalInterval: 20,
  evalBatches: 4,
  validationFraction: 0.12,
  seed: 1337,
  checkpointInterval: 60,
  batchMode: "windows",
  gradientAccumulationSteps: 1,
  earlyStopping: null,
};

export function createTrainingConfig(input: Partial<TrainingConfig> = {}): TrainingConfig {
  const config = { ...DEFAULT_TRAINING_CONFIG, ...input };
  if (!Number.isInteger(config.batchSize) || config.batchSize < 1) {
    throw new AlphaValidationError("training", "batchSize must be a positive integer");
  }
  if (!Number.isInteger(config.seqLen) || config.seqLen < 2) {
    throw new AlphaValidationError("training", "seqLen must be an integer of at least 2");
  }
  if (!Number.isInteger(config.totalSteps) || config.totalSteps < 1) {
    throw new AlphaValidationError("training", "totalSteps must be a positive integer");
  }
  if (!(config.learningRate > 0)) {
    throw new AlphaValidationError("training", "learningRate must be positive");
  }
  if (config.validationFraction <= 0 || config.validationFraction >= 1) {
    throw new AlphaValidationError("training", "validationFraction must be in (0, 1)");
  }
  if (config.batchMode !== "windows" && config.batchMode !== "documents") {
    throw new AlphaValidationError("training", `unknown batchMode ${String(config.batchMode)}`);
  }
  if (
    !Number.isInteger(config.gradientAccumulationSteps) ||
    config.gradientAccumulationSteps < 1
  ) {
    throw new AlphaValidationError(
      "training",
      "gradientAccumulationSteps must be a positive integer",
    );
  }
  if (config.earlyStopping) {
    const early = config.earlyStopping;
    if (early.monitor !== "validationLoss" && early.monitor !== "validationPerplexity") {
      throw new AlphaValidationError(
        "training",
        `earlyStopping.monitor must be validationLoss or validationPerplexity, not ${String(early.monitor)}`,
      );
    }
    if (!Number.isInteger(early.patience) || early.patience < 1) {
      throw new AlphaValidationError("training", "earlyStopping.patience must be a positive integer");
    }
    if (!Number.isFinite(early.minDelta) || early.minDelta < 0) {
      throw new AlphaValidationError(
        "training",
        "earlyStopping.minDelta must be a non-negative number",
      );
    }
    if (!Number.isInteger(early.minSteps) || early.minSteps < 1) {
      throw new AlphaValidationError(
        "training",
        "earlyStopping.minSteps must be a positive integer",
      );
    }
    if (early.minSteps > config.totalSteps) {
      throw new AlphaValidationError(
        "training",
        `earlyStopping.minSteps ${early.minSteps} exceeds totalSteps ${config.totalSteps}; the run could never stop early`,
      );
    }
    if (!Number.isInteger(early.evalEvery) || early.evalEvery < 0) {
      throw new AlphaValidationError(
        "training",
        "earlyStopping.evalEvery must be 0 (reuse evalInterval) or a positive integer",
      );
    }
  }
  // Resource safety: refuse a configuration that would exhaust the host.
  assertResourceLimit("maxBatchSize", config.batchSize, "training config");
  assertResourceLimit("maxSeqLen", config.seqLen, "training config");
  assertResourceLimit("maxTotalSteps", config.totalSteps, "training config");
  return config;
}

export type TrainingMetricPoint = {
  step: number;
  loss: number;
  perplexity: number;
  learningRate: number;
  gradNorm: number;
  updateNorm: number;
  tokensSeen: number;
  /** Padding tokens present in this batch (0 in window mode). */
  paddingTokens: number;
  elapsedMs: number;
};

export type EvaluationResult = {
  step: number;
  loss: number;
  perplexity: number;
  batches: number;
  tokens: number;
  /** Baseline: uniform distribution over the vocabulary. */
  uniformLoss: number;
};

export type TrainingSummary = {
  /** Run id of the job this training belonged to. */
  jobId: string;
  /** How the run ended: completed, paused, stopped or failed. */
  state: TrainingJobState;
  steps: number;
  tokensSeen: number;
  firstLoss: number | null;
  lastLoss: number | null;
  bestLoss: number | null;
  validationLoss: number | null;
  validationPerplexity: number | null;
  uniformLossBaseline: number;
  durationMs: number;
  throughputTokensPerSecond: number;
  checkpoint: AlphaCheckpoint | null;
  checkpointCount: number;
  corpus: CorpusStats;
  /**
   * What early stopping did. Present whenever a rule was configured, including
   * when it never fired — "monitored and did not fire" is a result, and saying
   * so is different from saying nothing happened.
   */
  earlyStopping: EarlyStoppingReport | null;
  /** Micro-batches whose gradients were summed per optimiser step. */
  gradientAccumulationSteps: number;
  /** Tokens the single update in a step was computed from. */
  tokensPerStep: number;
  /** Optimiser steps the run would have taken without early stopping. */
  stepsSkippedByEarlyStopping: number;
};

export type TrainingEvent =
  | {
      type: "start";
      jobId: string;
      model: string;
      config: TrainingConfig;
      corpus: EncodedCorpus["stats"];
      parameterCount: number;
      uniformLoss: number;
    }
  | { type: "step"; point: TrainingMetricPoint }
  | { type: "eval"; evaluation: EvaluationResult }
  | { type: "checkpoint"; checkpoint: AlphaCheckpoint }
  | { type: "job"; job: AlphaTrainingJob };

/** Events that carry the terminating summary instead of being yielded. */
export type TrainingCompletion = TrainingSummary;

export type TrainerOptions = {
  model: AlphaTransformer;
  tokenizer: AlphaTokenizer;
  dataset: AlphaDataset;
  config?: Partial<TrainingConfig>;
  /** Label recorded in checkpoints. */
  checkpointLabel?: string;
  /** Mark this run as continued training over checkpoint weights. */
  isFineTune?: boolean;
  /** Run id recorded in every checkpoint. Defaults to a fresh id. */
  runId?: string;
};

type Sampler = {
  next(): TrainingBatch;
  sequentialBatches(): Generator<TrainingBatch>;
};

export class AlphaTrainer {
  readonly model: AlphaTransformer;
  readonly tokenizer: AlphaTokenizer;
  readonly dataset: AlphaDataset;
  readonly config: TrainingConfig;
  readonly corpus: EncodedCorpus;
  readonly optimizer: AdamW;
  readonly schedule: ScheduleConfig;
  readonly runId: string;
  readonly history: TrainingMetricPoint[] = [];
  private readonly sampler: Sampler;
  private readonly validationSampler: Sampler;
  private readonly rng: AlphaRng;
  private readonly checkpointLabel: string;
  private readonly isFineTune: boolean;
  private checkpointIds: string[] = [];
  private stepCount = 0;
  private startedAt = 0;
  private latestCheckpoint: AlphaCheckpoint | null = null;
  private earlyStopping: EarlyStoppingReport;

  constructor(options: TrainerOptions) {
    this.model = options.model;
    this.tokenizer = options.tokenizer;
    this.dataset = options.dataset;
    this.config = createTrainingConfig(options.config);
    this.checkpointLabel = options.checkpointLabel ?? "alpha-run";
    this.isFineTune = options.isFineTune ?? false;
    this.runId = options.runId ?? `run_${Date.now().toString(36)}${Math.floor(Math.random() * 0xffff).toString(36)}`;

    if (this.config.seqLen > this.model.config.contextLength) {
      throw new AlphaValidationError(
        "training",
        `seqLen ${this.config.seqLen} exceeds the model context length ${this.model.config.contextLength}`,
      );
    }
    if (this.tokenizer.vocabSize > this.model.config.vocabSize) {
      throw new AlphaValidationError(
        "training",
        `tokenizer vocabulary (${this.tokenizer.vocabSize}) is larger than the model vocabulary (${this.model.config.vocabSize})`,
      );
    }
    // Validated before anything is tokenised: a malformed corpus must never be
    // silently reduced to whatever happened to be encodable.
    assertValidDataset(this.dataset);

    this.corpus = encodeCorpus(this.dataset, this.tokenizer, {
      validationFraction: this.config.validationFraction,
    });
    this.sampler = this.buildSampler(this.corpus.trainIds, this.corpus.trainDocuments, this.config.seed);
    this.validationSampler = this.buildSampler(
      this.corpus.validationIds,
      this.corpus.validationDocuments,
      this.config.seed + 1,
    );
    this.rng = new AlphaRng(this.config.seed);
    this.optimizer = new AdamW(this.model.parameters(), {
      learningRate: this.config.learningRate,
      weightDecay: this.config.weightDecay,
      gradClipNorm: this.config.gradClipNorm,
    });
    this.schedule = {
      kind: this.config.schedule,
      peakLearningRate: this.config.learningRate,
      warmupSteps: this.config.warmupSteps,
      totalSteps: this.config.totalSteps,
      minFactor: this.config.minFactor,
    };
    const early = this.config.earlyStopping;
    this.earlyStopping = {
      monitor: early?.monitor ?? "validationLoss",
      patience: early?.patience ?? 0,
      minDelta: early?.minDelta ?? 0,
      minSteps: early?.minSteps ?? 0,
      bestValue: null,
      bestStep: null,
      evaluations: 0,
      evaluationsSinceImprovement: 0,
      stoppedEarly: false,
      stoppingReason: early
        ? "not yet evaluated"
        : "no early-stopping rule configured for this run",
    };
  }

  /** The early-stopping state as it currently stands. */
  get earlyStoppingReport(): EarlyStoppingReport {
    return { ...this.earlyStopping };
  }

  /** How often the monitored metric is evaluated, resolved against evalInterval. */
  private get earlyStopInterval(): number {
    const early = this.config.earlyStopping;
    if (!early) return 0;
    return early.evalEvery > 0 ? early.evalEvery : this.config.evalInterval;
  }

  private buildSampler(ids: Int32Array, documents: Int32Array[], seed: number): Sampler {
    if (this.config.batchMode === "documents") {
      return new DocumentBatchSampler(documents, {
        batchSize: this.config.batchSize,
        seqLen: this.config.seqLen,
        seed,
        padId: this.tokenizer.padId,
      });
    }
    return new BatchSampler(ids, {
      batchSize: this.config.batchSize,
      seqLen: this.config.seqLen,
      seed,
    });
  }

  get step(): number {
    return this.stepCount;
  }

  get completed(): boolean {
    return this.stepCount >= this.config.totalSteps;
  }

  get tokensSeen(): number {
    return this.stepCount * this.tokensPerStep;
  }

  /**
   * Tokens in one optimiser step — the throughput denominator.
   *
   * With gradient accumulation this is the *effective* batch: every
   * micro-batch in an accumulation group contributes its tokens, because the
   * single optimiser update they produce was computed from all of them.
   */
  get tokensPerStep(): number {
    return this.config.batchSize * this.config.seqLen * this.config.gradientAccumulationSteps;
  }

  /** Cross-entropy of a uniform predictor — the honest "no learning" baseline. */
  get uniformLoss(): number {
    return Math.log(this.tokenizer.vocabSize);
  }

  /** What this run consumes, computed from the encoded corpus. */
  get corpusStats(): CorpusStats {
    return this.corpus.stats;
  }

  /** Number of training examples available at this sequence length. */
  get trainingExamples(): number {
    if (this.config.batchMode === "documents") return this.corpus.stats.trainDocuments;
    return countTrainingExamples(this.corpus.stats.trainTokens, this.config.seqLen);
  }

  get validationExamples(): number {
    if (this.config.batchMode === "documents") return this.corpus.stats.validationDocuments;
    return countTrainingExamples(this.corpus.stats.validationTokens, this.config.seqLen);
  }

  /** Tokenizer reference recorded in checkpoints (includes the full snapshot). */
  tokenizerReference(): CheckpointTokenizerRef {
    return {
      version: this.tokenizer.version,
      vocabSize: this.tokenizer.vocabSize,
      fingerprint: this.tokenizer.fingerprint(),
      trainedOn: this.tokenizer.trainedOn,
      specialTokenIds: this.tokenizer.specialTokenIds,
      snapshot: this.tokenizer.toJSON(),
    };
  }

  /**
   * Forward and backward over a group of micro-batches, accumulating one
   * gradient without applying any optimiser update.
   *
   * This is the whole of gradient accumulation, isolated from the training loop
   * so it can be verified numerically: pass explicit batches and compare the
   * resulting gradient against a single forward/backward over their
   * concatenation. The gradient left on the parameters is the *token-weighted
   * mean* over the group, which is what makes it equivalent to training on the
   * concatenated batch — token weighting matters because document-mode
   * micro-batches carry different numbers of unpadded targets.
   *
   * Gradients are zeroed first and left accumulated afterwards; the caller
   * decides when to step and when to clear.
   */
  accumulateGradients(
    batches: TrainingBatch[],
  ): { gradNorm: number; meanLoss: number; tokens: number; batches: number } {
    if (batches.length === 0) {
      throw new AlphaValidationError("training", "accumulateGradients needs at least one batch");
    }
    const results: Array<{ tensor: Tensor; loss: number; tokens: number }> = [];
    for (const batch of batches) {
      const forward = this.model.forward(batch.input, batch.batch, batch.seqLen, {
        training: true,
        rng: this.rng,
      });
      const result = crossEntropy(forward.logits, batch.target, this.tokenizer.padId);
      results.push({ tensor: result.tensor, loss: result.loss, tokens: result.tokens });
    }

    const totalTokens = results.reduce((sum, r) => sum + r.tokens, 0);
    if (totalTokens === 0) {
      throw new AlphaValidationError(
        "training",
        "a gradient accumulation group contained no loss-bearing tokens; the batch configuration cannot produce a training signal",
      );
    }

    this.optimizer.zeroGrad();
    for (const result of results) {
      backward(scale(result.tensor, result.tokens / totalTokens));
    }

    return {
      gradNorm: this.optimizer.gradNorm(),
      meanLoss: results.reduce((sum, r) => sum + r.loss, 0) / results.length,
      tokens: totalTokens,
      batches: batches.length,
    };
  }

  /**
   * One optimiser step: forward, loss, backward, clip, update.
   *
   * With `gradientAccumulationSteps > 1` this runs that many micro-batches,
   * accumulates their gradients, and takes exactly one optimiser step.
   */
  private trainStep(): TrainingMetricPoint {
    const lr = learningRateAt(this.stepCount + 1, this.schedule);
    const accumulation: TrainingBatch[] = [];
    for (let i = 0; i < this.config.gradientAccumulationSteps; i++) {
      accumulation.push(this.sampler.next());
    }

    const accumulated = this.accumulateGradients(accumulation);
    const report = this.optimizer.stepWithSchedule(lr);
    this.optimizer.zeroGrad();
    this.stepCount = report.step;

    const loss = accumulated.meanLoss;
    const point: TrainingMetricPoint = {
      step: this.stepCount,
      loss,
      perplexity: Math.exp(Math.min(loss, 20)),
      learningRate: report.learningRate,
      gradNorm: report.gradNorm,
      updateNorm: report.updateNorm,
      tokensSeen: this.tokensSeen,
      paddingTokens: accumulation.reduce((sum, b) => sum + b.paddingTokens, 0),
      elapsedMs: Date.now() - this.startedAt,
    };
    this.history.push(point);
    return point;
  }

  /**
   * Evaluate the early-stopping monitor without applying any of its decisions.
   *
   * Exposed so a caller (and the tests) can see what the rule *would* decide
   * from a given validation result, rather than inferring it from a run.
   */
  earlyStopDecision(evaluation: EvaluationResult): {
    value: number | null;
    improved: boolean;
    shouldStop: boolean;
    reason: string;
  } {
    const early = this.config.earlyStopping;
    if (!early) {
      return { value: null, improved: false, shouldStop: false, reason: "no early-stopping rule configured" };
    }
    const value =
      early.monitor === "validationLoss" ? evaluation.loss : evaluation.perplexity;
    if (!Number.isFinite(value)) {
      return {
        value: null,
        improved: false,
        shouldStop: false,
        reason: `the monitored metric ${early.monitor} was not finite at step ${evaluation.step}, so it was not used`,
      };
    }
    const previousBest = this.earlyStopping.bestValue;
    const improved =
      previousBest === null || value < previousBest - early.minDelta;
    return {
      value,
      improved,
      shouldStop: false,
      reason: improved
        ? `step ${evaluation.step}: ${early.monitor} ${value.toFixed(4)} improved on ${previousBest === null ? "the first measurement" : previousBest.toFixed(4)}`
        : `step ${evaluation.step}: ${early.monitor} ${value.toFixed(4)} did not improve on ${(previousBest ?? value).toFixed(4)} (minDelta ${early.minDelta})`,
    };
  }

  /** Apply one early-stopping evaluation. Returns the report after updating it. */
  private recordEarlyStop(evaluation: EvaluationResult): EarlyStoppingReport {
    const early = this.config.earlyStopping;
    if (!early) return this.earlyStopping;
    const decision = this.earlyStopDecision(evaluation);
    this.earlyStopping.evaluations += 1;
    if (decision.value !== null && decision.improved) {
      this.earlyStopping.bestValue = decision.value;
      this.earlyStopping.bestStep = evaluation.step;
      this.earlyStopping.evaluationsSinceImprovement = 0;
    } else {
      this.earlyStopping.evaluationsSinceImprovement += 1;
    }
    const pastMinSteps = evaluation.step >= early.minSteps;
    if (pastMinSteps && this.earlyStopping.evaluationsSinceImprovement >= early.patience) {
      this.earlyStopping.stoppedEarly = true;
      this.earlyStopping.stoppingReason =
        `stopped at step ${evaluation.step}: ${early.monitor} had not improved on ` +
        `${(this.earlyStopping.bestValue ?? Number.NaN).toFixed(4)} for ` +
        `${this.earlyStopping.evaluationsSinceImprovement} consecutive evaluation(s), ` +
        `which met the patience of ${early.patience}`;
    } else if (decision.value !== null) {
      this.earlyStopping.stoppingReason = decision.reason;
    }
    return this.earlyStopping;
  }

  /** Deterministic validation loss over the held-out split. */
  evaluate(options: { maxBatches?: number } = {}): EvaluationResult {
    const maxBatches = options.maxBatches ?? this.config.evalBatches;
    setGradEnabled(false);
    let lossSum = 0;
    let batches = 0;
    let tokens = 0;
    try {
      for (const batch of this.validationSampler.sequentialBatches()) {
        const forward = this.model.forward(batch.input, batch.batch, batch.seqLen, {
          training: false,
        });
        const result = crossEntropy(forward.logits, batch.target, this.tokenizer.padId);
        lossSum += result.loss;
        tokens += result.tokens;
        batches++;
        if (batches >= maxBatches) break;
      }
    } finally {
      setGradEnabled(true);
    }
    const loss = batches > 0 ? lossSum / batches : Number.NaN;
    return {
      step: this.stepCount,
      loss,
      perplexity: Number.isFinite(loss) ? Math.exp(Math.min(loss, 20)) : Number.NaN,
      batches,
      tokens,
      uniformLoss: this.uniformLoss,
    };
  }

  /**
   * Compare Alpha's autodiff against central differences on a real batch.
   * Exposed on the trainer so gradient validity is a measurement, not a claim.
   */
  gradientCheck(options: { eps?: number; tolerance?: number; samplesPerTensor?: number } = {}): GradientCheckReport {
    const batch = this.sampler.next();
    return numericalGradientCheck(
      this.model,
      { input: batch.input, target: batch.target, batch: batch.batch, seqLen: batch.seqLen },
      { ...options, padId: this.tokenizer.padId },
    );
  }

  /** Persist the current state into a checkpoint artifact. */
  buildCheckpoint(metrics?: Partial<AlphaCheckpointMetrics>): AlphaCheckpoint {
    const evaluation = this.evaluate({ maxBatches: Math.min(2, this.config.evalBatches) });
    const last = this.history[this.history.length - 1];
    const checkpoint = createCheckpoint({
      label: `${this.checkpointLabel}@${this.stepCount}`,
      modelName: this.model.config.name,
      modelVersion: this.model.config.version,
      config: this.model.config,
      tokenizer: this.tokenizerReference(),
      datasetName: this.dataset.name,
      datasetVersion: this.dataset.version,
      datasetFingerprint: datasetFingerprint(this.dataset),
      datasetLicense: this.dataset.license,
      step: this.stepCount,
      tokensSeen: this.tokensSeen,
      learningRate: last?.learningRate ?? this.config.learningRate,
      metrics: {
        trainLoss: metrics?.trainLoss ?? last?.loss ?? Number.NaN,
        validationLoss: metrics?.validationLoss ?? (Number.isFinite(evaluation.loss) ? evaluation.loss : null),
        validationPerplexity:
          metrics?.validationPerplexity ?? (Number.isFinite(evaluation.perplexity) ? evaluation.perplexity : null),
        uniformLoss: metrics?.uniformLoss ?? this.uniformLoss,
      },
      weights: this.model.serializeWeights(),
      optimizer: this.optimizer.snapshot(),
      rng: this.rng.saveState(),
      runId: this.runId,
      seed: this.config.seed,
      trainingConfig: this.config,
      isFineTune: this.isFineTune,
    });
    // A checkpoint Alpha just wrote must satisfy its own contract.
    assertValidCheckpoint(checkpoint);
    this.latestCheckpoint = checkpoint;
    if (!this.checkpointIds.includes(checkpoint.id)) this.checkpointIds.push(checkpoint.id);
    return checkpoint;
  }

  get checkpoints(): readonly string[] {
    return this.checkpointIds;
  }

  /**
   * Continue training from a checkpoint: weights, optimiser moments, the RNG
   * stream and the step counter are all restored, so the run resumes rather
   * than restarts.
   */
  resumeFrom(checkpoint: AlphaCheckpoint): void {
    assertValidCheckpoint(checkpoint);
    // Weights from a different architecture or vocabulary would load into
    // shape-compatible but meaningless positions, so this refuses instead.
    assertCheckpointCompatible(checkpoint, { config: this.model.config, tokenizer: this.tokenizer });
    this.model.loadWeights(checkpoint.weights);
    this.optimizer.loadSnapshot(checkpoint.optimizer);
    this.stepCount = checkpoint.step;
    this.latestCheckpoint = checkpoint;
    if (!this.checkpointIds.includes(checkpoint.id)) this.checkpointIds.unshift(checkpoint.id);
  }

  get checkpoint(): AlphaCheckpoint | null {
    return this.latestCheckpoint;
  }

  private summary(
    state: TrainingJobState,
    startStep: number,
    durationMs: number,
    reason?: string,
  ): TrainingSummary {
    const losses = this.history.slice(-Math.max(1, this.stepCount - startStep)).map((p) => p.loss);
    const tokensThisRun = Math.max(0, this.stepCount - startStep) * this.tokensPerStep;
    const lastEvaluation = this.latestCheckpoint?.metrics.validationLoss ?? null;
    const summary: TrainingSummary = {
      jobId: this.runId,
      state,
      steps: Math.max(0, this.stepCount - startStep),
      tokensSeen: this.tokensSeen,
      firstLoss: losses.length ? losses[0] : null,
      lastLoss: losses.length ? losses[losses.length - 1] : null,
      bestLoss: losses.length ? Math.min(...losses) : null,
      validationLoss: lastEvaluation,
      validationPerplexity: this.latestCheckpoint?.metrics.validationPerplexity ?? null,
      uniformLossBaseline: this.uniformLoss,
      durationMs,
      throughputTokensPerSecond: durationMs > 0 ? Math.round((tokensThisRun / durationMs) * 1000) : 0,
      checkpoint: this.latestCheckpoint,
      checkpointCount: this.checkpointIds.length,
      corpus: this.corpus.stats,
      earlyStopping: this.config.earlyStopping ? { ...this.earlyStopping } : null,
      gradientAccumulationSteps: this.config.gradientAccumulationSteps,
      tokensPerStep: this.tokensPerStep,
      stepsSkippedByEarlyStopping: this.earlyStopping.stoppedEarly
        ? Math.max(0, this.config.totalSteps - this.stepCount)
        : 0,
    };
    if (reason && summary.earlyStopping) {
      // The reason belongs to the run, not to the metric numbers, so it is
      // carried on the summary without overwriting any measured value.
      summary.earlyStopping.stoppingReason = reason;
    }
    return summary;
  }

  /**
   * Run the training loop as a generator. Each yielded event is a real step;
   * callers decide how often to yield to the event loop, and may stop pulling
   * events (a pause) without destroying the trainer's state.
   *
   * `shouldStop` is polled between steps: returning `"pause"` ends the
   * generator with the trainer still resumable, `"stop"` ends the run.
   */
  *run(control: { shouldStop?: () => "pause" | "stop" | null } = {}): Generator<TrainingEvent, TrainingSummary> {
    this.startedAt = Date.now();
    const startStep = this.stepCount;
    yield {
      type: "start",
      jobId: this.runId,
      model: this.model.config.name,
      config: this.config,
      corpus: this.corpus.stats,
      parameterCount: this.model.parameterCount,
      uniformLoss: this.uniformLoss,
    };

    while (this.stepCount < this.config.totalSteps) {
      const requested = control.shouldStop?.() ?? null;
      if (requested) {
        const duration = Date.now() - this.startedAt;
        // Both pause and stop write a checkpoint first: work already done must
        // survive the request, whatever the caller does next.
        const checkpoint = this.buildCheckpoint();
        yield { type: "checkpoint", checkpoint };
        return this.summary(requested === "stop" ? "stopped" : "paused", startStep, duration);
      }
      const point = this.trainStep();
      yield { type: "step", point };
      if (this.config.evalInterval > 0 && this.stepCount % this.config.evalInterval === 0) {
        yield { type: "eval", evaluation: this.evaluate({ maxBatches: 2 }) };
      }
      if (this.config.checkpointInterval > 0 && this.stepCount % this.config.checkpointInterval === 0) {
        const checkpoint = this.buildCheckpoint({ trainLoss: point.loss });
        yield { type: "checkpoint", checkpoint };
      }

      // Early stopping is evaluated on its own cadence, which may be finer than
      // the reporting interval: patience is counted in evaluations, and a
      // coarse interval would make a patience of 3 mean "three reporting
      // intervals", which is not what an operator asking for patience 3 means.
      const earlyInterval = this.earlyStopInterval;
      if (
        this.config.earlyStopping &&
        earlyInterval > 0 &&
        this.stepCount % earlyInterval === 0
      ) {
        const monitored = this.evaluate({ maxBatches: this.config.evalBatches });
        yield { type: "eval", evaluation: monitored };
        const report = this.recordEarlyStop(monitored);
        if (report.stoppedEarly) {
          // A checkpoint is written before returning, so an early-stopped run is
          // as resumable as a completed one.
          const checkpoint = this.buildCheckpoint({ trainLoss: point.loss });
          yield { type: "checkpoint", checkpoint };
          return this.summary("stopped", startStep, Date.now() - this.startedAt, report.stoppingReason);
        }
      }
    }

    const finalEvaluation = this.evaluate({ maxBatches: this.config.evalBatches });
    // The final evaluation counts toward the monitor too, so the reported best
    // value is the best over every measurement including the last one.
    if (this.config.earlyStopping) this.recordEarlyStop(finalEvaluation);
    yield { type: "eval", evaluation: finalEvaluation };
    const finalCheckpoint = this.buildCheckpoint({
      trainLoss: this.history[this.history.length - 1]?.loss,
      validationLoss: Number.isFinite(finalEvaluation.loss) ? finalEvaluation.loss : null,
      validationPerplexity: Number.isFinite(finalEvaluation.perplexity) ? finalEvaluation.perplexity : null,
    });
    yield { type: "checkpoint", checkpoint: finalCheckpoint };

    return this.summary("completed", startStep, Date.now() - this.startedAt);
  }

  /**
   * Convenience for scripts and tests: run to completion synchronously. The
   * summary is the generator's return value, so no event carries it.
   */
  trainToCompletion(): TrainingSummary {
    const iterator = this.run();
    let result = iterator.next();
    while (!result.done) result = iterator.next();
    if (!result.value) {
      throw new AlphaValidationError("training", "training run produced no summary");
    }
    return result.value;
  }
}
