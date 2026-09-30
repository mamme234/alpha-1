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
import { backward, crossEntropy, setGradEnabled } from "../core/tensor";
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
    return this.stepCount * this.config.batchSize * this.config.seqLen;
  }

  /** Tokens in one optimiser step — the throughput denominator. */
  get tokensPerStep(): number {
    return this.config.batchSize * this.config.seqLen;
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

  /** One optimiser step: forward, loss, backward, clip, update. */
  private trainStep(): TrainingMetricPoint {
    const batch = this.sampler.next();
    const lr = learningRateAt(this.stepCount + 1, this.schedule);
    const forward = this.model.forward(batch.input, batch.batch, batch.seqLen, {
      training: true,
      rng: this.rng,
    });
    const result = crossEntropy(forward.logits, batch.target, this.tokenizer.padId);
    backward(result.tensor);
    const report = this.optimizer.stepWithSchedule(lr);
    this.optimizer.zeroGrad();
    this.stepCount = report.step;
    const point: TrainingMetricPoint = {
      step: this.stepCount,
      loss: result.loss,
      perplexity: result.perplexity,
      learningRate: report.learningRate,
      gradNorm: report.gradNorm,
      updateNorm: report.updateNorm,
      tokensSeen: this.tokensSeen,
      paddingTokens: batch.paddingTokens,
      elapsedMs: Date.now() - this.startedAt,
    };
    this.history.push(point);
    return point;
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

  private summary(state: TrainingJobState, startStep: number, durationMs: number): TrainingSummary {
    const losses = this.history.slice(-Math.max(1, this.stepCount - startStep)).map((p) => p.loss);
    const tokensThisRun = Math.max(0, this.stepCount - startStep) * this.tokensPerStep;
    const lastEvaluation = this.latestCheckpoint?.metrics.validationLoss ?? null;
    return {
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
    };
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
    }

    const finalEvaluation = this.evaluate({ maxBatches: this.config.evalBatches });
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
