/**
 * Alpha Training Engine — the trainer.
 *
 * Responsibilities:
 *   dataset -> tokenised corpus -> batches -> forward pass -> cross-entropy
 *   -> backpropagation through Alpha's own autodiff -> AdamW update -> metrics
 *
 * It exposes training as a *generator*: each `next()` performs one optimiser
 * step and yields the metrics. That lets a browser UI run real training in
 * slices without blocking the main thread, and lets a Node script simply loop
 * to completion. No GPU, no external runtime, no hidden model download.
 */

import { AlphaRng } from "../core/rng";
import { AlphaValidationError } from "../core/errors";
import { backward, crossEntropy, setGradEnabled } from "../core/tensor";
import type { AlphaTokenizer } from "../tokenizer/bpe";
import type { AlphaDataset } from "../datasets/types";
import { BatchSampler, encodeCorpus, type EncodedCorpus } from "../datasets/corpus";
import type { AlphaTransformer } from "../model/transformer";
import { AdamW } from "./optimizer";
import { learningRateAt, type ScheduleConfig, type ScheduleKind } from "./schedule";
import {
  type AlphaCheckpoint,
  type AlphaCheckpointMetrics,
  createCheckpoint,
} from "./checkpoint";

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
};

export function createTrainingConfig(input: Partial<TrainingConfig> = {}): TrainingConfig {
  return { ...DEFAULT_TRAINING_CONFIG, ...input };
}

export type TrainingMetricPoint = {
  step: number;
  loss: number;
  perplexity: number;
  learningRate: number;
  gradNorm: number;
  updateNorm: number;
  tokensSeen: number;
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
};

export type TrainingEvent =
  | {
      type: "start";
      model: string;
      config: TrainingConfig;
      corpus: EncodedCorpus["stats"];
      parameterCount: number;
      uniformLoss: number;
    }
  | { type: "step"; point: TrainingMetricPoint }
  | { type: "eval"; evaluation: EvaluationResult }
  | { type: "checkpoint"; checkpoint: AlphaCheckpoint }
  | { type: "done"; summary: TrainingSummary };

export type TrainerOptions = {
  model: AlphaTransformer;
  tokenizer: AlphaTokenizer;
  dataset: AlphaDataset;
  config?: Partial<TrainingConfig>;
  /** Label recorded in checkpoints. */
  checkpointLabel?: string;
  /** Mark this run as continued training over checkpoint weights. */
  isFineTune?: boolean;
};

export class AlphaTrainer {
  readonly model: AlphaTransformer;
  readonly tokenizer: AlphaTokenizer;
  readonly dataset: AlphaDataset;
  readonly config: TrainingConfig;
  readonly corpus: EncodedCorpus;
  readonly optimizer: AdamW;
  readonly schedule: ScheduleConfig;
  readonly history: TrainingMetricPoint[] = [];
  private readonly sampler: BatchSampler;
  private readonly validationSampler: BatchSampler;
  private readonly rng: AlphaRng;
  private readonly checkpointLabel: string;
  private readonly isFineTune: boolean;
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

    this.corpus = encodeCorpus(this.dataset, this.tokenizer, {
      validationFraction: this.config.validationFraction,
    });
    this.sampler = new BatchSampler(this.corpus.trainIds, {
      batchSize: this.config.batchSize,
      seqLen: this.config.seqLen,
      seed: this.config.seed,
    });
    this.validationSampler = new BatchSampler(this.corpus.validationIds, {
      batchSize: this.config.batchSize,
      seqLen: this.config.seqLen,
      seed: this.config.seed + 1,
    });
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

  get step(): number {
    return this.stepCount;
  }

  get completed(): boolean {
    return this.stepCount >= this.config.totalSteps;
  }

  get tokensSeen(): number {
    return this.stepCount * this.config.batchSize * this.config.seqLen;
  }

  /** Cross-entropy of a uniform predictor — the honest "no learning" baseline. */
  get uniformLoss(): number {
    return Math.log(this.tokenizer.vocabSize);
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
      perplexity: Math.exp(Math.min(loss, 20)),
      batches,
      tokens,
      uniformLoss: this.uniformLoss,
    };
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
      tokenizerVersion: this.tokenizer.version,
      datasetName: this.dataset.name,
      datasetLicense: this.dataset.license,
      step: this.stepCount,
      tokensSeen: this.tokensSeen,
      learningRate: last?.learningRate ?? this.config.learningRate,
      metrics: {
        trainLoss: metrics?.trainLoss ?? last?.loss ?? Number.NaN,
        validationLoss: metrics?.validationLoss ?? evaluation.loss,
        validationPerplexity: metrics?.validationPerplexity ?? evaluation.perplexity,
      },
      weights: this.model.serializeWeights(),
      optimizer: this.optimizer.snapshot(),
      rng: this.rng.saveState(),
      isFineTune: this.isFineTune,
    });
    this.latestCheckpoint = checkpoint;
    return checkpoint;
  }

  /**
   * Continue training from a checkpoint: weights, optimiser moments and the
   * RNG stream are all restored, so the run resumes rather than restarts.
   */
  resumeFrom(checkpoint: AlphaCheckpoint): void {
    this.model.loadWeights(checkpoint.weights);
    this.optimizer.loadSnapshot(checkpoint.optimizer);
    this.stepCount = checkpoint.step;
    this.latestCheckpoint = checkpoint;
  }

  get checkpoint(): AlphaCheckpoint | null {
    return this.latestCheckpoint;
  }

  /**
   * Run the training loop as a generator. Each yielded event is a real step;
   * callers decide how often to yield to the event loop.
   */
  *run(): Generator<TrainingEvent, TrainingSummary> {
    this.startedAt = Date.now();
    const startStep = this.stepCount;
    yield {
      type: "start",
      model: this.model.config.name,
      config: this.config,
      corpus: this.corpus.stats,
      parameterCount: this.model.parameterCount,
      uniformLoss: this.uniformLoss,
    };

    while (this.stepCount < this.config.totalSteps) {
      const point = this.trainStep();
      yield { type: "step", point };
      if (
        this.config.evalInterval > 0 &&
        this.stepCount % this.config.evalInterval === 0
      ) {
        yield { type: "eval", evaluation: this.evaluate({ maxBatches: 2 }) };
      }
      if (
        this.config.checkpointInterval > 0 &&
        this.stepCount % this.config.checkpointInterval === 0
      ) {
        const checkpoint = this.buildCheckpoint({ trainLoss: point.loss });
        yield { type: "checkpoint", checkpoint };
      }
    }

    const finalEvaluation = this.evaluate({ maxBatches: this.config.evalBatches });
    yield { type: "eval", evaluation: finalEvaluation };
    const finalCheckpoint = this.buildCheckpoint({
      trainLoss: this.history[this.history.length - 1]?.loss,
      validationLoss: finalEvaluation.loss,
      validationPerplexity: finalEvaluation.perplexity,
    });
    yield { type: "checkpoint", checkpoint: finalCheckpoint };

    const losses = this.history.slice(startStep).map((p) => p.loss);
    const durationMs = Date.now() - this.startedAt;
    const tokensThisRun = (this.stepCount - startStep) * this.config.batchSize * this.config.seqLen;
    const summary: TrainingSummary = {
      steps: this.stepCount - startStep,
      tokensSeen: this.tokensSeen,
      firstLoss: losses.length ? losses[0] : null,
      lastLoss: losses.length ? losses[losses.length - 1] : null,
      bestLoss: losses.length ? Math.min(...losses) : null,
      validationLoss: finalEvaluation.loss,
      validationPerplexity: finalEvaluation.perplexity,
      uniformLossBaseline: this.uniformLoss,
      durationMs,
      throughputTokensPerSecond:
        durationMs > 0 ? Math.round((tokensThisRun / durationMs) * 1000) : 0,
      checkpoint: finalCheckpoint,
    };
    yield { type: "done", summary };
    return summary;
  }

  /** Convenience for scripts and tests: run to completion synchronously. */
  trainToCompletion(): TrainingSummary {
    const iterator = this.run();
    let result = iterator.next();
    let summary: TrainingSummary | null = null;
    while (!result.done) {
      if (result.value.type === "done") summary = result.value.summary;
      result = iterator.next();
    }
    if (!summary) {
      throw new AlphaValidationError("training", "training run produced no summary");
    }
    return summary;
  }
}
