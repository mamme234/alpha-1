/**
 * Alpha Training Engine — checkpoints.
 *
 * A checkpoint is the unit of "Alpha has actually trained". It carries the
 * weights, the optimiser moments, the RNG position, the step counter and the
 * metrics measured up to that point, which is what makes training resumable
 * instead of restartable.
 */

import { alphaId, type AlphaModelStage } from "../core/types";
import type { RngState } from "../core/rng";
import { base64ByteLength } from "../core/serialize";
import type { AlphaModelConfig } from "../model/config";
import type { SerializedWeights } from "../model/transformer";
import type { OptimizerStateSnapshot } from "./optimizer";

export type AlphaCheckpointMetrics = {
  trainLoss: number;
  validationLoss: number | null;
  validationPerplexity: number | null;
};

export type AlphaCheckpoint = {
  id: string;
  label: string;
  modelName: string;
  modelVersion: string;
  config: AlphaModelConfig;
  tokenizerVersion: string;
  datasetName: string;
  datasetLicense: string;
  /** Optimiser steps completed when this checkpoint was written. */
  step: number;
  tokensSeen: number;
  learningRate: number;
  metrics: AlphaCheckpointMetrics;
  /** Base64 float32 payload per parameter name. */
  weights: SerializedWeights;
  optimizer: OptimizerStateSnapshot;
  rng: RngState;
  createdAt: number;
  sizeBytes: number;
  /** Stage implied purely by the existence of this checkpoint. */
  stage: AlphaModelStage;
  notes: string[];
};

export type CreateCheckpointInput = {
  label: string;
  modelName: string;
  modelVersion: string;
  config: AlphaModelConfig;
  tokenizerVersion: string;
  datasetName: string;
  datasetLicense: string;
  step: number;
  tokensSeen: number;
  learningRate: number;
  metrics: AlphaCheckpointMetrics;
  weights: SerializedWeights;
  optimizer: OptimizerStateSnapshot;
  rng: RngState;
  /** True when the run continued from an earlier checkpoint of a trained model. */
  isFineTune?: boolean;
  id?: string;
};

/** Rough payload size in bytes, computed from the encoded weights. */
export function estimateCheckpointBytes(weights: SerializedWeights): number {
  let bytes = 0;
  for (const encoded of Object.values(weights.tensors)) bytes += base64ByteLength(encoded);
  return bytes;
}

export function createCheckpoint(input: CreateCheckpointInput): AlphaCheckpoint {
  const sizeBytes = estimateCheckpointBytes(input.weights);
  return {
    id: input.id ?? alphaId("ckpt"),
    label: input.label,
    modelName: input.modelName,
    modelVersion: input.modelVersion,
    config: input.config,
    tokenizerVersion: input.tokenizerVersion,
    datasetName: input.datasetName,
    datasetLicense: input.datasetLicense,
    step: input.step,
    tokensSeen: input.tokensSeen,
    learningRate: input.learningRate,
    metrics: input.metrics,
    weights: input.weights,
    optimizer: input.optimizer,
    rng: input.rng,
    createdAt: Date.now(),
    sizeBytes,
    stage: input.step > 0 ? (input.isFineTune ? "fine-tuned" : "trained") : "untrained",
    notes: [
      `Trained from scratch by the Alpha training engine in ${input.step} optimiser steps.`,
      `Tokens seen: ${input.tokensSeen.toLocaleString()}.`,
      `Corpus: ${input.datasetName} (${input.datasetLicense}).`,
      "No external model weights were used at any point.",
    ],
  };
}

export function checkpointToJson(checkpoint: AlphaCheckpoint): string {
  return JSON.stringify(checkpoint);
}

export function parseCheckpoint(json: string): AlphaCheckpoint {
  return JSON.parse(json) as AlphaCheckpoint;
}

/** Compact summary for lists and the workspace UI (drops the weight payload). */
export type AlphaCheckpointSummary = Omit<AlphaCheckpoint, "weights" | "optimizer">;

export function summariseCheckpoint(checkpoint: AlphaCheckpoint): AlphaCheckpointSummary {
  const { weights: _weights, optimizer: _optimizer, ...rest } = checkpoint;
  return rest;
}
