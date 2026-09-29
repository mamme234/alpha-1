/**
 * Alpha Training Engine — optimiser.
 *
 * AdamW with decoupled weight decay (Loshchilov & Hutter) plus global-norm
 * gradient clipping. Optimiser moments are serialisable so a checkpoint can
 * resume a run without resetting Adam's running estimates.
 */

import { AlphaValidationError } from "../core/errors";
import { Tensor, gradL2Norm } from "../core/tensor";

export type AdamWConfig = {
  learningRate: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  /** Clip the global gradient norm to this value (0 disables clipping). */
  gradClipNorm: number;
};

export const DEFAULT_ADAMW: AdamWConfig = {
  learningRate: 3e-3,
  beta1: 0.9,
  beta2: 0.95,
  eps: 1e-8,
  weightDecay: 0.01,
  gradClipNorm: 1.0,
};

export type OptimizerStateSnapshot = {
  step: number;
  /** parameter name -> base64 float32 moments */
  firstMoment: Record<string, string>;
  secondMoment: Record<string, string>;
};

export type ParameterHandle = { name: string; tensor: Tensor };

export type OptimizerStepReport = {
  step: number;
  learningRate: number;
  gradNorm: number;
  clippedGradNorm: number;
  /** Parameter update magnitude — useful to detect exploding/vanishing steps. */
  updateNorm: number;
};

export class AdamW {
  private readonly parameters: ParameterHandle[];
  private readonly config: AdamWConfig;
  private readonly first: Float32Array[];
  private readonly second: Float32Array[];
  private stepCount = 0;

  constructor(parameters: ParameterHandle[], config: Partial<AdamWConfig> = {}) {
    this.parameters = parameters;
    this.config = { ...DEFAULT_ADAMW, ...config };
    if (this.config.learningRate <= 0) {
      throw new AlphaValidationError("training", "learningRate must be positive");
    }
    this.first = this.parameters.map((p) => new Float32Array(p.tensor.size));
    this.second = this.parameters.map((p) => new Float32Array(p.tensor.size));
  }

  get step(): number {
    return this.stepCount;
  }

  get learningRate(): number {
    return this.config.learningRate;
  }

  setLearningRate(value: number): void {
    this.config.learningRate = value;
  }

  /** Zero every gradient in the parameter set. */
  zeroGrad(): void {
    for (const { tensor } of this.parameters) {
      if (tensor.grad) tensor.grad.fill(0);
    }
  }

  /** Global L2 norm of the current gradients, before clipping. */
  gradNorm(): number {
    return gradL2Norm(this.parameters.map((p) => p.tensor));
  }

  /**
   * Apply one update. `learningRateOverride` lets the training engine drive a
   * schedule without mutating the optimiser config.
   */
  stepWithSchedule(learningRateOverride?: number): OptimizerStepReport {
    const lr = learningRateOverride ?? this.config.learningRate;
    const gradNorm = this.gradNorm();
    let clipScale = 1;
    if (this.config.gradClipNorm > 0 && gradNorm > this.config.gradClipNorm) {
      clipScale = this.config.gradClipNorm / (gradNorm + 1e-12);
    }
    const { beta1, beta2, eps, weightDecay } = this.config;
    const biasCorrection1 = 1 - beta1 ** (this.stepCount + 1);
    const biasCorrection2 = 1 - beta2 ** (this.stepCount + 1);
    let updateSquares = 0;

    for (let i = 0; i < this.parameters.length; i++) {
      const tensor = this.parameters[i].tensor;
      const grad = tensor.grad;
      if (!grad) continue;
      const m = this.first[i];
      const v = this.second[i];
      for (let j = 0; j < grad.length; j++) {
        const g = grad[j] * clipScale;
        m[j] = beta1 * m[j] + (1 - beta1) * g;
        v[j] = beta2 * v[j] + (1 - beta2) * g * g;
        const mHat = m[j] / biasCorrection1;
        const vHat = v[j] / biasCorrection2;
        // Decoupled weight decay: applied to the weight, not through the gradient.
        const decay = weightDecay > 0 ? weightDecay * tensor.data[j] : 0;
        const delta = lr * (mHat / (Math.sqrt(vHat) + eps) + decay);
        tensor.data[j] -= delta;
        updateSquares += delta * delta;
      }
    }
    this.stepCount++;
    return {
      step: this.stepCount,
      learningRate: lr,
      gradNorm,
      clippedGradNorm: gradNorm * clipScale,
      updateNorm: Math.sqrt(updateSquares),
    };
  }

  /** Weight decay applied to non-normalisation, non-bias parameters only. */
  decayableParameters(): ParameterHandle[] {
    return this.parameters.filter(
      (p) => !p.name.includes("bias") && !p.name.includes("norm") && p.name.endsWith("weight"),
    );
  }

  snapshot(): OptimizerStateSnapshot {
    const firstMoment: Record<string, string> = {};
    const secondMoment: Record<string, string> = {};
    const encode = (values: Float32Array) => {
      const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
      let binary = "";
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
      const g = globalThis as { btoa?: (s: string) => string };
      return typeof g.btoa === "function"
        ? g.btoa(binary)
        : Buffer.from(binary, "binary").toString("base64");
    };
    this.parameters.forEach((p, index) => {
      firstMoment[p.name] = encode(this.first[index]);
      secondMoment[p.name] = encode(this.second[index]);
    });
    return { step: this.stepCount, firstMoment, secondMoment };
  }

  loadSnapshot(snapshot: OptimizerStateSnapshot): void {
    const decode = (encoded: string) => {
      const g = globalThis as { atob?: (s: string) => string };
      const binary =
        typeof g.atob === "function" ? g.atob(encoded) : Buffer.from(encoded, "base64").toString("binary");
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return new Float32Array(bytes.buffer, 0, bytes.length / 4);
    };
    this.parameters.forEach((p, index) => {
      const m = snapshot.firstMoment[p.name];
      const v = snapshot.secondMoment[p.name];
      if (m) {
        const values = decode(m);
        if (values.length === this.first[index].length) this.first[index].set(values);
      }
      if (v) {
        const values = decode(v);
        if (values.length === this.second[index].length) this.second[index].set(values);
      }
    });
    this.stepCount = snapshot.step;
  }
}
