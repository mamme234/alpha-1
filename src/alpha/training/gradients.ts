/**
 * Alpha Training Engine — gradient validation.
 *
 * The autodiff engine claims analytic gradients for every operation. This
 * module checks that claim rather than asserting it: it compares the gradient
 * produced by backpropagation against a central finite difference computed from
 * the loss itself.
 *
 * It is used by the verification suite (check C) and exposed on the trainer, so
 * "gradients are correct" is a measurement rather than a comment.
 */

import { backward, crossEntropy, setGradEnabled } from "../core/tensor";
import type { AlphaTransformer } from "../model/transformer";

export type GradientCheckBatch = {
  input: Int32Array;
  target: Int32Array;
  batch: number;
  seqLen: number;
};

export type GradientCheckReport = {
  checkedParameters: number;
  checkedValues: number;
  maxAbsoluteError: number;
  maxRelativeError: number;
  tolerance: number;
  absoluteTolerance: number;
  eps: number;
  passed: boolean;
  detail: string;
};

export type GradientCheckOptions = {
  /** Central-difference step. */
  eps?: number;
  /** Maximum allowed relative error per value. */
  tolerance?: number;
  /**
   * Absolute error that passes regardless of the relative one. Near-zero
   * gradients are dominated by float32 rounding, so judging them by a ratio
   * would fail on noise rather than on a wrong derivative.
   */
  absoluteTolerance?: number;
  /** How many entries to sample per tensor. */
  samplesPerTensor?: number;
  /** Ignore targets equal to this token id (padding). */
  padId?: number;
};

/**
 * Compare autodiff against numerical gradients.
 *
 * Relative error uses `|a - n| / max(1e-6, |a| + |n|)`, which stays meaningful
 * when a gradient is exactly zero — the failure mode a plain ratio would hide.
 */
export function numericalGradientCheck(
  model: AlphaTransformer,
  batch: GradientCheckBatch,
  options: GradientCheckOptions = {},
): GradientCheckReport {
  const eps = options.eps ?? 1e-3;
  const tolerance = options.tolerance ?? 5e-2;
  const absoluteTolerance = options.absoluteTolerance ?? 1e-4;
  const samplesPerTensor = options.samplesPerTensor ?? 3;
  if (batch.input.length !== batch.batch * batch.seqLen) {
    throw new Error("[alpha:training] gradient check: input shape does not match batch x seqLen");
  }

  const lossAt = (): number => {
    setGradEnabled(false);
    try {
      const forward = model.forward(batch.input, batch.batch, batch.seqLen, { training: false });
      return crossEntropy(forward.logits, batch.target, options.padId ?? -100).loss;
    } finally {
      setGradEnabled(true);
    }
  };

  // Analytic gradients.
  setGradEnabled(true);
  for (const parameter of model.parameters()) parameter.tensor.zeroGrad();
  const forward = model.forward(batch.input, batch.batch, batch.seqLen, { training: false });
  const loss = crossEntropy(forward.logits, batch.target, options.padId ?? -100);
  backward(loss.tensor);

  let checkedParameters = 0;
  let checkedValues = 0;
  let maxAbsoluteError = 0;
  let maxRelativeError = 0;
  let maxRelativeErrorAboveFloor = 0;
  let passed = true;
  const failures: string[] = [];

  for (const parameter of model.parameters()) {
    const tensor = parameter.tensor;
    const analytic = tensor.grad;
    if (!analytic) continue;
    checkedParameters++;
    const stride = Math.max(1, Math.floor(tensor.size / samplesPerTensor));
    let sampled = 0;
    for (let index = 0; index < tensor.size && sampled < samplesPerTensor; index += stride) {
      sampled++;
      const original = tensor.data[index];
      tensor.data[index] = original + eps;
      const lossPlus = lossAt();
      tensor.data[index] = original - eps;
      const lossMinus = lossAt();
      tensor.data[index] = original;

      const numerical = (lossPlus - lossMinus) / (2 * eps);
      const absolute = Math.abs(analytic[index] - numerical);
      const relative = absolute / Math.max(1e-6, Math.abs(analytic[index]) + Math.abs(numerical));
      if (absolute > maxAbsoluteError) maxAbsoluteError = absolute;
      if (relative > maxRelativeError) maxRelativeError = relative;
      checkedValues++;
      if (absolute > absoluteTolerance && relative > maxRelativeErrorAboveFloor) {
        maxRelativeErrorAboveFloor = relative;
      }
      const ok = absolute <= absoluteTolerance || relative <= tolerance;
      if (!ok && failures.length < 5) {
        failures.push(`${parameter.name}[${index}] autodiff ${analytic[index].toExponential(3)} vs numerical ${numerical.toExponential(3)} (absolute ${absolute.toExponential(3)}, relative ${relative.toExponential(3)})`);
      }
      if (!ok) passed = false;
    }
  }

  setGradEnabled(true);

  return {
    checkedParameters,
    checkedValues,
    maxAbsoluteError,
    maxRelativeError,
    tolerance,
    absoluteTolerance,
    eps,
    passed,
    detail: passed
      ? `Autodiff matches central differences on ${checkedValues} sampled value(s) across ${checkedParameters} tensor(s); largest relative error above the ${absoluteTolerance.toExponential(0)} absolute floor is ${maxRelativeErrorAboveFloor.toExponential(3)} (tolerance ${tolerance}), largest absolute error ${maxAbsoluteError.toExponential(3)}.`
      : `Autodiff diverges from central differences: ${failures.join("; ")}`,
  };
}
