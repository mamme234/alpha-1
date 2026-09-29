import { describe, expect, it } from "vitest";
import {
  Tensor,
  add,
  backward,
  causalSoftmax,
  crossEntropy,
  gelu,
  layerNorm,
  matmul,
  mergeHeads,
  mulElementwise,
  reshape,
  setGradEnabled,
  splitHeads,
  transposeLastTwo,
} from "../core/tensor";

/** Sum of all elements, expressed with the public ops. */
function sumAll(t: Tensor): Tensor {
  const flat = reshape(t, [1, t.size]);
  const ones = Tensor.from(new Array<number>(t.size).fill(1));
  return matmul(flat, reshape(ones, [t.size, 1]));
}

function sumSquares(t: Tensor): Tensor {
  return sumAll(mulElementwise(t, t));
}

/**
 * Numerical gradient check: compare the analytic backward pass against a
 * central difference of the scalar loss. This is what makes "the gradients are
 * real" a checkable claim instead of a comment.
 */
function expectGradientMatches(
  params: Tensor[],
  forward: () => Tensor,
  lossOf: (output: Tensor) => Tensor,
  tolerance = 2e-2,
): void {
  const output = forward();
  setGradEnabled(true);
  const loss = lossOf(output);
  backward(loss);
  const analytic = params.map((param) => (param.grad ? param.grad.slice() : new Float32Array(param.size)));

  const epsilon = 1e-3;
  for (let p = 0; p < params.length; p++) {
    const param = params[p];
    for (let i = 0; i < param.size; i++) {
      const original = param.data[i];
      setGradEnabled(false);
      param.data[i] = original + epsilon;
      const plus = lossOf(forward()).data[0];
      param.data[i] = original - epsilon;
      const minus = lossOf(forward()).data[0];
      param.data[i] = original;
      setGradEnabled(true);
      const numeric = (plus - minus) / (2 * epsilon);
      expect(Math.abs(analytic[p][i] - numeric)).toBeLessThan(tolerance);
    }
  }
  setGradEnabled(true);
}

describe("alpha core tensor engine", () => {
  it("computes the matmul forward pass", () => {
    const a = Tensor.from([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = Tensor.from([
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
    const y = matmul(a, b);
    expect(y.shape).toEqual([2, 2]);
    expect(y.toArray()).toEqual([4, 5, 10, 11]);
  });

  it("computes matmul gradients that match numerical differences", () => {
    const a = Tensor.from(
      [
        [0.5, -1.0],
        [1.5, 0.25],
      ],
      true,
    );
    const b = Tensor.from(
      [
        [0.75, 1.25],
        [-0.5, 0.5],
      ],
      true,
    );
    expectGradientMatches([a, b], () => matmul(a, b), sumSquares);
  });

  it("computes batched matmul gradients", () => {
    const a = Tensor.from([[[1, 0.5], [0.25, 2]], [[-1, 0.5], [0.75, 1]]], true); // [2,2,2]
    const b = Tensor.from([[[0.5, 1], [1, 0.5]], [[0.25, 2], [1, -0.5]]], true);
    const y = matmul(a, b);
    expect(y.shape).toEqual([2, 2, 2]);
    expectGradientMatches([a, b], () => matmul(a, b), sumSquares);
  });

  it("computes layerNorm gradients that match numerical differences", () => {
    const x = Tensor.from(
      [
        [0.5, -1.2, 2.0, 0.3],
        [1.1, 0.2, -0.4, 0.9],
      ],
      true,
    );
    const weight = Tensor.from([1.0, 0.8, 1.2, 0.5], true);
    const bias = Tensor.from([0.1, -0.1, 0.05, 0.0], true);
    const y = layerNorm(x, weight, bias, 1e-5);
    expect(y.shape).toEqual([2, 4]);
    expectGradientMatches([x, weight, bias], () => layerNorm(x, weight, bias, 1e-5), sumSquares);
  });

  it("computes gelu gradients that match numerical differences", () => {
    const x = Tensor.from([-1.5, -0.2, 0.0, 0.7, 2.4], true);
    const y = gelu(x);
    expect(y.data[2]).toBeCloseTo(0, 6);
    expectGradientMatches([x], () => gelu(x), sumSquares);
  });

  it("masks future positions in causal attention", () => {
    const scores = Tensor.from([
      [
        [1, 9, 9],
        [1, 1, 9],
        [1, 1, 1],
      ],
    ]);
    const probs = causalSoftmax(scores);
    expect(probs.data[1]).toBe(0);
    expect(probs.data[2]).toBe(0);
    expect(probs.data[0]).toBeCloseTo(1, 6);
    expect(probs.data[5]).toBe(0);
    expect(probs.data[3] + probs.data[4]).toBeCloseTo(1, 6);
  });

  it("computes causal softmax gradients", () => {
    const scores = Tensor.from(
      [
        [
          [0.3, 0.6, 0.9],
          [0.2, -0.4, 0.1],
          [1.1, 0.3, -0.7],
        ],
      ],
      true,
    );
    expectGradientMatches([scores], () => causalSoftmax(scores), sumSquares, 5e-2);
  });

  it("splits and merges attention heads without losing values", () => {
    const x = Tensor.from([
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
    ]); // [1,2,4]
    const heads = splitHeads(x, 2);
    expect(heads.shape).toEqual([2, 2, 2]);
    const merged = mergeHeads(heads, 1, 2, 2);
    expect(merged.shape).toEqual([1, 2, 4]);
    expect(merged.toArray()).toEqual(x.toArray());
  });

  it("transposes the last two dimensions and reverses the gradient", () => {
    const a = Tensor.from(
      [
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
      ],
      true,
    );
    const t = transposeLastTwo(a);
    expect(t.shape).toEqual([1, 3, 2]);
    expect(t.toArray()).toEqual([1, 4, 2, 5, 3, 6]);
    expectGradientMatches([a], () => transposeLastTwo(a), sumSquares);
  });

  it("produces cross-entropy gradients equal to softmax minus one-hot", () => {
    const logits = Tensor.from(
      [
        [0.2, 0.5, -0.3],
        [1.0, -1.0, 0.4],
      ],
      true,
    );
    const targets = Int32Array.from([1, 0]);
    const result = crossEntropy(logits, targets);
    expect(result.tokens).toBe(2);
    expect(result.loss).toBeGreaterThan(0);
    backward(result.tensor);
    const row0 = [Math.exp(0.2), Math.exp(0.5), Math.exp(-0.3)];
    const sum0 = row0.reduce((a, b) => a + b, 0);
    for (let j = 0; j < 3; j++) {
      const expected = (row0[j] / sum0 - (j === 1 ? 1 : 0)) / 2;
      expect(logits.grad![j]).toBeCloseTo(expected, 5);
    }
  });

  it("ignores padded targets in the loss", () => {
    const logits = Tensor.from([
      [0.1, 0.2],
      [0.3, 0.4],
    ]);
    const withPad = crossEntropy(logits, Int32Array.from([0, -100]), -100);
    const withoutPad = crossEntropy(logits, Int32Array.from([0, 1]), -100);
    expect(withPad.tokens).toBe(1);
    expect(withoutPad.tokens).toBe(2);
    // The padded run reports exactly the first row's loss, on its own.
    const row0 = [Math.exp(0.1), Math.exp(0.2)];
    const expected = -Math.log(row0[0] / (row0[0] + row0[1]));
    expect(withPad.loss).toBeCloseTo(expected, 5);
    expect(withPad.loss).not.toBeCloseTo(withoutPad.loss, 3);
  });

  it("broadcasts a bias vector during the add backward pass", () => {
    const a = Tensor.from(
      [
        [0.1, 0.2],
        [0.3, 0.4],
      ],
      true,
    );
    const bias = Tensor.from([0.5, 0.6], true);
    const out = add(a, bias);
    expect(out.toArray()[0]).toBeCloseTo(0.6, 6);
    expect(out.toArray()[3]).toBeCloseTo(1.0, 6);
    expectGradientMatches([a, bias], () => add(a, bias), sumSquares);
  });
});
