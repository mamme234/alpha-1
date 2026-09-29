/**
 * Alpha Core — Tensor & reverse-mode autodiff engine.
 *
 * This is the numerical foundation of Alpha's own language model. Everything
 * above it (transformer, training engine, embeddings) is written against this
 * interface. It is a small but *complete* autodiff implementation: every op
 * here has a real analytic backward pass, so gradients are true gradients, not
 * approximations and not delegated to any external library.
 *
 * Scope: dense tensors of rank 1..3, float32 storage, batched matrix multiply.
 */

export type Shape = number[];

/** Nested JS numbers, e.g. `[1, 2]`, `[[1, 2]]` or `[[[1, 2]]]`. */
export type NestedNumbers = number | NestedNumbers[];

/** Receives the upstream gradient of the op's output. */
export type BackwardFn = (grad: Float32Array) => void;

let gradEnabled = true;

/** Disable gradient recording for pure inference (saves memory and time). */
export function setGradEnabled(enabled: boolean): void {
  gradEnabled = enabled;
}

export function isGradEnabled(): boolean {
  return gradEnabled;
}

export function shapeSize(shape: Shape): number {
  let size = 1;
  for (const dim of shape) size *= dim;
  return size;
}

export function shapeStrides(shape: Shape): number[] {
  const strides = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= shape[i];
  }
  return strides;
}

/** Infer a shape from nested arrays, validating that every level is uniform. */
export function inferShape(values: NestedNumbers): Shape {
  const shape: Shape = [];
  let node: NestedNumbers = values;
  while (Array.isArray(node)) {
    shape.push(node.length);
    const first = node[0];
    if (first !== undefined && node.some((child) => Array.isArray(child) !== Array.isArray(first))) {
      throw new Error("[alpha:core] ragged nested arrays cannot be turned into a tensor");
    }
    node = (first ?? 0) as NestedNumbers;
  }
  if (shape.length === 0) shape.push(1);
  return shape;
}

export function assertShape(actual: Shape, expected: Shape, label: string): void {
  const a = actual.join("x");
  const e = expected.join("x");
  if (a !== e) {
    throw new Error(`[alpha:core] shape mismatch in ${label}: expected ${e}, got ${a}`);
  }
}

export class Tensor {
  readonly shape: Shape;
  readonly size: number;
  data: Float32Array;
  grad: Float32Array | null = null;
  requiresGrad: boolean;
  parents: Tensor[] = [];
  backwardFn: BackwardFn | null = null;

  constructor(data: Float32Array, shape: Shape, requiresGrad = false) {
    if (shapeSize(shape) !== data.length) {
      throw new Error(
        `[alpha:core] data length ${data.length} does not match shape ${shape.join("x")}`,
      );
    }
    this.data = data;
    this.shape = shape;
    this.size = data.length;
    this.requiresGrad = requiresGrad && gradEnabled;
  }

  static zeros(shape: Shape, requiresGrad = false): Tensor {
    return new Tensor(new Float32Array(shapeSize(shape)), shape, requiresGrad);
  }

  /** Build a tensor from nested numbers of any depth (1-D to 3-D in practice). */
  static from(values: NestedNumbers, requiresGrad = false): Tensor {
    const shape = inferShape(values);
    const flat = new Float32Array(shapeSize(shape));
    let cursor = 0;
    const walk = (node: NestedNumbers): void => {
      if (Array.isArray(node)) {
        for (const child of node) walk(child as NestedNumbers);
        return;
      }
      flat[cursor++] = node as number;
    };
    walk(values);
    return new Tensor(flat, shape, requiresGrad);
  }

  get rank(): number {
    return this.shape.length;
  }

  /** Size of the last dimension. */
  get lastDim(): number {
    return this.shape[this.shape.length - 1];
  }

  value(index: number): number {
    return this.data[index];
  }

  /** Allocate (or reuse) a zero gradient buffer. */
  zeroGrad(): void {
    if (!this.grad || this.grad.length !== this.size) {
      this.grad = new Float32Array(this.size);
    } else {
      this.grad.fill(0);
    }
  }

  /** Accumulate `delta` into the gradient buffer. */
  accumulateGrad(delta: Float32Array): void {
    if (!this.grad) this.grad = new Float32Array(this.size);
    const g = this.grad;
    for (let i = 0; i < g.length; i++) g[i] += delta[i];
  }

  /** Snapshot of the first `n` values, useful for debugging and tests. */
  toArray(n = this.size): number[] {
    return Array.from(this.data.subarray(0, Math.min(n, this.size)));
  }

  toMatrix(): number[][] {
    if (this.rank !== 2) {
      throw new Error("[alpha:core] toMatrix() requires a rank-2 tensor");
    }
    const [rows, cols] = this.shape;
    const out: number[][] = [];
    for (let r = 0; r < rows; r++) {
      out.push(Array.from(this.data.subarray(r * cols, (r + 1) * cols)));
    }
    return out;
  }
}

/**
 * Create an op node and attach its backward pass. The backward closure receives
 * the node itself so gradients always accumulate on that exact tensor.
 */
function op(
  shape: Shape,
  data: Float32Array,
  requiresGrad: boolean,
  parents: Tensor[],
  attach: (node: Tensor, grad: Float32Array) => void,
): Tensor {
  const tensor = new Tensor(data, shape, requiresGrad);
  if (requiresGrad && gradEnabled) {
    tensor.parents = parents;
    tensor.backwardFn = (grad) => attach(tensor, grad);
  }
  return tensor;
}

function wantsGrad(...tensors: Tensor[]): boolean {
  return gradEnabled && tensors.some((t) => t.requiresGrad);
}

function gradOf(t: Tensor): Float32Array {
  if (!t.grad) t.grad = new Float32Array(t.size);
  return t.grad;
}

/**
 * Run reverse-mode autodiff from `root` back through the recorded graph.
 * The graph is topologically sorted once per call (cheap relative to fwd/bwd).
 */
export function backward(root: Tensor): void {
  const topo: Tensor[] = [];
  const seen = new Set<Tensor>();
  const visit = (t: Tensor) => {
    if (seen.has(t)) return;
    seen.add(t);
    for (const p of t.parents) visit(p);
    topo.push(t);
  };
  visit(root);
  root.zeroGrad();
  root.grad!.fill(1);
  for (let i = topo.length - 1; i >= 0; i--) {
    const node = topo[i];
    if (node.backwardFn && node.grad) node.backwardFn(node.grad);
  }
}

/** Zero every gradient buffer reachable from the given roots. */
export function resetGrad(roots: Tensor[]): void {
  for (const root of roots) {
    const seen = new Set<Tensor>();
    const stack = [root];
    while (stack.length) {
      const t = stack.pop()!;
      if (seen.has(t)) continue;
      seen.add(t);
      if (t.grad) t.grad.fill(0);
      for (const p of t.parents) stack.push(p);
    }
  }
}

// ---------------------------------------------------------------------------
// Elementwise ops
// ---------------------------------------------------------------------------

/** Elementwise add, with broadcast of a rank-1 bias or a rank-0 scalar. */
export function add(a: Tensor, b: Tensor): Tensor {
  const rg = wantsGrad(a, b);
  if (a.shape.length === b.shape.length) {
    assertShape(a.shape, b.shape, "add");
    const out = new Float32Array(a.size);
    for (let i = 0; i < out.length; i++) out[i] = a.data[i] + b.data[i];
    return op(a.shape, out, rg, [a, b], (_node, grad) => {
      if (a.requiresGrad) {
        const ga = gradOf(a);
        for (let i = 0; i < grad.length; i++) ga[i] += grad[i];
      }
      if (b.requiresGrad) {
        const gb = gradOf(b);
        for (let i = 0; i < grad.length; i++) gb[i] += grad[i];
      }
    });
  }
  // Broadcasting: [..., N] + [N]
  const n = b.size;
  if (a.shape[a.shape.length - 1] !== n) {
    throw new Error("[alpha:core] add: incompatible shapes for broadcast");
  }
  const out = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = a.data[i] + b.data[i % n];
  return op(a.shape, out, rg, [a, b], (_node, grad) => {
    if (a.requiresGrad) {
      const ga = gradOf(a);
      for (let i = 0; i < grad.length; i++) ga[i] += grad[i];
    }
    if (b.requiresGrad) {
      const gb = gradOf(b);
      for (let i = 0; i < grad.length; i++) gb[i % n] += grad[i];
    }
  });
}

export function addScalar(a: Tensor, value: number): Tensor {
  const rg = wantsGrad(a);
  const out = new Float32Array(a.size);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] + value;
  return op(a.shape, out, rg, [a], (_node, grad) => {
    if (!a.requiresGrad) return;
    const ga = gradOf(a);
    for (let i = 0; i < grad.length; i++) ga[i] += grad[i];
  });
}

export function scale(a: Tensor, factor: number): Tensor {
  const rg = wantsGrad(a);
  const out = new Float32Array(a.size);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] * factor;
  return op(a.shape, out, rg, [a], (_node, grad) => {
    if (!a.requiresGrad) return;
    const ga = gradOf(a);
    for (let i = 0; i < grad.length; i++) ga[i] += grad[i] * factor;
  });
}

export function mulElementwise(a: Tensor, b: Tensor): Tensor {
  assertShape(a.shape, b.shape, "mulElementwise");
  const rg = wantsGrad(a, b);
  const out = new Float32Array(a.size);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] * b.data[i];
  return op(a.shape, out, rg, [a, b], (_node, grad) => {
    if (a.requiresGrad) {
      const ga = gradOf(a);
      for (let i = 0; i < grad.length; i++) ga[i] += grad[i] * b.data[i];
    }
    if (b.requiresGrad) {
      const gb = gradOf(b);
      for (let i = 0; i < grad.length; i++) gb[i] += grad[i] * a.data[i];
    }
  });
}

// ---------------------------------------------------------------------------
// Shape ops
// ---------------------------------------------------------------------------

export function reshape(a: Tensor, shape: Shape): Tensor {
  const rg = wantsGrad(a);
  if (shapeSize(shape) !== a.size) {
    throw new Error("[alpha:core] reshape: size mismatch");
  }
  // Sharing the buffer is safe: every op reads only and every backward writes
  // to the parent's own gradient buffer.
  return op(shape, a.data, rg, [a], (_node, grad) => {
    if (!a.requiresGrad) return;
    const ga = gradOf(a);
    for (let i = 0; i < grad.length; i++) ga[i] += grad[i];
  });
}

/** Swap the last two dimensions of a rank-3 tensor (used for K^T). */
export function transposeLastTwo(a: Tensor): Tensor {
  const [b, m, n] = a.shape;
  const rg = wantsGrad(a);
  const out = new Float32Array(a.size);
  for (let bi = 0; bi < b; bi++) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        out[bi * n * m + j * m + i] = a.data[bi * m * n + i * n + j];
      }
    }
  }
  return op([b, n, m], out, rg, [a], (_node, grad) => {
    if (!a.requiresGrad) return;
    const ga = gradOf(a);
    for (let bi = 0; bi < b; bi++) {
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          ga[bi * m * n + i * n + j] += grad[bi * n * m + j * m + i];
        }
      }
    }
  });
}

/**
 * Generic permutation/gather over flat storage.
 * `map[outIndex] = inIndex` — the backward pass scatters the same way, so any
 * reindexing op (head split, slicing) is exactly reversible.
 */
export function permute(x: Tensor, map: Int32Array, outShape: Shape): Tensor {
  const rg = wantsGrad(x);
  const out = new Float32Array(map.length);
  for (let i = 0; i < map.length; i++) out[i] = x.data[map[i]];
  return op(outShape, out, rg, [x], (_node, grad) => {
    if (!x.requiresGrad) return;
    const gx = gradOf(x);
    for (let i = 0; i < map.length; i++) gx[map[i]] += grad[i];
  });
}

/**
 * Split [B, T, C] into per-head tensors packed as [B*H, T, headDim].
 * Keeps head math batched so the whole block runs through `matmul`.
 */
export function splitHeads(x: Tensor, nHeads: number): Tensor {
  const [b, t, c] = x.shape;
  const headDim = c / nHeads;
  if (!Number.isInteger(headDim)) {
    throw new Error("[alpha:core] splitHeads: model width must divide by heads");
  }
  const map = new Int32Array(b * nHeads * t * headDim);
  let k = 0;
  for (let bi = 0; bi < b; bi++) {
    for (let h = 0; h < nHeads; h++) {
      for (let ti = 0; ti < t; ti++) {
        for (let d = 0; d < headDim; d++) {
          map[k++] = bi * t * c + ti * c + h * headDim + d;
        }
      }
    }
  }
  return permute(x, map, [b * nHeads, t, headDim]);
}

/** Inverse of `splitHeads`: [B*H, T, headDim] -> [B, T, C]. */
export function mergeHeads(x: Tensor, batch: number, seqLen: number, nHeads: number): Tensor {
  const [bh, t, headDim] = x.shape;
  if (bh !== batch * nHeads || t !== seqLen) {
    throw new Error("[alpha:core] mergeHeads: shape does not match batch configuration");
  }
  const c = nHeads * headDim;
  const map = new Int32Array(batch * seqLen * c);
  let k = 0;
  for (let bi = 0; bi < batch; bi++) {
    for (let ti = 0; ti < t; ti++) {
      for (let h = 0; h < nHeads; h++) {
        for (let d = 0; d < headDim; d++) {
          map[k++] = (bi * nHeads + h) * t * headDim + ti * headDim + d;
        }
      }
    }
  }
  return permute(x, map, [batch, seqLen, c]);
}

/** Take a contiguous window of rows along the sequence axis of [B, T, C]. */
export function sliceSeq(x: Tensor, start: number, length: number): Tensor {
  const [b, t, c] = x.shape;
  if (start + length > t) {
    throw new Error("[alpha:core] sliceSeq: window out of range");
  }
  const map = new Int32Array(b * length * c);
  let k = 0;
  for (let bi = 0; bi < b; bi++) {
    for (let ti = 0; ti < length; ti++) {
      for (let ci = 0; ci < c; ci++) {
        map[k++] = bi * t * c + (start + ti) * c + ci;
      }
    }
  }
  return permute(x, map, [b, length, c]);
}

/** Gather rows from a rank-2 table (token embedding lookup). */
export function gatherRows(table: Tensor, indices: Int32Array): Tensor {
  const [rows, width] = table.shape;
  const rg = wantsGrad(table);
  const out = new Float32Array(indices.length * width);
  for (let i = 0; i < indices.length; i++) {
    const row = indices[i];
    if (row < 0 || row >= rows) {
      throw new Error(`[alpha:core] gatherRows: index ${row} out of range (0..${rows - 1})`);
    }
    for (let j = 0; j < width; j++) out[i * width + j] = table.data[row * width + j];
  }
  return op([indices.length, width], out, rg, [table], (_node, grad) => {
    if (!table.requiresGrad) return;
    const gt = gradOf(table);
    for (let i = 0; i < indices.length; i++) {
      const row = indices[i] * width;
      for (let j = 0; j < width; j++) gt[row + j] += grad[i * width + j];
    }
  });
}

// ---------------------------------------------------------------------------
// Matrix multiply
// ---------------------------------------------------------------------------

/**
 * Batched matrix multiply: a[..., M, K] x b[..., K, N] -> [..., M, N].
 * Leading dimensions must match. Backward is the standard two-multiply form.
 */
export function matmul(a: Tensor, b: Tensor): Tensor {
  const rankA = a.rank;
  const rankB = b.rank;
  const m = a.shape[rankA - 2];
  const k = a.shape[rankA - 1];
  const k2 = b.shape[rankB - 2];
  const n = b.shape[rankB - 1];
  if (k !== k2) {
    throw new Error(`[alpha:core] matmul: inner dims differ (${k} vs ${k2})`);
  }
  const batchA = a.size / (m * k);
  const batchB = b.size / (k * n);
  if (batchA !== batchB) {
    throw new Error(`[alpha:core] matmul: batch mismatch (${batchA} vs ${batchB})`);
  }
  const batch = batchA;
  const out = new Float32Array(batch * m * n);
  const ad = a.data;
  const bd = b.data;
  for (let bi = 0; bi < batch; bi++) {
    const aOff = bi * m * k;
    const bOff = bi * k * n;
    const oOff = bi * m * n;
    for (let i = 0; i < m; i++) {
      for (let kk = 0; kk < k; kk++) {
        const av = ad[aOff + i * k + kk];
        if (av === 0) continue;
        const bRow = bOff + kk * n;
        const oRow = oOff + i * n;
        for (let j = 0; j < n; j++) out[oRow + j] += av * bd[bRow + j];
      }
    }
  }
  const outShape: Shape =
    rankA === 2 && rankB === 2 ? [m, n] : [...a.shape.slice(0, rankA - 2), m, n];
  const rg = wantsGrad(a, b);
  return op(outShape, out, rg, [a, b], (_node, grad) => {
    const ga = a.requiresGrad ? gradOf(a) : null;
    const gb = b.requiresGrad ? gradOf(b) : null;
    for (let bi = 0; bi < batch; bi++) {
      const aOff = bi * m * k;
      const bOff = bi * k * n;
      const oOff = bi * m * n;
      if (ga) {
        // ga += gY @ b^T
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < n; j++) {
            const gv = grad[oOff + i * n + j];
            if (gv === 0) continue;
            for (let kk = 0; kk < k; kk++) {
              ga[aOff + i * k + kk] += gv * bd[bOff + kk * n + j];
            }
          }
        }
      }
      if (gb) {
        // gb += a^T @ gY
        for (let kk = 0; kk < k; kk++) {
          for (let i = 0; i < m; i++) {
            const av = ad[aOff + i * k + kk];
            if (av === 0) continue;
            for (let j = 0; j < n; j++) {
              gb[bOff + kk * n + j] += av * grad[oOff + i * n + j];
            }
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Non-linearities and normalisation
// ---------------------------------------------------------------------------

const GELU_C = Math.sqrt(2 / Math.PI);

/** GELU (tanh approximation) with its exact analytic derivative. */
export function gelu(a: Tensor): Tensor {
  const rg = wantsGrad(a);
  const n = a.size;
  const out = new Float32Array(n);
  const tanhValues = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = a.data[i];
    const u = GELU_C * (x + 0.044715 * x * x * x);
    const t = Math.tanh(u);
    tanhValues[i] = t;
    out[i] = 0.5 * x * (1 + t);
  }
  return op(a.shape, out, rg, [a], (_node, grad) => {
    if (!a.requiresGrad) return;
    const ga = gradOf(a);
    for (let i = 0; i < n; i++) {
      const x = a.data[i];
      const t = tanhValues[i];
      const du = GELU_C * (1 + 3 * 0.044715 * x * x);
      const dy = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du;
      ga[i] += grad[i] * dy;
    }
  });
}

/**
 * Layer normalization over the last dimension of [B, T, C] (or [N, C]).
 * Weight/bias are learnable rank-1 tensors.
 */
export function layerNorm(a: Tensor, weight: Tensor, bias: Tensor, eps = 1e-5): Tensor {
  const c = a.lastDim;
  const rows = a.size / c;
  const rg = wantsGrad(a, weight, bias);
  const out = new Float32Array(a.size);
  const xhat = new Float32Array(a.size);
  const invStd = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    const off = r * c;
    let mean = 0;
    for (let i = 0; i < c; i++) mean += a.data[off + i];
    mean /= c;
    let variance = 0;
    for (let i = 0; i < c; i++) {
      const d = a.data[off + i] - mean;
      variance += d * d;
    }
    variance /= c;
    const inv = 1 / Math.sqrt(variance + eps);
    invStd[r] = inv;
    for (let i = 0; i < c; i++) {
      const h = (a.data[off + i] - mean) * inv;
      xhat[off + i] = h;
      out[off + i] = h * weight.data[i] + bias.data[i];
    }
  }
  return op(a.shape, out, rg, [a, weight, bias], (_node, grad) => {
    const gw = weight.requiresGrad ? gradOf(weight) : null;
    const gb = bias.requiresGrad ? gradOf(bias) : null;
    const ga = a.requiresGrad ? gradOf(a) : null;
    for (let r = 0; r < rows; r++) {
      const off = r * c;
      const inv = invStd[r];
      let sumDy = 0;
      let sumDyXhat = 0;
      for (let i = 0; i < c; i++) {
        const dy = grad[off + i] * weight.data[i];
        if (gw) gw[i] += grad[off + i] * xhat[off + i];
        if (gb) gb[i] += grad[off + i];
        sumDy += dy;
        sumDyXhat += dy * xhat[off + i];
      }
      if (ga) {
        const meanDy = sumDy / c;
        const meanDyXhat = sumDyXhat / c;
        for (let i = 0; i < c; i++) {
          const dy = grad[off + i] * weight.data[i];
          ga[off + i] += inv * (dy - meanDy - xhat[off + i] * meanDyXhat);
        }
      }
    }
  });
}

/** Causal masked softmax over the last axis of [B, T, T]. */
export function causalSoftmax(scores: Tensor): Tensor {
  const [b, t, u] = scores.shape;
  if (t !== u) {
    throw new Error("[alpha:core] causalSoftmax expects a square [B, T, T] tensor");
  }
  const rg = wantsGrad(scores);
  const out = new Float32Array(scores.size);
  const rows = b * t;
  for (let r = 0; r < rows; r++) {
    const off = r * u;
    const column = r % t;
    let max = -Infinity;
    for (let j = 0; j <= column; j++) {
      if (scores.data[off + j] > max) max = scores.data[off + j];
    }
    let sum = 0;
    for (let j = 0; j < u; j++) {
      if (j > column) {
        out[off + j] = 0;
        continue;
      }
      const e = Math.exp(scores.data[off + j] - max);
      out[off + j] = e;
      sum += e;
    }
    const inv = sum > 0 ? 1 / sum : 0;
    for (let j = 0; j <= column; j++) out[off + j] *= inv;
  }
  return op(scores.shape, out, rg, [scores], (_node, grad) => {
    if (!scores.requiresGrad) return;
    const gs = gradOf(scores);
    for (let r = 0; r < rows; r++) {
      const off = r * u;
      let dot = 0;
      for (let j = 0; j < u; j++) dot += grad[off + j] * out[off + j];
      for (let j = 0; j < u; j++) {
        gs[off + j] += out[off + j] * (grad[off + j] - dot);
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Loss
// ---------------------------------------------------------------------------

export type CrossEntropyResult = {
  loss: number;
  tensor: Tensor;
  /** exp(loss) in nats/token — the value to report next to a uniform baseline. */
  perplexity: number;
  tokens: number;
};

/**
 * Fused softmax + cross-entropy over [N, V] logits with integer targets.
 * `ignoreIndex` excludes padding tokens from the loss.
 */
export function crossEntropy(
  logits: Tensor,
  targets: Int32Array,
  ignoreIndex = -100,
): CrossEntropyResult {
  const [n, v] = logits.shape;
  if (targets.length !== n) {
    throw new Error("[alpha:core] crossEntropy: target count must match rows");
  }
  const probs = new Float32Array(logits.size);
  let lossSum = 0;
  let counted = 0;
  for (let i = 0; i < n; i++) {
    const off = i * v;
    const target = targets[i];
    let max = -Infinity;
    for (let j = 0; j < v; j++) if (logits.data[off + j] > max) max = logits.data[off + j];
    let sum = 0;
    for (let j = 0; j < v; j++) {
      const e = Math.exp(logits.data[off + j] - max);
      probs[off + j] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let j = 0; j < v; j++) probs[off + j] *= inv;
    if (target === ignoreIndex || target < 0 || target >= v) continue;
    lossSum += -Math.log(probs[off + target] + 1e-12);
    counted++;
  }
  const mean = counted > 0 ? lossSum / counted : 0;
  const rg = wantsGrad(logits);
  const lossTensor = op([1], Float32Array.from([mean]), rg, [logits], (node, grad) => {
    if (!logits.requiresGrad) return;
    const gl = gradOf(logits);
    const norm = counted > 0 ? grad[0] / counted : 0;
    for (let i = 0; i < n; i++) {
      const off = i * v;
      const target = targets[i];
      const valid = target !== ignoreIndex && target >= 0 && target < v;
      for (let j = 0; j < v; j++) {
        let d = probs[off + j];
        if (valid && j === target) d -= 1;
        if (!valid) d = 0;
        gl[off + j] += d * norm;
      }
    }
    void node;
  });
  return {
    loss: mean,
    tensor: lossTensor,
    perplexity: Math.exp(Math.min(mean, 20)),
    tokens: counted,
  };
}

// ---------------------------------------------------------------------------
// Regularisation
// ---------------------------------------------------------------------------

export type RandomFn = () => number;

/** Inverted dropout. A no-op (identity) when `training` is false. */
export function dropout(a: Tensor, p: number, rng: RandomFn, training: boolean): Tensor {
  if (!training || p <= 0) return a;
  const keep = 1 - p;
  const mask = new Float32Array(a.size);
  const out = new Float32Array(a.size);
  for (let i = 0; i < out.length; i++) {
    const kept = rng() < keep ? 1 : 0;
    mask[i] = kept / keep;
    out[i] = a.data[i] * mask[i];
  }
  const rg = wantsGrad(a);
  return op(a.shape, out, rg, [a], (_node, grad) => {
    if (!a.requiresGrad) return;
    const ga = gradOf(a);
    for (let i = 0; i < grad.length; i++) ga[i] += grad[i] * mask[i];
  });
}

// ---------------------------------------------------------------------------
// Utilities used by tests, diagnostics and the training engine
// ---------------------------------------------------------------------------

/** Max absolute difference between two same-shaped tensors. */
export function maxAbsDiff(a: Tensor, b: Tensor): number {
  assertShape(a.shape, b.shape, "maxAbsDiff");
  let max = 0;
  for (let i = 0; i < a.size; i++) {
    const d = Math.abs(a.data[i] - b.data[i]);
    if (d > max) max = d;
  }
  return max;
}

/** L2 norm of a tensor's gradient — used for gradient clipping and tests. */
export function gradL2Norm(tensors: Tensor[]): number {
  let sum = 0;
  for (const t of tensors) {
    if (!t.grad) continue;
    for (let i = 0; i < t.grad.length; i++) sum += t.grad[i] * t.grad[i];
  }
  return Math.sqrt(sum);
}
