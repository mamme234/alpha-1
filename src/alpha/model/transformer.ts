/**
 * Alpha LLM Core — the transformer itself.
 *
 * Architecture: decoder-only transformer.
 *   token embedding + positional signal
 *   -> N x [ layer norm -> multi-head causal self-attention -> residual
 *            -> layer norm -> feed-forward (GELU) -> residual ]
 *   -> final layer norm -> output projection (tied to the token embedding)
 *
 * Every operation is implemented in `../core/tensor` with a real backward
 * pass, so this model trains by backpropagation through Alpha's own autodiff
 * engine. Nothing is fetched, nothing is proxied.
 *
 * Parameter names are stable strings (`layer0.attn.wq`, ...) — checkpoints and
 * the workspace UI both depend on that.
 */

import { AlphaRng } from "../core/rng";
import { AlphaValidationError } from "../core/errors";
import { base64ToFloat32, float32ToBase64 } from "../core/serialize";
import {
  type Shape,
  Tensor,
  add,
  causalSoftmax,
  dropout,
  gatherRows,
  gelu,
  layerNorm,
  matmul,
  mergeHeads,
  reshape,
  scale,
  splitHeads,
  transposeLastTwo,
} from "../core/tensor";
import { type AlphaModelConfig, countParameters, validateModelConfig } from "./config";
import { kvCacheCommit, kvCacheWriteLayer, type KvCache } from "./kv-cache";

/** Options for the cached, incremental inference path. */
export type CachedForwardOptions = {
  /**
   * Cache to read and write. `null` means "prime the cache from scratch": the
   * whole prompt is processed and every position's keys/values are stored, so
   * the next call can be a single-token step.
   */
  cache: KvCache;
  /** Reuse keys/values already in the cache and only process these ids. */
  incremental: boolean;
};

export type ForwardOptions = {
  /** Enable dropout during training. */
  training?: boolean;
  rng?: AlphaRng;
  /** Return the final hidden state as well as the logits. */
  returnHidden?: boolean;
};

export type ForwardResult = {
  /** Logits shaped [batch * seq, vocab]. */
  logits: Tensor;
  batch: number;
  seq: number;
  /** Final layer-normed hidden states, [batch, seq, dModel]. */
  hidden: Tensor | null;
};

export type ModelParameter = {
  name: string;
  tensor: Tensor;
};

type LayerParameters = {
  attn_wq: Tensor;
  attn_wk: Tensor;
  attn_wv: Tensor;
  attn_wo: Tensor;
  attn_bq: Tensor;
  attn_bk: Tensor;
  attn_bv: Tensor;
  attn_bo: Tensor;
  norm1_w: Tensor;
  norm1_b: Tensor;
  mlp_w1: Tensor;
  mlp_b1: Tensor;
  mlp_w2: Tensor;
  mlp_b2: Tensor;
  norm2_w: Tensor;
  norm2_b: Tensor;
};

export type SerializedWeights = {
  config: AlphaModelConfig;
  /** parameter name -> base64 float32 */
  tensors: Record<string, string>;
  shapes: Record<string, Shape>;
};

function uniformTable(rows: number, cols: number, rng: AlphaRng, std: number): Tensor {
  const data = new Float32Array(rows * cols);
  for (let i = 0; i < data.length; i++) data[i] = rng.normal() * std;
  return new Tensor(data, [rows, cols], true);
}

function zerosRow(cols: number): Tensor {
  return new Tensor(new Float32Array(cols), [cols], true);
}

function onesRow(cols: number): Tensor {
  const data = new Float32Array(cols);
  data.fill(1);
  return new Tensor(data, [cols], true);
}

/** y = x @ W + b, with x flattened to 2-D and restored afterwards. */
function linear(x: Tensor, weight: Tensor, bias: Tensor | null): Tensor {
  const c = x.lastDim;
  const outFeatures = weight.shape[1];
  if (weight.shape[0] !== c) {
    throw new AlphaValidationError(
      "model",
      `linear: weight expects ${weight.shape[0]} inputs but received ${c}`,
    );
  }
  const leading = x.shape.slice(0, -1);
  const flat = reshape(x, [x.size / c, c]);
  const projected = reshape(matmul(flat, weight), [...leading, outFeatures]);
  return bias ? add(projected, bias) : projected;
}

export class AlphaTransformer {
  readonly config: AlphaModelConfig;
  /** Token embedding table, also used as the tied output projection. */
  tokEmbedding: Tensor;
  posEmbedding: Tensor | null;
  layers: LayerParameters[] = [];
  finalNormW: Tensor;
  finalNormB: Tensor;
  outputProjection: Tensor | null = null;
  outputBias: Tensor | null = null;

  private readonly parameterList: ModelParameter[] = [];

  constructor(config: AlphaModelConfig, seed = 1337) {
    validateModelConfig(config);
    this.config = config;
    const rng = new AlphaRng(seed);
    const c = config.dModel;
    const std = config.initStd;

    this.tokEmbedding = uniformTable(config.vocabSize, c, rng, std);
    this.tokEmbedding = this.nameTensor("token_embedding", this.tokEmbedding);
    this.posEmbedding =
      config.positionalEncoding === "learned"
        ? this.nameTensor("position_embedding", uniformTable(config.contextLength, c, rng, std))
        : null;

    for (let l = 0; l < config.nLayers; l++) {
      this.layers.push({
        attn_wq: this.nameTensor(`layer${l}.attn.wq`, uniformTable(c, c, rng, std)),
        attn_wk: this.nameTensor(`layer${l}.attn.wk`, uniformTable(c, c, rng, std)),
        attn_wv: this.nameTensor(`layer${l}.attn.wv`, uniformTable(c, c, rng, std)),
        attn_wo: this.nameTensor(`layer${l}.attn.wo`, uniformTable(c, c, rng, std)),
        attn_bq: this.nameTensor(`layer${l}.attn.bq`, zerosRow(c)),
        attn_bk: this.nameTensor(`layer${l}.attn.bk`, zerosRow(c)),
        attn_bv: this.nameTensor(`layer${l}.attn.bv`, zerosRow(c)),
        attn_bo: this.nameTensor(`layer${l}.attn.bo`, zerosRow(c)),
        norm1_w: this.nameTensor(`layer${l}.norm1.weight`, onesRow(c)),
        norm1_b: this.nameTensor(`layer${l}.norm1.bias`, zerosRow(c)),
        mlp_w1: this.nameTensor(
          `layer${l}.mlp.w1`,
          uniformTable(c, config.dFeedForward, rng, std),
        ),
        mlp_b1: this.nameTensor(`layer${l}.mlp.b1`, zerosRow(config.dFeedForward)),
        mlp_w2: this.nameTensor(
          `layer${l}.mlp.w2`,
          uniformTable(config.dFeedForward, c, rng, std),
        ),
        mlp_b2: this.nameTensor(`layer${l}.mlp.b2`, zerosRow(c)),
        norm2_w: this.nameTensor(`layer${l}.norm2.weight`, onesRow(c)),
        norm2_b: this.nameTensor(`layer${l}.norm2.bias`, zerosRow(c)),
      });
    }

    this.finalNormW = this.nameTensor("final_norm.weight", onesRow(c));
    this.finalNormB = this.nameTensor("final_norm.bias", zerosRow(c));
    if (!config.tieEmbeddings) {
      this.outputProjection = this.nameTensor(
        "output_projection.weight",
        uniformTable(c, config.vocabSize, rng, std),
      );
      this.outputBias = this.nameTensor(
        "output_projection.bias",
        zerosRow(config.vocabSize),
      );
    }
  }

  private nameTensor(name: string, tensor: Tensor): Tensor {
    this.parameterList.push({ name, tensor });
    return tensor;
  }

  get parameterCount(): number {
    return countParameters(this.config);
  }

  parameters(): ModelParameter[] {
    return this.parameterList;
  }

  parameterMap(): Map<string, Tensor> {
    const map = new Map<string, Tensor>();
    for (const p of this.parameterList) map.set(p.name, p.tensor);
    return map;
  }

  /** Resolved output projection, honouring weight tying. */
  private headWeight(): Tensor {
    if (this.outputProjection) return this.outputProjection;
    // [vocab, dModel] -> [1, vocab, dModel] -> [1, dModel, vocab] -> [dModel, vocab]
    const asHeads = reshape(this.tokEmbedding, [1, this.config.vocabSize, this.config.dModel]);
    const transposed = transposeLastTwo(asHeads);
    return reshape(transposed, [this.config.dModel, this.config.vocabSize]);
  }

  /** Fixed sinusoidal table, used when positionalEncoding === "sinusoidal". */
  private sinusoidalPositions(seq: number): Tensor {
    return this.sinusoidalPositionsFrom(0, seq);
  }

  /** Sinusoidal rows for absolute positions [start, start + seq). */
  private sinusoidalPositionsFrom(start: number, seq: number): Tensor {
    const c = this.config.dModel;
    const data = new Float32Array(seq * c);
    for (let t = 0; t < seq; t++) {
      const position = start + t;
      for (let i = 0; i < c; i += 2) {
        const freq = 1 / 10000 ** (i / c);
        data[t * c + i] = Math.sin(position * freq);
        if (i + 1 < c) data[t * c + i + 1] = Math.cos(position * freq);
      }
    }
    return new Tensor(data, [seq, c], false);
  }

  /**
   * Forward pass.
   * `ids` is flat [batch * seq] token ids, `batch` and `seq` give the shape.
   */
  forward(ids: Int32Array, batch: number, seq: number, options: ForwardOptions = {}): ForwardResult {
    const { training = false, returnHidden = false, rng } = options;
    const c = this.config.dModel;
    if (ids.length !== batch * seq) {
      throw new AlphaValidationError(
        "model",
        `forward: expected ${batch * seq} token ids, received ${ids.length}`,
      );
    }
    if (seq > this.config.contextLength) {
      throw new AlphaValidationError(
        "model",
        `forward: sequence length ${seq} exceeds context length ${this.config.contextLength}`,
      );
    }

    // Token + positional signal.
    let x = reshape(gatherRows(this.tokEmbedding, ids), [batch, seq, c]);
    if (this.posEmbedding) {
      const posIdx = new Int32Array(batch * seq);
      for (let b = 0; b < batch; b++) {
        for (let t = 0; t < seq; t++) posIdx[b * seq + t] = t;
      }
      x = add(x, reshape(gatherRows(this.posEmbedding, posIdx), [batch, seq, c]));
    } else {
      const table = this.sinusoidalPositions(seq);
      x = add(x, reshape(table, [seq, c]));
    }

    const nHeads = this.config.nHeads;
    const headDim = c / nHeads;
    const attnScale = 1 / Math.sqrt(headDim);
    const p = this.config.dropout;

    for (const layer of this.layers) {
      // --- attention block ---
      const normed = layerNorm(x, layer.norm1_w, layer.norm1_b, this.config.normEps);
      const q = splitHeads(linear(normed, layer.attn_wq, layer.attn_bq), nHeads);
      const k = splitHeads(linear(normed, layer.attn_wk, layer.attn_bk), nHeads);
      const v = splitHeads(linear(normed, layer.attn_wv, layer.attn_bv), nHeads);
      const scores = scale(matmul(q, transposeLastTwo(k)), attnScale);
      const probs = causalSoftmax(scores);
      const context = mergeHeads(matmul(probs, v), batch, seq, nHeads);
      const attended = linear(context, layer.attn_wo, layer.attn_bo);
      const attnOut = training && rng ? dropout(attended, p, () => rng.next(), true) : attended;
      x = add(x, attnOut);

      // --- feed-forward block ---
      const normed2 = layerNorm(x, layer.norm2_w, layer.norm2_b, this.config.normEps);
      const hidden = gelu(linear(normed2, layer.mlp_w1, layer.mlp_b1));
      const projected = linear(hidden, layer.mlp_w2, layer.mlp_b2);
      const ffOut = training && rng ? dropout(projected, p, () => rng.next(), true) : projected;
      x = add(x, ffOut);
    }

    const finalHidden = layerNorm(x, this.finalNormW, this.finalNormB, this.config.normEps);
    const logits3 = linear(finalHidden, this.headWeight(), this.outputBias);
    const logits = reshape(logits3, [batch * seq, this.config.vocabSize]);
    return { logits, batch, seq, hidden: returnHidden ? finalHidden : null };
  }

  /**
   * Cached forward pass, used by the inference engine.
   *
   * `incremental: false` primes the cache: every prompt position is computed
   * and its keys/values stored, and the logits of the final position are
   * returned. `incremental: true` computes only the given ids (normally one),
   * appends their keys/values, and attends against everything cached — so the
   * prefix is never recomputed.
   *
   * This path records no autograd graph. It is inference-only, and the token
   * sequence it produces is verified to match `forward` exactly.
   */
  forwardCached(
    ids: Int32Array,
    batch: number,
    seq: number,
    options: CachedForwardOptions,
  ): ForwardResult {
    const { cache, incremental } = options;
    if (batch !== 1) {
      throw new AlphaValidationError(
        "model",
        "forwardCached: the KV cache serves one sequence at a time (batch must be 1)",
      );
    }
    if (incremental && seq > 1) {
      throw new AlphaValidationError(
        "model",
        "forwardCached: an incremental step processes one position at a time",
      );
    }
    const c = this.config.dModel;
    const nHeads = this.config.nHeads;
    const headDim = c / nHeads;
    const attnScale = 1 / Math.sqrt(headDim);

    // The position index this call starts at, which for an incremental step is
    // the number of positions already cached.
    const startPosition = incremental ? cache.length : 0;
    const totalPositions = startPosition + seq;
    if (totalPositions > this.config.contextLength) {
      throw new AlphaValidationError(
        "model",
        `forwardCached: ${totalPositions} positions exceed the model context length ${this.config.contextLength}`,
      );
    }

    let x = this.embedPositions(ids, startPosition, batch, seq);
    const vocab = this.config.vocabSize;

    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const normed = layerNorm(x, layer.norm1_w, layer.norm1_b, this.config.normEps);
      // Q is only needed for the new positions; K and V are needed for the new
      // positions and then kept.
      const q = this.project(normed, layer.attn_wq, layer.attn_bq);
      const k = this.project(normed, layer.attn_wk, layer.attn_bk);
      const v = this.project(normed, layer.attn_wv, layer.attn_bv);

      // Write this layer's K/V for the new positions before attending over
      // them, so the row for the current position is already in the cache.
      for (let t = 0; t < seq; t++) {
        const offset = t * c;
        kvCacheWriteLayer(
          cache,
          l,
          startPosition + t,
          k.data.subarray(offset, offset + c),
          v.data.subarray(offset, offset + c),
        );
      }
      const attended = this.attendAgainstCache(
        q,
        cache.layers[l],
        seq,
        startPosition,
        attnScale,
      );
      const attnOut = this.project(attended, layer.attn_wo, layer.attn_bo);
      x = add(x, reshape(attnOut, [batch, seq, c]));

      const normed2 = layerNorm(x, layer.norm2_w, layer.norm2_b, this.config.normEps);
      const hidden = gelu(linear(normed2, layer.mlp_w1, layer.mlp_b1));
      const ffOut = linear(hidden, layer.mlp_w2, layer.mlp_b2);
      x = add(x, reshape(ffOut, [batch, seq, c]));
    }
    // Every layer now holds the new positions, so the length advances once.
    kvCacheCommit(cache, seq, incremental ? startPosition : 0);

    const finalHidden = layerNorm(x, this.finalNormW, this.finalNormB, this.config.normEps);
    const logits3 = linear(finalHidden, this.headWeight(), this.outputBias);
    const logits = reshape(logits3, [batch * seq, vocab]);
    return { logits, batch, seq, hidden: finalHidden };
  }

  /** Token + positional signal, with the positional index supplied by the caller. */
  private embedPositions(ids: Int32Array, startPosition: number, batch: number, seq: number): Tensor {
    const c = this.config.dModel;
    let x = reshape(gatherRows(this.tokEmbedding, ids), [batch, seq, c]);
    if (this.posEmbedding) {
      const posIdx = new Int32Array(batch * seq);
      for (let b = 0; b < batch; b++) {
        for (let t = 0; t < seq; t++) posIdx[b * seq + t] = startPosition + t;
      }
      x = add(x, reshape(gatherRows(this.posEmbedding, posIdx), [batch, seq, c]));
    } else {
      const table = this.sinusoidalPositionsFrom(startPosition, seq);
      x = add(x, reshape(table, [seq, c]));
    }
    return x;
  }

  /** A plain matmul projection returning [seq, outFeatures] with no bias folding. */
  private project(x: Tensor, weight: Tensor, bias: Tensor | null): Tensor {
    return linear(x, weight, bias);
  }

  /**
   * Attention for the new positions against every cached position.
   *
   * Causality is implicit: the cache only ever holds positions at or before the
   * current one, so no mask is needed. Scores are computed in the same order as
   * the batched path so the arithmetic — and therefore the chosen token —
   * matches the uncached implementation.
   */
  /**
   * Attention for the new positions against every cached position.
   *
   * Causality is implicit: the cache only ever holds positions at or before the
   * current one, so no mask is needed. The accumulation order and precision
   * deliberately match `matmul` in `core/tensor` (float32 accumulation, same
   * index order), so the cached path selects the same token as the batched one
   * rather than merely a close one.
   */
  private attendAgainstCache(
    q: Tensor,
    layer: { keys: Float32Array; values: Float32Array },
    seq: number,
    startPosition: number,
    attnScale: number,
  ): Tensor {
    const c = this.config.dModel;
    const nHeads = this.config.nHeads;
    const headDim = c / nHeads;
    // Rows are laid out [position, dModel]; read the cached rows directly.
    const context = new Float32Array(seq * c);

    for (let h = 0; h < nHeads; h++) {
      const headOffset = h * headDim;
      for (let t = 0; t < seq; t++) {
        const qRow = t * c + headOffset;
        // Causality: this position may attend to itself and everything before
        // it, never after. When priming a whole prompt the later positions are
        // already in the cache, so the bound has to be applied explicitly.
        const visible = startPosition + t + 1;
        // scores[j] = dot(q_head, k[j]) * scale, accumulated in float32 in the
        // same order `matmul` would.
        const scores = new Float32Array(visible);
        for (let d = 0; d < headDim; d++) {
          const qv = q.data[qRow + d];
          if (qv === 0) continue;
          for (let j = 0; j < visible; j++) {
            scores[j] += qv * layer.keys[j * c + headOffset + d];
          }
        }
        let max = -Infinity;
        for (let j = 0; j < visible; j++) {
          scores[j] *= attnScale;
          if (scores[j] > max) max = scores[j];
        }
        let sum = 0;
        for (let j = 0; j < visible; j++) {
          const e = Math.exp(scores[j] - max);
          scores[j] = e;
          sum += e;
        }
        const inv = sum > 0 ? 1 / sum : 0;
        for (let j = 0; j < visible; j++) scores[j] *= inv;
        // context = probs @ values, again matching matmul's order and precision.
        for (let j = 0; j < visible; j++) {
          const p = scores[j];
          if (p === 0) continue;
          for (let d = 0; d < headDim; d++) {
            context[t * c + headOffset + d] += p * layer.values[j * c + headOffset + d];
          }
        }
      }
    }
    return new Tensor(context, [1, seq, c], false);
  }

  /**
   * Mean-pooled hidden states — the substrate for Alpha Embeddings. It is the
   * model's own representation; quality depends on how well Alpha was trained,
   * which is why the embeddings module reports the model stage alongside it.
   */
  embedHidden(ids: Int32Array, batch: number, seq: number): Float32Array {
    const hidden = this.forward(ids, batch, seq, { training: false, returnHidden: true }).hidden;
    if (!hidden) {
      throw new AlphaValidationError("model", "embedHidden: hidden state unavailable");
    }
    const c = this.config.dModel;
    const out = new Float32Array(batch * c);
    for (let b = 0; b < batch; b++) {
      for (let t = 0; t < seq; t++) {
        const off = (b * seq + t) * c;
        for (let i = 0; i < c; i++) out[b * c + i] += hidden.data[off + i] / seq;
      }
    }
    return out;
  }

  // -------------------------------------------------------------------------
  // Weight (de)serialisation — the basis of checkpoints
  // -------------------------------------------------------------------------

  serializeWeights(): SerializedWeights {
    const tensors: Record<string, string> = {};
    const shapes: Record<string, Shape> = {};
    for (const { name, tensor } of this.parameterList) {
      tensors[name] = float32ToBase64(tensor.data);
      shapes[name] = [...tensor.shape];
    }
    return { config: this.config, tensors, shapes };
  }

  loadWeights(payload: SerializedWeights): void {
    const target = this.parameterMap();
    for (const [name, encoded] of Object.entries(payload.tensors)) {
      const tensor = target.get(name);
      if (!tensor) continue; // architecture changed; extra tensors are ignored
      const values = base64ToFloat32(encoded);
      if (values.length !== tensor.size) continue;
      tensor.data.set(values);
    }
  }

  /** Copy weights into a fresh, gradient-free snapshot (used by inference). */
  snapshot(): AlphaTransformer {
    const clone = new AlphaTransformer(this.config, 1);
    const cloneParams = clone.parameterMap();
    for (const { name, tensor } of this.parameterList) {
      const dest = cloneParams.get(name);
      if (dest) dest.data.set(tensor.data);
    }
    clone.disableGrad();
    return clone;
  }

  disableGrad(): void {
    for (const { tensor } of this.parameterList) {
      tensor.requiresGrad = false;
      tensor.grad = null;
      tensor.parents = [];
      tensor.backwardFn = null;
    }
  }
}
