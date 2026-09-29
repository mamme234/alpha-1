/**
 * Alpha Embeddings — Alpha's own vector representation.
 *
 * There is no embedding API here. A text is tokenised, run through Alpha's
 * transformer, and the final hidden states are pooled and L2-normalised. The
 * dimension is whatever the model width is, and the quality is whatever the
 * model has actually learned: an untrained model produces vectors whose
 * geometry is meaningless, and the embedder reports its stage alongside them so
 * callers never mistake one for the other.
 */

import { setGradEnabled } from "../core/tensor";
import type { AlphaModelStage } from "../core/types";
import type { AlphaTransformer } from "../model/transformer";
import type { AlphaTokenizer } from "../tokenizer/bpe";

export type PoolingStrategy = "mean" | "last-token";

export type EmbeddingRecord = {
  /** The embedded text (or the original text when embedding a document). */
  text: string;
  vector: number[];
  dimension: number;
  tokens: number;
  truncated: boolean;
  modelStage: AlphaModelStage;
};

export type AlphaEmbedderOptions = {
  model: AlphaTransformer;
  tokenizer: AlphaTokenizer;
  stage?: AlphaModelStage;
  pooling?: PoolingStrategy;
  /** Longest text accepted; longer inputs are truncated from the left. */
  maxTokens?: number;
};

export type SimilarityFn = (a: number[], b: number[]) => number;

export class AlphaEmbedder {
  readonly model: AlphaTransformer;
  readonly tokenizer: AlphaTokenizer;
  readonly stage: AlphaModelStage;
  readonly pooling: PoolingStrategy;
  readonly maxTokens: number;

  constructor(options: AlphaEmbedderOptions) {
    this.model = options.model;
    this.tokenizer = options.tokenizer;
    this.stage = options.stage ?? "untrained";
    this.pooling = options.pooling ?? "mean";
    this.maxTokens = Math.min(
      options.maxTokens ?? options.model.config.contextLength,
      options.model.config.contextLength,
    );
  }

  get dimension(): number {
    return this.model.config.dModel;
  }

  /**
   * Embed a single text. Documents and queries share one encoder — Alpha has no
   * separate retrieval encoder, and pretending otherwise would be a fiction.
   */
  embed(text: string): EmbeddingRecord {
    const encoded = this.tokenizer.encodeDetailed(text, {
      maxLength: this.maxTokens,
      truncation: "right",
      addBos: true,
    });
    const ids = Int32Array.from(encoded.ids.length ? encoded.ids : [this.tokenizer.bosId]);
    setGradEnabled(false);
    let vector: number[];
    try {
      const hidden = this.model.embedHidden(ids, 1, ids.length);
      const dimension = this.model.config.dModel;
      if (this.pooling === "last-token") {
        const offset = (ids.length - 1) * dimension;
        vector = Array.from(hidden.subarray(offset, offset + dimension));
      } else {
        vector = Array.from(hidden.subarray(0, dimension));
      }
    } finally {
      setGradEnabled(true);
    }
    return {
      text,
      vector: normalize(vector),
      dimension: vector.length,
      tokens: ids.length,
      truncated: encoded.truncated,
      modelStage: this.stage,
    };
  }

  /** Embed many texts (batch of one keeps memory bounded and code simple). */
  embedDocuments(texts: string[]): EmbeddingRecord[] {
    return texts.map((text) => this.embed(text));
  }

  /** Query embedding — same encoder, documented so callers do not assume a prefix. */
  embedQuery(text: string): EmbeddingRecord {
    return this.embed(text);
  }

  /** Plain vectors, for callers that only need the numbers. */
  vectors(texts: string[]): number[][] {
    return this.embedDocuments(texts).map((record) => record.vector);
  }
}

/** L2-normalise a vector; a zero vector is returned unchanged. */
export function normalize(vector: number[]): number[] {
  let sum = 0;
  for (const value of vector) sum += value * value;
  const norm = Math.sqrt(sum);
  if (norm === 0) return [...vector];
  return vector.map((value) => value / norm);
}

/** Cosine similarity. Inputs are normalised by the embedder, so this is a dot product. */
export function cosineSimilarity(a: number[], b: number[]): number {
  const length = Math.min(a.length, b.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export function dotProduct(a: number[], b: number[]): number {
  const length = Math.min(a.length, b.length);
  let dot = 0;
  for (let i = 0; i < length; i++) dot += a[i] * b[i];
  return dot;
}

export function euclideanDistance(a: number[], b: number[]): number {
  const length = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

export const SIMILARITY_FUNCTIONS: Record<"cosine" | "dot" | "euclidean", SimilarityFn> = {
  cosine: cosineSimilarity,
  dot: dotProduct,
  euclidean: euclideanDistance,
};

/** Centroid of a set of vectors — used for coherence diagnostics. */
export function centroid(vectors: number[][]): number[] {
  if (vectors.length === 0) return [];
  const dimension = vectors[0].length;
  const out = new Array<number>(dimension).fill(0);
  for (const vector of vectors) {
    for (let i = 0; i < dimension; i++) out[i] += vector[i];
  }
  return normalize(out.map((value) => value / vectors.length));
}
