/**
 * Alpha Datasets — corpus encoding and batching.
 *
 * A corpus is a flat stream of token ids. Training samples are windows of that
 * stream; the target is the same window shifted by one position, which is the
 * standard next-token objective.
 */

import { AlphaRng } from "../core/rng";
import { AlphaValidationError } from "../core/errors";
import type { AlphaTokenizer } from "../tokenizer/bpe";
import { type AlphaDataset, datasetStats, splitDocuments } from "./types";

export type EncodedCorpus = {
  datasetName: string;
  datasetVersion: string;
  license: string;
  tokenizerVersion: string;
  vocabSize: number;
  /** Full token stream in corpus order. */
  tokenIds: Int32Array;
  trainIds: Int32Array;
  validationIds: Int32Array;
  stats: {
    documents: number;
    trainTokens: number;
    validationTokens: number;
    characters: number;
    unknownTokens: number;
  };
};

export type EncodeCorpusOptions = {
  validationFraction?: number;
  /** Append the BOS token between documents so boundaries are learnable. */
  documentBoundaries?: boolean;
};

/** Tile a token stream until it is long enough to sample windows from. */
function tileToLength(ids: Int32Array, minimum: number): Int32Array {
  if (ids.length >= minimum) return ids;
  const repeats = Math.ceil(minimum / Math.max(ids.length, 1));
  const out = new Int32Array(ids.length * repeats);
  for (let r = 0; r < repeats; r++) out.set(ids, r * ids.length);
  return out;
}

export function encodeCorpus(
  dataset: AlphaDataset,
  tokenizer: AlphaTokenizer,
  options: EncodeCorpusOptions = {},
): EncodedCorpus {
  const validationFraction = options.validationFraction ?? 0.1;
  const { train, validation } = splitDocuments(dataset, validationFraction);
  const encodeDocs = (docs: string[]) => {
    const ids: number[] = [];
    let unknown = 0;
    for (const doc of docs) {
      const encoded = tokenizer.encodeDetailed(doc, {
        addBos: options.documentBoundaries ?? true,
        addEos: options.documentBoundaries ?? true,
      });
      ids.push(...encoded.ids);
      unknown += encoded.unknown;
    }
    return { ids: Int32Array.from(ids), unknown };
  };
  const trainEncoded = encodeDocs(train.documents);
  const validationEncoded = encodeDocs(validation.documents);
  const full = new Int32Array(trainEncoded.ids.length + validationEncoded.ids.length);
  full.set(trainEncoded.ids, 0);
  full.set(validationEncoded.ids, trainEncoded.ids.length);

  const stats = datasetStats(dataset);
  return {
    datasetName: dataset.name,
    datasetVersion: dataset.version,
    license: dataset.license,
    tokenizerVersion: tokenizer.version,
    vocabSize: tokenizer.vocabSize,
    tokenIds: full,
    trainIds: trainEncoded.ids,
    validationIds: validationEncoded.ids,
    stats: {
      documents: stats.documents,
      trainTokens: trainEncoded.ids.length,
      validationTokens: validationEncoded.ids.length,
      characters: stats.characters,
      unknownTokens: trainEncoded.unknown + validationEncoded.unknown,
    },
  };
}

export type TrainingBatch = {
  input: Int32Array;
  target: Int32Array;
  batch: number;
  seqLen: number;
  /** Mean input token id — a cheap sanity signal that batches look sane. */
  meanTokenId: number;
};

/**
 * Random-window sampler over a token stream.
 *
 * Windows are drawn uniformly and can overlap: with a corpus of a few hundred
 * thousand tokens, the risk of losing generalisation to leakage is negligible,
 * and it keeps the sampler simple and reproducible.
 */
export class BatchSampler {
  private readonly ids: Int32Array;
  private readonly rng: AlphaRng;
  readonly batchSize: number;
  readonly seqLen: number;

  constructor(
    ids: Int32Array,
    options: { batchSize: number; seqLen: number; seed?: number },
  ) {
    if (options.seqLen < 2) {
      throw new AlphaValidationError("datasets", "seqLen must be at least 2");
    }
    if (options.batchSize < 1) {
      throw new AlphaValidationError("datasets", "batchSize must be at least 1");
    }
    this.seqLen = options.seqLen;
    this.batchSize = options.batchSize;
    this.ids = tileToLength(ids, options.seqLen * 2 + 2);
    this.rng = new AlphaRng(options.seed ?? 1234);
  }

  get corpusLength(): number {
    return this.ids.length;
  }

  next(): TrainingBatch {
    const input = new Int32Array(this.batchSize * this.seqLen);
    const target = new Int32Array(this.batchSize * this.seqLen);
    const maxStart = this.ids.length - this.seqLen - 1;
    let tokenSum = 0;
    for (let b = 0; b < this.batchSize; b++) {
      const start = this.rng.int(maxStart + 1);
      for (let t = 0; t < this.seqLen; t++) {
        const id = this.ids[start + t];
        input[b * this.seqLen + t] = id;
        target[b * this.seqLen + t] = this.ids[start + t + 1];
        tokenSum += id;
      }
    }
    return {
      input,
      target,
      batch: this.batchSize,
      seqLen: this.seqLen,
      meanTokenId: tokenSum / input.length,
    };
  }

  /** Deterministic sweep used for validation loss. */
  *sequentialBatches(): Generator<TrainingBatch> {
    const stride = this.seqLen;
    for (let b = 0; b + this.batchSize * stride + 1 < this.ids.length; b += this.batchSize * stride) {
      const input = new Int32Array(this.batchSize * this.seqLen);
      const target = new Int32Array(this.batchSize * this.seqLen);
      let tokenSum = 0;
      for (let i = 0; i < this.batchSize; i++) {
        const start = b + i * stride;
        for (let t = 0; t < this.seqLen; t++) {
          const id = this.ids[start + t];
          input[i * this.seqLen + t] = id;
          target[i * this.seqLen + t] = this.ids[start + t + 1];
          tokenSum += id;
        }
      }
      if (this.batchSize * this.seqLen === 0) return;
      yield {
        input,
        target,
        batch: this.batchSize,
        seqLen: this.seqLen,
        meanTokenId: tokenSum / input.length,
      };
    }
  }
}
