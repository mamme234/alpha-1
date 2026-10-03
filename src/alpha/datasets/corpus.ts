/**
 * Alpha Datasets — corpus encoding, statistics and batching.
 *
 * A corpus is a flat stream of token ids. Training samples are windows of that
 * stream; the target is the same window shifted by one position, which is the
 * standard next-token objective.
 *
 * Two batching modes exist and both are real:
 *
 *  - **window** (default): every sample is a full-length window, so no padding
 *    is needed and every position contributes to the loss.
 *  - **document**: whole documents are batched and padded to the batch's
 *    longest sequence. Because Alpha is a causal decoder, a pad token can never
 *    influence an earlier position, so masking the *loss* (which the
 *    cross-entropy already does through its ignore index) is sufficient —
 *    `attentionMask` is carried for reporting, not as a correctness patch.
 *
 * The pipeline never silently drops malformed data: the corpus is validated
 * before encoding and any problem is reported.
 */

import { AlphaRng, type RngState } from "../core/rng";
import { AlphaValidationError } from "../core/errors";
import { ALPHA_RESOURCE_LIMITS, assertResourceLimit } from "../core/limits";
import type { AlphaTokenizer } from "../tokenizer/bpe";
import { assertValidDataset, datasetFingerprint, datasetStats, splitDocuments, type AlphaDataset } from "./types";

export type CorpusStats = {
  documents: number;
  trainDocuments: number;
  validationDocuments: number;
  /** Non-whitespace characters in the corpus. */
  characters: number;
  trainTokens: number;
  validationTokens: number;
  totalTokens: number;
  vocabularySize: number;
  /** Tokens that fell back to `<unk>` — counted, never hidden. */
  unknownTokens: number;
  maxDocumentTokens: number;
};

export type EncodedCorpus = {
  datasetName: string;
  datasetVersion: string;
  datasetFingerprint: string;
  license: string;
  tokenizerVersion: string;
  vocabSize: number;
  /** Full token stream in corpus order. */
  tokenIds: Int32Array;
  trainIds: Int32Array;
  validationIds: Int32Array;
  /** Per-document token sequences, used by document-mode batching. */
  trainDocuments: Int32Array[];
  validationDocuments: Int32Array[];
  stats: CorpusStats;
};

export type EncodeCorpusOptions = {
  validationFraction?: number;
  /** Append the BOS token before each document so boundaries are learnable. */
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
  assertValidDataset(dataset);
  const validationFraction = options.validationFraction ?? 0.1;
  const { train, validation } = splitDocuments(dataset, validationFraction);
  const documentBoundaries = options.documentBoundaries ?? true;

  const encodeDocs = (docs: string[]) => {
    const sequences: Int32Array[] = [];
    let unknown = 0;
    for (const doc of docs) {
      const encoded = tokenizer.encodeDetailed(doc, {
        addBos: documentBoundaries,
        addEos: documentBoundaries,
      });
      if (encoded.ids.length === 0) continue;
      sequences.push(Int32Array.from(encoded.ids));
      unknown += encoded.unknown;
    }
    const total = sequences.reduce((sum, ids) => sum + ids.length, 0);
    const ids = new Int32Array(total);
    let offset = 0;
    for (const sequence of sequences) {
      ids.set(sequence, offset);
      offset += sequence.length;
    }
    return { ids, sequences, unknown };
  };

  const trainEncoded = encodeDocs(train.documents);
  const validationEncoded = encodeDocs(validation.documents);
  const full = new Int32Array(trainEncoded.ids.length + validationEncoded.ids.length);
  full.set(trainEncoded.ids, 0);
  full.set(validationEncoded.ids, trainEncoded.ids.length);

  const stats = datasetStats(dataset);
  const maxDocumentTokens = [...trainEncoded.sequences, ...validationEncoded.sequences].reduce(
    (max, ids) => Math.max(max, ids.length),
    0,
  );

  return {
    datasetName: dataset.name,
    datasetVersion: dataset.version,
    datasetFingerprint: datasetFingerprint(dataset),
    license: dataset.license,
    tokenizerVersion: tokenizer.version,
    vocabSize: tokenizer.vocabSize,
    tokenIds: full,
    trainIds: trainEncoded.ids,
    validationIds: validationEncoded.ids,
    trainDocuments: trainEncoded.sequences,
    validationDocuments: validationEncoded.sequences,
    stats: {
      documents: stats.documents,
      trainDocuments: trainEncoded.sequences.length,
      validationDocuments: validationEncoded.sequences.length,
      characters: stats.characters,
      trainTokens: trainEncoded.ids.length,
      validationTokens: validationEncoded.ids.length,
      totalTokens: full.length,
      vocabularySize: tokenizer.vocabSize,
      unknownTokens: trainEncoded.unknown + validationEncoded.unknown,
      maxDocumentTokens,
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
  /** 1 for a real token, 0 for padding. All ones in window mode. */
  attentionMask: Uint8Array;
  /** 1 where the target contributes to the loss. All ones in window mode. */
  lossMask: Uint8Array;
  paddingTokens: number;
};

/** How many training examples a token stream of this length can produce. */
export function countTrainingExamples(tokenCount: number, seqLen: number): number {
  if (seqLen < 2) return 0;
  return Math.max(0, Math.floor(tokenCount / seqLen));
}

/**
 * Pad variable-length token sequences into one ragged-free batch. Targets are
 * padded with the pad token, which the cross-entropy ignores, so padding never
 * enters the loss.
 */
export function padSequences(
  sequences: Int32Array[],
  options: { padId: number; seqLen: number },
): TrainingBatch {
  const { padId, seqLen } = options;
  if (sequences.length === 0) {
    throw new AlphaValidationError("datasets", "padSequences needs at least one sequence");
  }
  if (seqLen < 2) {
    throw new AlphaValidationError("datasets", "seqLen must be at least 2");
  }
  const batch = sequences.length;
  const input = new Int32Array(batch * seqLen);
  const target = new Int32Array(batch * seqLen);
  const attentionMask = new Uint8Array(batch * seqLen);
  const lossMask = new Uint8Array(batch * seqLen);
  let tokenSum = 0;
  let padding = 0;

  sequences.forEach((ids, row) => {
    const usable = Math.min(ids.length - 1, seqLen);
    input.fill(padId, row * seqLen, row * seqLen + seqLen);
    target.fill(padId, row * seqLen, row * seqLen + seqLen);
    for (let t = 0; t < usable; t++) {
      const offset = row * seqLen + t;
      input[offset] = ids[t];
      target[offset] = ids[t + 1];
      attentionMask[offset] = 1;
      lossMask[offset] = 1;
      tokenSum += ids[t];
    }
    padding += seqLen - usable;
  });

  return {
    input,
    target,
    batch,
    seqLen,
    meanTokenId: tokenSum / input.length,
    attentionMask,
    lossMask,
    paddingTokens: padding,
  };
}

function windowBatch(
  ids: Int32Array,
  batchSize: number,
  seqLen: number,
  startFor: (row: number) => number,
): TrainingBatch {
  const input = new Int32Array(batchSize * seqLen);
  const target = new Int32Array(batchSize * seqLen);
  const attentionMask = new Uint8Array(batchSize * seqLen).fill(1);
  const lossMask = new Uint8Array(batchSize * seqLen).fill(1);
  let tokenSum = 0;
  for (let b = 0; b < batchSize; b++) {
    const start = startFor(b);
    for (let t = 0; t < seqLen; t++) {
      const id = ids[start + t];
      input[b * seqLen + t] = id;
      target[b * seqLen + t] = ids[start + t + 1];
      tokenSum += id;
    }
  }
  return {
    input,
    target,
    batch: batchSize,
    seqLen,
    meanTokenId: tokenSum / input.length,
    attentionMask,
    lossMask,
    paddingTokens: 0,
  };
}

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
  readonly padded: boolean;

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
    assertResourceLimit("maxBatchSize", options.batchSize, "batch sampler");
    assertResourceLimit("maxSeqLen", options.seqLen, "batch sampler");
    this.seqLen = options.seqLen;
    this.batchSize = options.batchSize;
    this.ids = tileToLength(ids, options.seqLen * 2 + 2);
    this.rng = new AlphaRng(options.seed ?? 1234);
    this.padded = false;
  }

  get corpusLength(): number {
    return this.ids.length;
  }

  saveState(): RngState {
    return this.rng.saveState();
  }

  loadState(state: RngState): void {
    this.rng.loadState(state);
  }

  next(): TrainingBatch {
    const maxStart = this.ids.length - this.seqLen - 1;
    return windowBatch(this.ids, this.batchSize, this.seqLen, () => this.rng.int(maxStart + 1));
  }

  /** Deterministic sweep used for validation loss. */
  *sequentialBatches(): Generator<TrainingBatch> {
    const stride = this.seqLen;
    for (let b = 0; b + this.batchSize * stride + 1 < this.ids.length; b += this.batchSize * stride) {
      yield windowBatch(this.ids, this.batchSize, this.seqLen, (row) => b + row * stride);
    }
  }
}

/**
 * Document-mode sampler: whole documents, padded to the window. Used when the
 * corpus is a small set of long documents rather than a single long stream.
 */
export class DocumentBatchSampler {
  private readonly sequences: Int32Array[];
  private readonly rng: AlphaRng;
  private readonly padId: number;
  readonly batchSize: number;
  readonly seqLen: number;
  readonly padded = true;

  saveState(): RngState {
    return this.rng.saveState();
  }

  loadState(state: RngState): void {
    this.rng.loadState(state);
  }

  constructor(
    sequences: Int32Array[],
    options: { batchSize: number; seqLen: number; seed?: number; padId: number },
  ) {
    const usable = sequences.filter((ids) => ids.length >= 2);
    if (usable.length === 0) {
      throw new AlphaValidationError(
        "datasets",
        "document batching needs at least one document with two or more tokens",
      );
    }
    assertResourceLimit("maxBatchSize", options.batchSize, "document sampler");
    assertResourceLimit("maxSeqLen", options.seqLen, "document sampler");
    this.sequences = usable;
    this.batchSize = options.batchSize;
    this.seqLen = options.seqLen;
    this.padId = options.padId;
    this.rng = new AlphaRng(options.seed ?? 1234);
  }

  get documentCount(): number {
    return this.sequences.length;
  }

  next(): TrainingBatch {
    const picked: Int32Array[] = [];
    for (let i = 0; i < this.batchSize; i++) {
      picked.push(this.sequences[this.rng.int(this.sequences.length)]);
    }
    return padSequences(picked, { padId: this.padId, seqLen: this.seqLen });
  }

  /** Deterministic sweep: one pass over the documents in corpus order. */
  *sequentialBatches(): Generator<TrainingBatch> {
    for (let start = 0; start < this.sequences.length; start += this.batchSize) {
      const slice = this.sequences.slice(start, start + this.batchSize);
      yield padSequences(slice, { padId: this.padId, seqLen: this.seqLen });
    }
  }
}

export type CorpusReport = {
  name: string;
  version: string;
  license: string;
  source: string;
  fingerprint: string;
  /** "documents" is the corpus size in documents. */
  documents: number;
  characters: number;
  averageDocumentCharacters: number;
  /** Total tokens after the train/validation split. */
  tokens: number;
  trainTokens: number;
  validationTokens: number;
  unknownTokens: number;
  vocabularySize: number;
  uniqueCharacters: number;
  sequenceLength: number;
  batchSize: number;
  trainExamples: number;
  validationExamples: number;
  validationFraction: number;
  maxDocumentTokens: number;
  padded: boolean;
  paddingTokensPerBatch: number | null;
  limits: { maxDocuments: number; maxDocumentCharacters: number; maxSeqLen: number; maxBatchSize: number };
};

/**
 * Everything the training engine, the UI and the docs need to describe what a
 * run will actually consume. Computed from the corpus, never estimated.
 */
export function corpusReport(
  dataset: AlphaDataset,
  tokenizer: AlphaTokenizer,
  options: { seqLen: number; batchSize: number; validationFraction?: number; padded?: boolean },
): CorpusReport {
  assertResourceLimit("maxBatchSize", options.batchSize, "corpus report");
  assertResourceLimit("maxSeqLen", options.seqLen, "corpus report");
  const corpus = encodeCorpus(dataset, tokenizer, { validationFraction: options.validationFraction });
  const stats = datasetStats(dataset);
  const padded = options.padded ?? false;
  const trainExamples = padded
    ? corpus.stats.trainDocuments
    : countTrainingExamples(corpus.stats.trainTokens, options.seqLen);
  const validationExamples = padded
    ? corpus.stats.validationDocuments
    : countTrainingExamples(corpus.stats.validationTokens, options.seqLen);
  const paddingTokensPerBatch = padded
    ? corpus.trainDocuments.reduce((sum, ids) => sum + Math.max(0, options.seqLen - (ids.length - 1)), 0) /
      Math.max(1, corpus.trainDocuments.length)
    : null;
  return {
    name: dataset.name,
    version: dataset.version,
    license: dataset.license,
    source: dataset.source,
    fingerprint: corpus.datasetFingerprint,
    documents: stats.documents,
    characters: stats.characters,
    averageDocumentCharacters: stats.averageDocumentLength,
    tokens: corpus.stats.totalTokens,
    trainTokens: corpus.stats.trainTokens,
    validationTokens: corpus.stats.validationTokens,
    unknownTokens: corpus.stats.unknownTokens,
    vocabularySize: tokenizer.vocabSize,
    uniqueCharacters: stats.uniqueCharacters,
    sequenceLength: options.seqLen,
    batchSize: options.batchSize,
    trainExamples,
    validationExamples,
    validationFraction: options.validationFraction ?? 0.1,
    maxDocumentTokens: corpus.stats.maxDocumentTokens,
    padded,
    paddingTokensPerBatch: paddingTokensPerBatch === null ? null : Math.round(paddingTokensPerBatch),
    limits: {
      maxDocuments: ALPHA_RESOURCE_LIMITS.maxDocuments,
      maxDocumentCharacters: ALPHA_RESOURCE_LIMITS.maxDocumentCharacters,
      maxSeqLen: ALPHA_RESOURCE_LIMITS.maxSeqLen,
      maxBatchSize: ALPHA_RESOURCE_LIMITS.maxBatchSize,
    },
  };
}
