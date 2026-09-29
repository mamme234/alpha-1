/**
 * Alpha Tokenizer — Alpha's own vocabulary.
 *
 * A character-level BPE trained from scratch on a corpus supplied to Alpha
 * (see `src/alpha/datasets`). No pretrained vocabulary is downloaded, and no
 * tokenizer service is called.
 *
 * Pipeline:
 *   text -> pre-tokenize (words / numbers / punctuation / whitespace)
 *        -> character symbols -> BPE merges (learned, ranked)
 *        -> token ids (with special tokens, padding, truncation)
 *   ids  -> token strings -> text   (exact round trip for characters in vocab)
 *
 * Characters outside the trained alphabet map to `<unk>`; the tokenizer
 * reports that explicitly instead of silently corrupting text.
 */

import { AlphaRng } from "../core/rng";
import { AlphaValidationError } from "../core/errors";

export type AlphaSpecialTokens = {
  pad: string;
  unk: string;
  bos: string;
  eos: string;
};

export const DEFAULT_SPECIAL_TOKENS: AlphaSpecialTokens = {
  pad: "<pad>",
  unk: "<unk>",
  bos: "<bos>",
  eos: "<eos>",
};

export type BpeMerge = [string, string];

export type AlphaTokenizerSnapshot = {
  /** Vocabulary version — bump when the merge table or alphabet changes. */
  version: string;
  /** Identifier of the training dataset the vocabulary came from. */
  trainedOn: string;
  merges: BpeMerge[];
  alphabet: string[];
  specialTokens: AlphaSpecialTokens;
  tokens: string[];
  stats: {
    documents: number;
    characters: number;
    mergeSteps: number;
    builtAt: number;
  };
};

export type EncodeOptions = {
  maxLength?: number;
  padding?: "left" | "right";
  truncation?: "left" | "right" | "error";
  addBos?: boolean;
  addEos?: boolean;
  /** Pad to `maxLength` using the pad token. */
  padToMaxLength?: boolean;
};

export type EncodedSequence = {
  ids: number[];
  tokens: string[];
  /** True when truncation dropped content. */
  truncated: boolean;
  /** Number of tokens replaced by `<unk>`. */
  unknown: number;
  attentionMask: number[];
};

export type BpeTrainingOptions = {
  /** Target vocabulary size including special tokens. */
  vocabSize: number;
  specialTokens?: AlphaSpecialTokens;
  version?: string;
  trainedOn?: string;
  /** Stop early if no pair occurs at least this many times. */
  minPairFrequency?: number;
  onProgress?: (info: { merges: number; vocabSize: number }) => void;
};

const PRE_TOKEN_PATTERN = /[A-Za-z]+|[0-9]+|\s+|./gu;

/** Split text into word-like units; whitespace is preserved as its own unit. */
export function preTokenize(text: string): string[] {
  const units = text.match(PRE_TOKEN_PATTERN) ?? [];
  return units;
}

function countPairs(symbols: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (let i = 0; i < symbols.length - 1; i++) {
    const key = `${symbols[i]}\u0000${symbols[i + 1]}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

function mergeSymbols(symbols: string[], left: string, right: string, merged: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < symbols.length; i++) {
    if (i < symbols.length - 1 && symbols[i] === left && symbols[i + 1] === right) {
      out.push(merged);
      i++;
    } else {
      out.push(symbols[i]);
    }
  }
  return out;
}

export class AlphaTokenizer {
  readonly version: string;
  readonly trainedOn: string;
  readonly specialTokens: AlphaSpecialTokens;
  private tokens: string[];
  private tokenToId: Map<string, number>;
  /** merge rank lookup: "left\u0000right" -> rank */
  private mergeRanks: Map<string, number>;
  private merges: BpeMerge[];
  readonly alphabet: string[];
  readonly stats: AlphaTokenizerSnapshot["stats"];

  constructor(snapshot: AlphaTokenizerSnapshot) {
    this.version = snapshot.version;
    this.trainedOn = snapshot.trainedOn;
    this.specialTokens = snapshot.specialTokens;
    this.tokens = [...snapshot.tokens];
    this.merges = snapshot.merges.map((m) => [m[0], m[1]] as BpeMerge);
    this.alphabet = [...snapshot.alphabet];
    this.stats = snapshot.stats;
    this.tokenToId = new Map(this.tokens.map((t, i) => [t, i]));
    this.mergeRanks = new Map();
    this.merges.forEach(([left, right], index) => {
      this.mergeRanks.set(`${left}\u0000${right}`, index);
    });
  }

  get vocabSize(): number {
    return this.tokens.length;
  }

  get vocabulary(): readonly string[] {
    return this.tokens;
  }

  idForToken(token: string): number {
    return this.tokenToId.get(token) ?? this.idFor(this.specialTokens.unk);
  }

  tokenForId(id: number): string {
    return this.tokens[id] ?? this.specialTokens.unk;
  }

  /** Special-token ids are fixed and stable across vocabulary versions. */
  idFor(special: string): number {
    return this.tokenToId.get(special) ?? 0;
  }

  get padId(): number {
    return this.idFor(this.specialTokens.pad);
  }

  get bosId(): number {
    return this.idFor(this.specialTokens.bos);
  }

  get eosId(): number {
    return this.idFor(this.specialTokens.eos);
  }

  /** BPE-encode one pre-token unit into subword tokens. */
  private encodeUnit(unit: string): { tokens: string[]; unknown: number } {
    let symbols = Array.from(unit);
    let unknown = 0;
    if (symbols.length > 1) {
      // Iteratively apply the lowest-rank adjacent merge available.
      let merged = true;
      while (merged) {
        merged = false;
        let bestRank = Infinity;
        let bestIndex = -1;
        for (let i = 0; i < symbols.length - 1; i++) {
          const rank = this.mergeRanks.get(`${symbols[i]}\u0000${symbols[i + 1]}`);
          if (rank !== undefined && rank < bestRank) {
            bestRank = rank;
            bestIndex = i;
          }
        }
        if (bestIndex >= 0) {
          const left = symbols[bestIndex];
          const right = symbols[bestIndex + 1];
          symbols = [
            ...symbols.slice(0, bestIndex),
            left + right,
            ...symbols.slice(bestIndex + 2),
          ];
          merged = true;
        }
      }
    }
    const out: string[] = [];
    for (const symbol of symbols) {
      if (this.tokenToId.has(symbol)) {
        out.push(symbol);
      } else {
        out.push(this.specialTokens.unk);
        unknown++;
      }
    }
    return { tokens: out, unknown };
  }

  /** text -> token ids. */
  encode(text: string): number[] {
    return this.encodeDetailed(text).ids;
  }

  encodeDetailed(text: string, options: EncodeOptions = {}): EncodedSequence {
    const units = preTokenize(text);
    const tokens: string[] = [];
    let unknown = 0;
    if (options.addBos) tokens.push(this.specialTokens.bos);
    for (const unit of units) {
      const { tokens: encoded, unknown: missed } = this.encodeUnit(unit);
      tokens.push(...encoded);
      unknown += missed;
    }
    if (options.addEos) tokens.push(this.specialTokens.eos);

    let truncated = false;
    const maxLength = options.maxLength;
    if (maxLength !== undefined && tokens.length > maxLength) {
      if (options.truncation === "error") {
        throw new AlphaValidationError(
          "tokenizer",
          `input is ${tokens.length} tokens, longer than maxLength ${maxLength}`,
        );
      }
      if (options.truncation === "left") {
        tokens.splice(0, tokens.length - maxLength);
      } else {
        tokens.length = maxLength;
      }
      truncated = true;
    }

    const ids = tokens.map((t) => this.idForToken(t));
    const attentionMask: number[] = tokens.map((t) => (t === this.specialTokens.pad ? 0 : 1));

    if (options.padToMaxLength && maxLength !== undefined && ids.length < maxLength) {
      const deficit = maxLength - ids.length;
      const padId = this.padId;
      if (options.padding === "left") {
        ids.unshift(...new Array<number>(deficit).fill(padId));
        tokens.unshift(...new Array<string>(deficit).fill(this.specialTokens.pad));
        attentionMask.unshift(...new Array<number>(deficit).fill(0));
      } else {
        ids.push(...new Array<number>(deficit).fill(padId));
        tokens.push(...new Array<string>(deficit).fill(this.specialTokens.pad));
        attentionMask.push(...new Array<number>(deficit).fill(0));
      }
    }

    return { ids, tokens, truncated, unknown, attentionMask };
  }

  /** token ids -> text. Special tokens are skipped, subwords are concatenated. */
  decode(ids: number[] | Int32Array, options: { skipSpecial?: boolean } = {}): string {
    const skipSpecial = options.skipSpecial ?? true;
    const specials = new Set(Object.values(this.specialTokens));
    let out = "";
    for (const id of Array.from(ids)) {
      const token = this.tokenForId(id);
      if (skipSpecial && specials.has(token)) continue;
      out += token;
    }
    return out;
  }

  countTokens(text: string): number {
    return this.encode(text).length;
  }

  toJSON(): AlphaTokenizerSnapshot {
    return {
      version: this.version,
      trainedOn: this.trainedOn,
      merges: this.merges.map((m) => [m[0], m[1]] as BpeMerge),
      alphabet: [...this.alphabet],
      specialTokens: this.specialTokens,
      tokens: [...this.tokens],
      stats: this.stats,
    };
  }

  static fromJSON(snapshot: AlphaTokenizerSnapshot): AlphaTokenizer {
    return new AlphaTokenizer(snapshot);
  }

  /**
   * Train a vocabulary on Alpha's own corpus.
   *
   * This is a real BPE trainer: it counts adjacent symbol pairs across the
   * corpus and greedily merges the most frequent pair until the target
   * vocabulary size is reached. The resulting merge table is stored in the
   * snapshot, so a trained tokenizer is a portable artifact.
   */
  static train(documents: string[], options: BpeTrainingOptions): AlphaTokenizer {
    const specials = options.specialTokens ?? DEFAULT_SPECIAL_TOKENS;
    const specialList = [specials.pad, specials.unk, specials.bos, specials.eos];
    if (options.vocabSize < specialList.length + 2) {
      throw new AlphaValidationError(
        "tokenizer",
        `vocabSize must be at least ${specialList.length + 2}`,
      );
    }
    if (documents.length === 0) {
      throw new AlphaValidationError("tokenizer", "cannot train a vocabulary on an empty corpus");
    }

    const wordCounts = new Map<string, number>();
    let characters = 0;
    for (const doc of documents) {
      characters += doc.length;
      for (const unit of preTokenize(doc)) {
        wordCounts.set(unit, (wordCounts.get(unit) ?? 0) + 1);
      }
    }

    const alphabetSet = new Set<string>();
    for (const unit of wordCounts.keys()) {
      for (const ch of unit) alphabetSet.add(ch);
    }
    const alphabet = Array.from(alphabetSet).sort();

    // Working representation: unique word -> symbol array.
    const words = new Map<string, string[]>();
    for (const unit of wordCounts.keys()) words.set(unit, Array.from(unit));

    const merges: BpeMerge[] = [];
    const minPairFrequency = options.minPairFrequency ?? 2;
    const targetMerges = options.vocabSize - specialList.length - alphabet.length;
    const rng = new AlphaRng(7);

    for (let step = 0; step < targetMerges; step++) {
      const pairCounts = new Map<string, number>();
      for (const [word, symbols] of words) {
        const weight = wordCounts.get(word) ?? 1;
        for (const [key, count] of countPairs(symbols)) {
          pairCounts.set(key, (pairCounts.get(key) ?? 0) + count * weight);
        }
      }
      let bestKey: string | null = null;
      let bestCount = 0;
      for (const [key, count] of pairCounts) {
        if (count > bestCount) {
          bestCount = count;
          bestKey = key;
        }
      }
      if (!bestKey || bestCount < minPairFrequency) break;
      const [left, right] = bestKey.split("\u0000");
      const merged = left + right;
      merges.push([left, right]);
      for (const [word, symbols] of words) {
        words.set(word, mergeSymbols(symbols, left, right, merged));
      }
      if (options.onProgress && (step + 1) % 32 === 0) {
        options.onProgress({
          merges: merges.length,
          vocabSize: specialList.length + alphabet.length + merges.length,
        });
      }
    }
    void rng;

    const tokenSet = new Set<string>(specialList);
    for (const ch of alphabet) tokenSet.add(ch);
    for (const [left, right] of merges) tokenSet.add(left + right);
    const tokens = Array.from(tokenSet);

    return new AlphaTokenizer({
      version: options.version ?? "0.1.0",
      trainedOn: options.trainedOn ?? "alpha-corpus",
      merges,
      alphabet,
      specialTokens: specials,
      tokens,
      stats: {
        documents: documents.length,
        characters,
        mergeSteps: merges.length,
        builtAt: Date.now(),
      },
    });
  }
}
