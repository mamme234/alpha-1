/**
 * Tokenizer measurement.
 *
 * Step 5 says: retrain or modify the tokenizer *only* if measurements
 * demonstrate a meaningful reason. That means the measurement has to exist
 * first, and the threshold has to be a stated number rather than a feeling.
 *
 * What is measured here:
 *
 *   vocabularyCoverage   share of distinct corpus characters the vocabulary can
 *                        represent without falling back to <unk>
 *   tokensPerCharacter   sequence efficiency — lower means fewer positions to
 *                        attend over and fewer parameters in the output layer
 *   tokensPerWord        how finely words are being split
 *   unknownCharacters    characters the vocabulary cannot represent at all
 *   unknownTokens        how many positions actually fell back to <unk>
 *   roundTripExact       whether encode∘decode is lossless
 *   tokenFrequency       the distribution, so a vocabulary that is mostly rare
 *                        fragments is visible rather than assumed
 *
 * `decideTokenizerChange` turns two measurements (current and candidate) into a
 * decision with the measured numbers attached. The thresholds are parameters
 * with documented defaults, so a caller that disagrees can pass different ones
 * and see the decision change.
 */

import type { AlphaTokenizer } from "./bpe";
import { words } from "../datasets/diversity";

export type TokenizerMeasurement = {
  tokenizerVersion: string;
  tokenizerFingerprint: string;
  vocabularySize: number;
  characters: number;
  /** Distinct characters in the corpus. */
  distinctCharacters: number;
  /** Distinct corpus characters the vocabulary covers. */
  coveredCharacters: number;
  vocabularyCoverage: number;
  /** Characters present in the corpus but absent from the alphabet. */
  unknownCharacters: string[];
  totalTokens: number;
  unknownTokens: number;
  unknownTokenShare: number;
  tokensPerCharacter: number;
  tokensPerWord: number;
  words: number;
  /** Characters consumed per token; higher means more efficient. */
  charactersPerToken: number;
  /** Merges relative to words — how aggressively the vocabulary merged. */
  tokensPerWordPenalty: number;
  roundTripExact: boolean;
  /** Fraction of token occurrences held by the top 100 most frequent tokens. */
  top100Coverage: number;
  /** Tokens that occur exactly once across the corpus. */
  hapaxTokens: number;
  hapaxTokenShare: number;
  /** Distinct tokens actually used, over vocabulary size. */
  vocabularyUtilisation: number;
};

export type MeasureOptions = {
  /** Sample this many documents when the corpus is larger. Default 400. */
  sampleSize?: number;
  seed?: number;
};

/**
 * Measure a tokenizer against a corpus.
 *
 * Deterministic: the same corpus and options give the same numbers, because the
 * sample is taken with a fixed stride rather than randomly.
 */
export function measureTokenizer(
  tokenizer: AlphaTokenizer,
  documents: string[],
  options: MeasureOptions = {},
): TokenizerMeasurement {
  const sampleSize = options.sampleSize ?? 400;
  // Stride sampling rather than random sampling: reproducible without an rng,
  // and it spreads across the corpus rather than clustering its start.
  const stride = Math.max(1, Math.floor(documents.length / Math.max(1, sampleSize)));
  const sample: string[] = [];
  for (let i = 0; i < documents.length && sample.length < sampleSize; i += stride) {
    sample.push(documents[i]);
  }

  let characters = 0;
  let totalTokens = 0;
  let unknownTokens = 0;
  let wordCount = 0;
  const corpusChars = new Set<string>();
  const covered = new Set<string>();
  const tokenCounts = new Map<number, number>();
  let roundTripExact = true;

  for (const document of sample) {
    characters += document.length;
    for (const ch of document) corpusChars.add(ch);
    wordCount += words(document).length;

    const encoded = tokenizer.encodeDetailed(document);
    totalTokens += encoded.ids.length;
    unknownTokens += encoded.unknown;
    for (const id of encoded.ids) tokenCounts.set(id, (tokenCounts.get(id) ?? 0) + 1);

    // Coverage is about the alphabet, not the merge table: a character is
    // covered when the tokenizer can represent it on its own.
    const alphabet = new Set(tokenizer.alphabet);
    for (const ch of document) {
      if (alphabet.has(ch)) covered.add(ch);
    }

    const decoded = tokenizer.decode(encoded.ids);
    if (tokenizer.encode(decoded).join(",") !== encoded.ids.join(",")) roundTripExact = false;
  }

  const unknownCharacters = [...corpusChars].filter((ch) => !covered.has(ch)).sort();
  const hapax = [...tokenCounts.values()].filter((count) => count === 1).length;
  const top100 = [...tokenCounts.values()]
    .sort((a, b) => b - a)
    .slice(0, 100)
    .reduce((sum, count) => sum + count, 0);
  const distinctTokens = tokenCounts.size;

  return {
    tokenizerVersion: tokenizer.version,
    tokenizerFingerprint: tokenizer.fingerprint(),
    vocabularySize: tokenizer.vocabSize,
    characters,
    distinctCharacters: corpusChars.size,
    coveredCharacters: covered.size,
    vocabularyCoverage: corpusChars.size === 0 ? 0 : covered.size / corpusChars.size,
    unknownCharacters,
    totalTokens,
    unknownTokens,
    unknownTokenShare: totalTokens === 0 ? 0 : unknownTokens / totalTokens,
    tokensPerCharacter: characters === 0 ? 0 : totalTokens / characters,
    tokensPerWord: wordCount === 0 ? 0 : totalTokens / wordCount,
    words: wordCount,
    charactersPerToken: totalTokens === 0 ? 0 : characters / totalTokens,
    tokensPerWordPenalty: wordCount === 0 ? 0 : (totalTokens - wordCount) / wordCount,
    roundTripExact,
    top100Coverage: totalTokens === 0 ? 0 : top100 / totalTokens,
    hapaxTokens: hapax,
    hapaxTokenShare: distinctTokens === 0 ? 0 : hapax / distinctTokens,
    vocabularyUtilisation: tokenizer.vocabSize === 0 ? 0 : distinctTokens / tokenizer.vocabSize,
  };
}

export type TokenizerDecisionReason =
  | "candidate-is-not-better"
  | "coverage-improvable"
  | "efficiency-improvable"
  | "coverage-loss"
  | "round-trip-broken"
  | "candidate-adopted";

export type TokenizerDecision = {
  retrain: boolean;
  reason: TokenizerDecisionReason;
  /** Measured, never estimated. */
  current: {
    vocabularySize: number;
    vocabularyCoverage: number;
    tokensPerCharacter: number;
    charactersPerToken: number;
    unknownTokenShare: number;
    fingerprint: string;
  };
  candidate: {
    vocabularySize: number;
    vocabularyCoverage: number;
    tokensPerCharacter: number;
    charactersPerToken: number;
    unknownTokenShare: number;
    fingerprint: string;
  };
  /** charactersPerToken candidate minus current; positive means denser tokens. */
  efficiencyDelta: number;
  /** Coverage candidate minus current. */
  coverageDelta: number;
  thresholds: Required<TokenizerDecisionThresholds>;
  /** Written reasoning, so the decision can be argued with. */
  explanation: string;
};

export type TokenizerDecisionThresholds = {
  /**
   * Minimum relative improvement in characters-per-token before a retrain is
   * worth the cost of invalidating every checkpoint. Default 0.05 (5%).
   */
  minEfficiencyImprovement: number;
  /**
   * Candidate coverage may not fall below this, even if it is more efficient.
   * Default 0.99.
   */
  minCoverage: number;
  /** Any unknown token share above this disqualifies the candidate. Default 0.001. */
  maxUnknownTokenShare: number;
};

export const DEFAULT_TOKENIZER_DECISION_THRESHOLDS: TokenizerDecisionThresholds = {
  minEfficiencyImprovement: 0.05,
  minCoverage: 0.99,
  maxUnknownTokenShare: 0.001,
};

function summarise(m: TokenizerMeasurement) {
  return {
    vocabularySize: m.vocabularySize,
    vocabularyCoverage: m.vocabularyCoverage,
    tokensPerCharacter: m.tokensPerCharacter,
    charactersPerToken: m.charactersPerToken,
    unknownTokenShare: m.unknownTokenShare,
    fingerprint: m.tokenizerFingerprint,
  };
}

/**
 * Decide whether a candidate tokenizer justifies replacing the current one.
 *
 * The decision is deliberately conservative: changing a tokenizer invalidates
 * every checkpoint trained against the old one, so "slightly denser tokens" is
 * not on its own a reason. Coverage or a broken round trip *is* a reason to act
 * even if efficiency is flat.
 */
export function decideTokenizerChange(
  current: TokenizerMeasurement,
  candidate: TokenizerMeasurement,
  thresholds: TokenizerDecisionThresholds = DEFAULT_TOKENIZER_DECISION_THRESHOLDS,
): TokenizerDecision {
  const resolved = { ...DEFAULT_TOKENIZER_DECISION_THRESHOLDS, ...thresholds };
  const efficiencyDelta = candidate.charactersPerToken - current.charactersPerToken;
  const coverageDelta = candidate.vocabularyCoverage - current.vocabularyCoverage;
  const relativeEfficiency =
    current.charactersPerToken === 0 ? 0 : efficiencyDelta / current.charactersPerToken;

  const base = {
    current: summarise(current),
    candidate: summarise(candidate),
    efficiencyDelta,
    coverageDelta,
    thresholds: resolved,
  };

  if (!candidate.roundTripExact) {
    return {
      ...base,
      retrain: false,
      reason: "round-trip-broken",
      explanation:
        "the candidate tokenizer does not round-trip its own output, so it is rejected regardless of efficiency",
    };
  }
  if (candidate.unknownTokenShare > resolved.maxUnknownTokenShare) {
    return {
      ...base,
      retrain: false,
      reason: "round-trip-broken",
      explanation:
        `the candidate leaves ${(candidate.unknownTokenShare * 100).toFixed(3)}% of tokens unknown, ` +
        `above the ${(resolved.maxUnknownTokenShare * 100).toFixed(3)}% ceiling`,
    };
  }
  if (candidate.vocabularyCoverage < resolved.minCoverage) {
    return {
      ...base,
      retrain: false,
      reason: "coverage-loss",
      explanation:
        `the candidate covers ${(candidate.vocabularyCoverage * 100).toFixed(2)}% of corpus characters, ` +
        `below the ${(resolved.minCoverage * 100).toFixed(0)}% floor; unknown characters: ` +
        `${candidate.unknownCharacters.join(" ") || "none"}`,
    };
  }
  if (relativeEfficiency >= resolved.minEfficiencyImprovement) {
    return {
      ...base,
      retrain: true,
      reason: "candidate-adopted",
      explanation:
        `the candidate is ${(relativeEfficiency * 100).toFixed(1)}% denser ` +
        `(${current.charactersPerToken.toFixed(4)} -> ${candidate.charactersPerToken.toFixed(4)} characters per token) ` +
        `at ${(candidate.vocabularyCoverage * 100).toFixed(2)}% coverage, above the ` +
        `${(resolved.minEfficiencyImprovement * 100).toFixed(0)}% threshold`,
    };
  }
  if (coverageDelta > 0.001) {
    return {
      ...base,
      retrain: true,
      reason: "coverage-improvable",
      explanation:
        `efficiency is effectively unchanged (${(relativeEfficiency * 100).toFixed(2)}%), but coverage improved ` +
        `by ${(coverageDelta * 100).toFixed(2)} points and the current tokenizer leaves ` +
        `${current.unknownCharacters.length} corpus character(s) unrepresented`,
    };
  }
  return {
    ...base,
    retrain: false,
    reason: "candidate-is-not-better",
    explanation:
      `the candidate is ${(relativeEfficiency * 100).toFixed(2)}% different in efficiency ` +
      `(threshold ${(resolved.minEfficiencyImprovement * 100).toFixed(0)}%) and coverage is unchanged ` +
      `(${coverageDelta >= 0 ? "+" : ""}${(coverageDelta * 100).toFixed(2)} points); ` +
      `switching would invalidate existing checkpoints for no measured gain`,
  };
}

/** One-line measurement summary for a report. */
export function summariseTokenizerMeasurement(m: TokenizerMeasurement): string {
  return (
    `${m.tokenizerVersion} (${m.tokenizerFingerprint}) · ${m.vocabularySize} tokens · ` +
    `coverage ${(m.vocabularyCoverage * 100).toFixed(2)}% of ${m.distinctCharacters} characters · ` +
    `${m.tokensPerCharacter.toFixed(4)} tokens/char (${m.charactersPerToken.toFixed(3)} chars/token) · ` +
    `${m.tokensPerWord.toFixed(2)} tokens/word · ` +
    `${m.unknownTokens} unknown token(s) (${(m.unknownTokenShare * 100).toFixed(4)}%) · ` +
    `utilisation ${(m.vocabularyUtilisation * 100).toFixed(1)}% · top-100 coverage ${(m.top100Coverage * 100).toFixed(1)}%`
  );
}

/** One-line decision summary. */
export function summariseTokenizerDecision(decision: TokenizerDecision): string {
  return (
    `${decision.retrain ? "RETRAIN" : "KEEP"} (${decision.reason}): ${decision.explanation}`
  );
}
