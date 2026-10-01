/**
 * Corpus diversity metrics.
 *
 * Step 4's quality module answered "is this corpus structurally usable". This
 * one answers "does this corpus contain variety", which is the question that
 * decides whether a model can learn anything general from it.
 *
 * Every metric here is a *count or a ratio computed from the text*. None of them
 * is a judgement. Where a threshold is used (for boilerplate, for low
 * information) the threshold is an explicit option with a stated default, and
 * the measured value is always reported alongside the finding so a reader can
 * disagree with the threshold without re-running the analysis.
 *
 * The metrics are deliberately tokenizer-independent where possible — they
 * operate on characters, words and sentences — because a diversity number that
 * changes when the vocabulary changes cannot be compared across versions. The
 * one token-level measure, token diversity, takes a tokenizer and is reported
 * separately.
 *
 * Nothing is deleted. Findings are findings.
 */

/** Words are runs of letters/digits; punctuation and whitespace are structure. */
const WORD_PATTERN = /[\p{L}\p{N}][\p{L}\p{N}'’-]*/gu;

/** Split a document into sentences on terminal punctuation followed by space. */
export function splitSentences(text: string): string[] {
  return text
    .split(/\n+/)
    .flatMap((line) => line.split(/(?<=[.!?])\s+(?=[\p{Lu}\p{N}"'])/u))
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

/** Extract lowercase word tokens from a text. */
export function words(text: string): string[] {
  return (text.match(WORD_PATTERN) ?? []).map((w) => w.toLowerCase());
}

export type DiversityOptions = {
  /**
   * A sentence whose length in characters is below this is not counted as a
   * sentence for the sentence-diversity measure. Default 10.
   */
  minSentenceCharacters?: number;
  /**
   * A document whose share of characters that also appear in other documents is
   * above this is reported as boilerplate. Default 0.6.
   */
  boilerplateThreshold?: number;
  /**
   * A document is "low information" when its distinct-word ratio falls below
   * this. Default 0.25.
   */
  lowInformationThreshold?: number;
};

export type DistributionSummary = {
  count: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  p10: number;
  p90: number;
};

export type DiversityReport = {
  documents: number;
  characters: number;
  /** Distinct words across the corpus and their type/token ratio. */
  vocabulary: {
    size: number;
    tokens: number;
    typeTokenRatio: number;
    /** Words that appear exactly once, and their share of the vocabulary. */
    hapaxRatio: number;
    /** Share of word occurrences taken by the most common 100 words. */
    top100Coverage: number;
  };
  /** Sentence-level variety. */
  sentences: {
    count: number;
    perDocument: number;
    /** Distinct sentence "shapes" (first three words), over total sentences. */
    distinctOpeningRatio: number;
    /** Mean characters per sentence. */
    meanCharacters: number;
  };
  /** Character-level variety. */
  characterVariety: {
    distinct: number;
    /** Shannon entropy over the character distribution, in bits. */
    entropyBits: number;
    /** Share of characters that are letters, whitespace or digits. */
    printableRatio: number;
  };
  /** Document length distribution, in characters. */
  length: DistributionSummary;
  /** Word-length distribution. */
  wordLength: DistributionSummary;
  /** Token-level variety; null when no tokenizer was supplied. */
  tokens: {
    measured: boolean;
    distinct: number | null;
    total: number | null;
    typeTokenRatio: number | null;
  };
  /** Boilerplate: documents sharing most of their content with the corpus. */
  boilerplate: {
    documents: number;
    ratio: number;
    /** Mean character overlap of a document with the rest of the corpus. */
    meanOverlap: number;
  };
  /** Documents that repeat themselves far beyond ordinary prose. */
  lowInformation: {
    documents: number;
    ratio: number;
    /** Mean distinct-word ratio across documents, for comparison. */
    meanDistinctWordRatio: number;
  };
  options: Required<DiversityOptions>;
};

function summarise(values: number[]): DistributionSummary {
  if (values.length === 0) {
    return { count: 0, min: 0, max: 0, mean: 0, median: 0, p10: 0, p90: 0 };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  const sum = values.reduce((s, v) => s + v, 0);
  return {
    count: values.length,
    min: sorted[0],
    max: sorted[sorted.length - 1],
    mean: sum / values.length,
    median: at(0.5),
    p10: at(0.1),
    p90: at(0.9),
  };
}

function shannonEntropy(counts: Map<string, number>, total: number): number {
  if (total <= 0) return 0;
  let bits = 0;
  for (const count of counts.values()) {
    const p = count / total;
    if (p > 0) bits -= p * Math.log2(p);
  }
  return bits;
}

/**
 * Compute the diversity report for a corpus.
 *
 * `documents` is the text of every document. `tokenizer` is optional; when
 * present, token diversity is measured as well.
 */
export function analyseDiversity(
  documents: string[],
  tokenizer?: { encode(text: string): number[] },
  options: DiversityOptions = {},
): DiversityReport {
  const resolved: Required<DiversityOptions> = {
    minSentenceCharacters: options.minSentenceCharacters ?? 10,
    boilerplateThreshold: options.boilerplateThreshold ?? 0.6,
    lowInformationThreshold: options.lowInformationThreshold ?? 0.25,
  };

  const characters = documents.reduce((sum, d) => sum + d.length, 0);

  // --- word level -----------------------------------------------------------
  const wordCounts = new Map<string, number>();
  let wordTokens = 0;
  const perDocumentWords: string[][] = [];
  for (const document of documents) {
    const docWords = words(document);
    perDocumentWords.push(docWords);
    for (const word of docWords) {
      wordCounts.set(word, (wordCounts.get(word) ?? 0) + 1);
    }
    wordTokens += docWords.length;
  }
  const hapax = [...wordCounts.values()].filter((c) => c === 1).length;
  const top100 = [...wordCounts.values()]
    .sort((a, b) => b - a)
    .slice(0, 100)
    .reduce((s, c) => s + c, 0);

  // --- sentence level -------------------------------------------------------
  const allSentences: string[] = [];
  const openingKeys = new Map<string, number>();
  let sentenceCharacters = 0;
  for (const document of documents) {
    for (const sentence of splitSentences(document)) {
      if (sentence.length < resolved.minSentenceCharacters) continue;
      allSentences.push(sentence);
      sentenceCharacters += sentence.length;
      const key = words(sentence).slice(0, 3).join(" ");
      openingKeys.set(key, (openingKeys.get(key) ?? 0) + 1);
    }
  }

  // --- character level ------------------------------------------------------
  const charCounts = new Map<string, number>();
  let printable = 0;
  for (const document of documents) {
    for (const ch of document) {
      charCounts.set(ch, (charCounts.get(ch) ?? 0) + 1);
      if (ch === " " || ch === "\n" || ch === "\t" || /[\p{L}\p{N}]/u.test(ch)) printable++;
    }
  }

  // --- boilerplate ---------------------------------------------------------
  // A document is boilerplate when a large share of its distinct 5-word shingles
  // also occur in other documents. Templated corpora fail this immediately.
  const globalShingles = new Map<string, number>();
  const perDocumentShingles: Set<string>[] = documents.map((document) => {
    const set = new Set<string>();
    const docWords = words(document);
    for (let i = 0; i + 5 <= docWords.length; i++) {
      const key = docWords.slice(i, i + 5).join(" ");
      set.add(key);
      globalShingles.set(key, (globalShingles.get(key) ?? 0) + 1);
    }
    return set;
  });
  let boilerplateDocuments = 0;
  let overlapSum = 0;
  perDocumentShingles.forEach((set, index) => {
    if (set.size === 0) return;
    let shared = 0;
    for (const shingle of set) {
      if ((globalShingles.get(shingle) ?? 0) > 1) shared++;
    }
    const overlap = shared / set.size;
    overlapSum += overlap;
    if (overlap >= resolved.boilerplateThreshold) boilerplateDocuments++;
    void index;
  });

  // --- low information -----------------------------------------------------
  // A document with very few distinct words relative to its length carries
  // almost no learnable signal, whatever it is about.
  let lowInformationDocuments = 0;
  let distinctWordRatioSum = 0;
  for (const docWords of perDocumentWords) {
    const distinct = new Set(docWords).size;
    const ratio = docWords.length === 0 ? 0 : distinct / docWords.length;
    distinctWordRatioSum += ratio;
    if (ratio < resolved.lowInformationThreshold) lowInformationDocuments++;
  }

  // --- token level ---------------------------------------------------------
  let tokenDistinct: number | null = null;
  let tokenTotal: number | null = null;
  if (tokenizer) {
    const ids = new Set<number>();
    let total = 0;
    for (const document of documents) {
      const encoded = tokenizer.encode(document);
      total += encoded.length;
      for (const id of encoded) ids.add(id);
    }
    tokenDistinct = ids.size;
    tokenTotal = total;
  }

  const docCount = documents.length;

  return {
    documents: docCount,
    characters,
    vocabulary: {
      size: wordCounts.size,
      tokens: wordTokens,
      typeTokenRatio: wordTokens === 0 ? 0 : wordCounts.size / wordTokens,
      hapaxRatio: wordCounts.size === 0 ? 0 : hapax / wordCounts.size,
      top100Coverage: wordTokens === 0 ? 0 : top100 / wordTokens,
    },
    sentences: {
      count: allSentences.length,
      perDocument: docCount === 0 ? 0 : allSentences.length / docCount,
      distinctOpeningRatio:
        allSentences.length === 0 ? 0 : openingKeys.size / allSentences.length,
      meanCharacters: allSentences.length === 0 ? 0 : sentenceCharacters / allSentences.length,
    },
    characterVariety: {
      distinct: charCounts.size,
      entropyBits: shannonEntropy(charCounts, characters),
      printableRatio: characters === 0 ? 0 : printable / characters,
    },
    length: summarise(documents.map((d) => d.length)),
    wordLength: summarise([...wordCounts.keys()].map((w) => w.length)),
    tokens: {
      measured: tokenizer !== undefined,
      distinct: tokenDistinct,
      total: tokenTotal,
      typeTokenRatio:
        tokenDistinct !== null && tokenTotal !== null && tokenTotal > 0
          ? tokenDistinct / tokenTotal
          : null,
    },
    boilerplate: {
      documents: boilerplateDocuments,
      ratio: docCount === 0 ? 0 : boilerplateDocuments / docCount,
      meanOverlap: perDocumentShingles.length === 0 ? 0 : overlapSum / perDocumentShingles.length,
    },
    lowInformation: {
      documents: lowInformationDocuments,
      ratio: docCount === 0 ? 0 : lowInformationDocuments / docCount,
      meanDistinctWordRatio:
        perDocumentWords.length === 0 ? 0 : distinctWordRatioSum / perDocumentWords.length,
    },
    options: resolved,
  };
}

export type DiversityDelta = {
  metric: string;
  step4: number;
  step5: number;
  /** Absolute change, step5 minus step4. */
  delta: number;
  /** True when the change is in the direction that means "more varied". */
  improved: boolean;
};

/**
 * Compare two diversity reports metric by metric.
 *
 * `higherIsBetter` decides which direction counts as an improvement for each
 * metric, because "more boilerplate" and "more vocabulary" cannot both be
 * improved by moving in the same direction. The decision is a parameter, not a
 * hidden assumption, and the raw values are always shown next to the verdict.
 */
export function compareDiversity(
  step4: DiversityReport,
  step5: DiversityReport,
  higherIsBetter: Record<string, boolean>,
): DiversityDelta[] {
  const rows: Array<[string, number, number]> = [
    ["vocabulary.size", step4.vocabulary.size, step5.vocabulary.size],
    ["vocabulary.typeTokenRatio", step4.vocabulary.typeTokenRatio, step5.vocabulary.typeTokenRatio],
    ["vocabulary.hapaxRatio", step4.vocabulary.hapaxRatio, step5.vocabulary.hapaxRatio],
    ["sentences.perDocument", step4.sentences.perDocument, step5.sentences.perDocument],
    [
      "sentences.distinctOpeningRatio",
      step4.sentences.distinctOpeningRatio,
      step5.sentences.distinctOpeningRatio,
    ],
    ["characterVariety.distinct", step4.characterVariety.distinct, step5.characterVariety.distinct],
    ["characterVariety.entropyBits", step4.characterVariety.entropyBits, step5.characterVariety.entropyBits],
    ["length.p90", step4.length.p90, step5.length.p90],
    ["boilerplate.ratio", step4.boilerplate.ratio, step5.boilerplate.ratio],
    ["boilerplate.meanOverlap", step4.boilerplate.meanOverlap, step5.boilerplate.meanOverlap],
    [
      "lowInformation.ratio",
      step4.lowInformation.ratio,
      step5.lowInformation.ratio,
    ],
  ];
  return rows.map(([metric, before, after]) => {
    const higher = higherIsBetter[metric] ?? true;
    return {
      metric,
      step4: before,
      step5: after,
      delta: after - before,
      improved: higher ? after > before : after < before,
    };
  });
}

/** Metrics where a *lower* value means a more varied corpus. */
export const DIVERSITY_LOWER_IS_BETTER = new Set([
  "boilerplate.ratio",
  "boilerplate.meanOverlap",
  "lowInformation.ratio",
]);

/** A compact, printable line. Numbers only. */
export function summariseDiversity(report: DiversityReport): string {
  return (
    `${report.documents} docs · ${report.characters.toLocaleString()} chars · ` +
    `vocab ${report.vocabulary.size.toLocaleString()} (TTR ${report.vocabulary.typeTokenRatio.toFixed(3)}) · ` +
    `${report.sentences.count.toLocaleString()} sentences (${report.sentences.perDocument.toFixed(1)}/doc, ` +
    `distinct openings ${report.sentences.distinctOpeningRatio.toFixed(3)}) · ` +
    `char entropy ${report.characterVariety.entropyBits.toFixed(2)} bits · ` +
    `boilerplate ${(report.boilerplate.ratio * 100).toFixed(1)}% · ` +
    `low-information ${(report.lowInformation.ratio * 100).toFixed(1)}%` +
    (report.tokens.measured && report.tokens.typeTokenRatio !== null
      ? ` · token TTR ${report.tokens.typeTokenRatio.toFixed(3)}`
      : "")
  );
}
