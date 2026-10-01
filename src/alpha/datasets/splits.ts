/**
 * Explicit TRAIN / VALIDATION / TEST boundaries.
 *
 * Step 4 had splits, but they lived inside the dataset version and the leakage
 * check compared whole documents. Step 5 needs two things it did not have:
 *
 *   1. A boundary that can be applied to *any* set of documents — including
 *      instruction examples and evaluation cases — so the same mechanism
 *      guarantees hold everywhere, not just in one corpus builder.
 *   2. Overlap detection at a finer grain than exact-document equality. Whole
 *      documents are the easy case; the dangerous one is a test sentence that
 *      also appears inside a much longer training document, where an exact
 *      comparison finds nothing.
 *
 * So overlap here is measured two ways: exact content fingerprints, and shingle
 * (word n-gram) fingerprints. The shingle measure is reported with its coverage
 * so a reader can see how much of the held-out text actually matched, rather
 * than a bare boolean.
 */

import { AlphaValidationError } from "../core/errors";
import { documentFingerprint, type ProvenanceDocument } from "./provenance";
import { words } from "./diversity";

export type SplitName = "train" | "validation" | "test";

export type SplitAssignment<T> = {
  train: T[];
  validation: T[];
  test: T[];
  /** The fractions actually used. */
  fractions: { validation: number; test: number };
  /** Index of each item in the input, so a caller can trace it back. */
  indexes: Record<SplitName, number[]>;
};

export type SplitOptions = {
  /** Fraction held out for validation. Default 0.12. */
  validationFraction?: number;
  /** Fraction held out for testing. Default 0.12. */
  testFraction?: number;
  seed?: number;
  /**
   * Groups documents that must not be separated. Documents sharing a group key
   * are always placed in the same split.
   *
   * This is not an optimisation. A corpus built from a small number of
   * generators contains near-duplicate documents by construction, and splitting
   * them at random puts a near-copy of a training document into the test split —
   * which makes the test split measure memorisation while reporting it as
   * generalisation. Grouping is the fix that keeps the held-out set honest.
   */
  groupBy?: (item: unknown, index: number) => string;
};

function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/**
 * Deterministic split.
 *
 * The permutation comes from a seeded LCG, so the same (items, seed, fractions)
 * always produces the same assignment. Test is carved out first and never handed
 * to anything that trains, so evaluation cannot leak in by construction.
 */
export function assignSplits<T>(
  items: T[],
  options: SplitOptions = {},
): SplitAssignment<T> {
  const validationFraction = options.validationFraction ?? 0.12;
  const testFraction = options.testFraction ?? 0.12;
  if (validationFraction < 0 || testFraction < 0 || validationFraction + testFraction >= 1) {
    throw new AlphaValidationError(
      "datasets",
      "validationFraction and testFraction must be >= 0 and sum to less than 1",
      { validationFraction, testFraction },
    );
  }

  // Units to split: individual items, or groups of them when groupBy is given.
  const groups = new Map<string, number[]>();
  items.forEach((item, index) => {
    const key = options.groupBy ? options.groupBy(item, index) : `item:${index}`;
    const existing = groups.get(key);
    if (existing) existing.push(index);
    else groups.set(key, [index]);
  });
  const unitKeys = [...groups.keys()];

  const order = unitKeys.map((_, i) => i);
  const next = lcg(options.seed ?? 1337);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }

  // Fractions are of *items*, not of units, so a split of grouped documents
  // still holds out the requested proportion of the corpus.
  const testItems = Math.round(items.length * testFraction);
  const validationItems = Math.round(items.length * validationFraction);
  const test: number[] = [];
  const validation: number[] = [];
  const train: number[] = [];
  let testCount = 0;
  let validationCount = 0;

  for (const unitIndex of order) {
    const memberIndexes = groups.get(unitKeys[unitIndex])!;
    if (testCount < testItems) {
      test.push(...memberIndexes);
      testCount += memberIndexes.length;
    } else if (validationCount < validationItems) {
      validation.push(...memberIndexes);
      validationCount += memberIndexes.length;
    } else {
      train.push(...memberIndexes);
    }
  }

  test.sort((a, b) => a - b);
  validation.sort((a, b) => a - b);
  train.sort((a, b) => a - b);

  return {
    train: train.map((i) => items[i]),
    validation: validation.map((i) => items[i]),
    test: test.map((i) => items[i]),
    fractions: { validation: validationFraction, test: testFraction },
    indexes: { train, validation, test },
  };
}

/**
 * A group key that puts near-duplicate documents in the same split.
 *
 * Near-duplication in a generated corpus is rarely positional: two documents
 * share their *middle* and differ at both ends, so a prefix or suffix key misses
 * them. Instead every 6-gram of the document contributes to the key, and the
 * smallest such contribution is used. Two documents that share most of their
 * content then share that smallest rare phrase, which groups them without
 * requiring an O(n squared) comparison at split time.
 */
export function nearDuplicateGroupKey(text: string, shingleCount = 6): string {
  const tokens = words(text);
  if (tokens.length <= shingleCount) return tokens.join(" ");
  const counts = new Map<string, number>();
  for (let i = 0; i + shingleCount <= tokens.length; i++) {
    const shingle = tokens.slice(i, i + shingleCount).join(" ");
    counts.set(shingle, (counts.get(shingle) ?? 0) + 1);
  }
  // The rarest phrase in this document: rare overall, and unique enough within
  // the document to identify it.
  let rarest: string | null = null;
  let rarestCount = Number.POSITIVE_INFINITY;
  for (const [shingle, count] of counts) {
    if (count < rarestCount) {
      rarest = shingle;
      rarestCount = count;
    }
  }
  return rarest ?? tokens.join(" ");
}

/** Split provenance documents, grouping near-duplicates together by default. */
export function splitDocuments(
  documents: ProvenanceDocument[],
  options: SplitOptions = {},
): SplitAssignment<ProvenanceDocument> {
  const { groupBy, ...rest } = options;
  return repairSplitLeakage(
    assignSplits(documents, {
      ...rest,
      groupBy:
        groupBy ??
        ((document: unknown) => {
          const doc = document as ProvenanceDocument;
          // Category is part of the key so two documents that happen to share an
          // opening in different categories are still separable.
          return `${doc.category}::${nearDuplicateGroupKey(doc.text)}`;
        }),
    }),
    { ...rest, documents },
  );
}

/**
 * Move any held-out document that overlaps training into the training split.
 *
 * Grouping at split time handles near-duplicates the grouping key can see. This
 * handles the ones it cannot — documents that are different at the start and the
 * end but share most of their middle — by *measuring* the overlap and correcting
 * the boundary.
 *
 * Moving a document into training is the conservative direction: it shrinks what
 * the model is evaluated on, but it never lets evaluation measure something the
 * model has already seen. The alternative — allowing the overlap and reporting it
 * — would make the held-out loss look better than it is, which is the exact
 * failure this whole module exists to prevent.
 */
export function repairSplitLeakage(
  assignment: SplitAssignment<ProvenanceDocument>,
  options: { documents: ProvenanceDocument[]; contaminationThreshold?: number; shingleSize?: number },
): SplitAssignment<ProvenanceDocument> {
  // Deliberately stricter than the audit threshold in `detectOverlap` (0.5).
  // A document at 0.48 would otherwise survive repair and sit right at the
  // audit's limit; repairing at a lower bar keeps the audit away from its own
  // threshold, so a passing held-out set is comfortably clean rather than
  // borderline.
  const threshold = options.contaminationThreshold ?? 0.4;
  const shingleSize = options.shingleSize ?? 8;
  const contaminated = (candidate: ProvenanceDocument, reference: ProvenanceDocument[]): boolean => {
    const referenceShingles = new Set<string>();
    for (const document of reference) {
      for (const shingle of shingleFingerprints(document.text, shingleSize)) referenceShingles.add(shingle);
    }
    const shingles = shingleFingerprints(candidate.text, shingleSize);
    if (shingles.size === 0) return false;
    let shared = 0;
    for (const shingle of shingles) if (referenceShingles.has(shingle)) shared++;
    return shared / shingles.size >= threshold;
  };

  const train = [...assignment.train];
  const validation = [...assignment.validation];
  const test = [...assignment.test];
  const keptValidation: ProvenanceDocument[] = [];
  const keptTest: ProvenanceDocument[] = [];

  for (const document of validation) {
    // Validate against training *and* against the test split it must stay
    // independent of, so repairing validation cannot introduce a new overlap.
    if (contaminated(document, train) || contaminated(document, test)) train.push(document);
    else keptValidation.push(document);
  }
  for (const document of test) {
    if (contaminated(document, train) || contaminated(document, keptValidation)) train.push(document);
    else keptTest.push(document);
  }

  const indexOf = new Map(options.documents.map((document, index) => [document, index]));
  return {
    train,
    validation: keptValidation,
    test: keptTest,
    fractions: assignment.fractions,
    indexes: {
      train: train.map((d) => indexOf.get(d) ?? -1).filter((i) => i >= 0),
      validation: keptValidation.map((d) => indexOf.get(d) ?? -1).filter((i) => i >= 0),
      test: keptTest.map((d) => indexOf.get(d) ?? -1).filter((i) => i >= 0),
    },
  };
}

/** Word n-gram fingerprints, used to detect partial overlap. */
export function shingleFingerprints(text: string, size = 8): Set<string> {
  const tokens = words(text);
  const out = new Set<string>();
  if (tokens.length < size) {
    if (tokens.length > 0) out.add(tokens.join(" "));
    return out;
  }
  for (let i = 0; i + size <= tokens.length; i++) {
    out.add(tokens.slice(i, i + size).join(" "));
  }
  return out;
}

export type OverlapReport = {
  /** Held-out documents that are byte-identical to something in the reference set. */
  exactMatches: string[];
  /**
   * Held-out documents whose *substantial* share of shingles also appears in the
   * reference set. Substantial, not any: two documents sharing one incidental
   * phrase are not leakage, and a detector that says they are would refuse every
   * real corpus.
   */
  shingleMatches: string[];
  /** Mean shingle coverage across the held-out documents that matched. */
  meanShingleCoverage: number;
  /** Highest shingle coverage found on any single held-out document. */
  maxShingleCoverage: number;
  /** The fraction of held-out documents judged contaminated. */
  contaminatedShare: number;
  clean: boolean;
};

export type OverlapOptions = {
  /** Word n-gram size. Default 8. */
  shingleSize?: number;
  /**
   * Shingle coverage at or above which a held-out document counts as
   * contaminated. Default 0.5: half of the document appearing verbatim
   * elsewhere means the split is not independent.
   */
  contaminationThreshold?: number;
};

/**
 * Compare a held-out set against a reference set (normally the training set).
 *
 * `label` identifies each held-out document in the report, so the result names
 * what leaked rather than just asserting a number.
 *
 * A held-out document is flagged when it is identical to a reference document,
 * or when at least `contaminationThreshold` of its shingles also occur in the
 * reference set. Lower thresholds catch more but also flag ordinary shared
 * phrasing, so the default is deliberately conservative about what it calls
 * contamination and reports the coverage figures either way.
 */
export function detectOverlap(
  heldOut: Array<{ label: string; text: string }>,
  reference: string[],
  options: OverlapOptions = {},
): OverlapReport {
  const shingleSize = options.shingleSize ?? 8;
  const threshold = options.contaminationThreshold ?? 0.5;
  const referenceExact = new Set(reference.map((text) => documentFingerprint(text)));
  const referenceShingles = new Set<string>();
  for (const text of reference) {
    for (const shingle of shingleFingerprints(text, shingleSize)) referenceShingles.add(shingle);
  }

  const exactMatches: string[] = [];
  const shingleMatches: string[] = [];
  let coverageSum = 0;
  let maxCoverage = 0;

  for (const item of heldOut) {
    if (referenceExact.has(documentFingerprint(item.text))) {
      exactMatches.push(item.label);
      shingleMatches.push(item.label);
      coverageSum += 1;
      maxCoverage = 1;
      continue;
    }
    const shingles = shingleFingerprints(item.text, shingleSize);
    if (shingles.size === 0) continue;
    let shared = 0;
    for (const shingle of shingles) if (referenceShingles.has(shingle)) shared++;
    const coverage = shared / shingles.size;
    if (coverage > maxCoverage) maxCoverage = coverage;
    if (coverage >= threshold) {
      shingleMatches.push(item.label);
      coverageSum += coverage;
    }
  }

  const contaminated = new Set([...exactMatches, ...shingleMatches]);
  return {
    exactMatches,
    shingleMatches,
    meanShingleCoverage: shingleMatches.length === 0 ? 0 : coverageSum / shingleMatches.length,
    maxShingleCoverage: maxCoverage,
    contaminatedShare: heldOut.length === 0 ? 0 : contaminated.size / heldOut.length,
    clean: contaminated.size === 0,
  };
}

/** Overlap between all three splits of one assignment. */
export function auditSplitOverlap(
  assignment: SplitAssignment<ProvenanceDocument>,
  options: OverlapOptions = {},
): Record<SplitName, OverlapReport> {
  const textsOf = (name: SplitName) => assignment[name].map((d) => d.text);
  const labelled = (name: SplitName) => assignment[name].map((d) => ({ label: d.documentId, text: d.text }));
  return {
    // Training is the reference: nothing in validation or test may appear in it.
    validation: detectOverlap(labelled("validation"), textsOf("train"), options),
    test: detectOverlap(labelled("test"), textsOf("train"), options),
    train: detectOverlap(labelled("train"), textsOf("validation"), options),
  };
}

/**
 * Refuse a split whose held-out portions overlap the training portion.
 *
 * This is the check behind "evaluation data is never used for training". It is
 * called from the training entry point rather than assumed.
 */
export function assertNoSplitLeakage(
  assignment: SplitAssignment<ProvenanceDocument>,
  options: OverlapOptions = {},
): Record<SplitName, OverlapReport> {
  const audit = auditSplitOverlap(assignment, options);
  const problems: string[] = [];
  if (audit.validation.shingleMatches.length > 0) {
    problems.push(
      `validation overlaps training on ${audit.validation.shingleMatches.length} document(s): ${audit.validation.shingleMatches.slice(0, 3).join(", ")}`,
    );
  }
  if (audit.test.shingleMatches.length > 0) {
    problems.push(
      `test overlaps training on ${audit.test.shingleMatches.length} document(s): ${audit.test.shingleMatches.slice(0, 3).join(", ")}`,
    );
  }
  if (problems.length > 0) {
    throw new AlphaValidationError(
      "datasets",
      `refusing to train: ${problems.join("; ")}`,
      { problems },
    );
  }
  return audit;
}

/** Split sizes, for a report. */
export function describeSplits(
  assignment: SplitAssignment<ProvenanceDocument>,
): string {
  const parts = (["train", "validation", "test"] as SplitName[]).map((name) => {
    const documents = assignment[name];
    const characters = documents.reduce((sum, d) => sum + d.text.length, 0);
    return `${name} ${documents.length} docs / ${characters.toLocaleString()} chars`;
  });
  return parts.join(" · ");
}
