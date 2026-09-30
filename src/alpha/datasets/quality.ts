/**
 * Dataset quality.
 *
 * Step 1's `validateDataset` answers "is this corpus structurally usable". That
 * is not the same question as "is this corpus any good", and it is certainly
 * not the question that decides whether a model is allowed to be trained on
 * it. This module answers the second question, deterministically, and produces
 * numbers rather than opinions.
 *
 * Two rules govern everything here:
 *
 *   1. Nothing is deleted. The analysis reports what it found; a caller that
 *      wants bad documents removed says so explicitly through the filter
 *      settings in `createDatasetVersion`, and the removal is recorded there.
 *   2. Every finding is reproducible. Same corpus, same seed, same report.
 */

import { datasetFingerprint, datasetStats, type AlphaDataset, type DatasetStats } from "./types";
import { hasInvalidUnicode, repetitionRatio, type AlphaDatasetVersion } from "./versions";

export type QualitySeverity = "error" | "warning" | "info";

export type QualityFinding = {
  /** Stable machine-readable code, e.g. "duplicate-documents". */
  code: string;
  severity: QualitySeverity;
  message: string;
  /** How many documents the finding applies to. */
  affected: number;
  /** Indexes into the dataset's `documents`, when the finding is document-specific. */
  documentIndexes?: number[];
};

export type QualityReport = {
  fingerprint: string;
  documents: number;
  characters: number;
  stats: DatasetStats;
  findings: QualityFinding[];
  /** Counts of documents by class, always summing to `documents`. */
  counts: {
    empty: number;
    duplicate: number;
    invalidUnicode: number;
    repetitive: number;
    suspiciouslyShort: number;
    oversized: number;
    clean: number;
  };
  duplicateRatio: number;
  /** Documents shorter than this share a `warning`. */
  shortThreshold: number;
  repetitionThreshold: number;
  /** Split names present, with their sizes. */
  splits: Array<{ name: string; documents: number }>;
  /** Sources with missing licence or unknown origin. */
  unlicensedSources: string[];
  pass: boolean;
  errors: number;
  warnings: number;
};

export type QualityOptions = {
  /** Documents below this many characters count as suspiciously short. Default 40. */
  shortThreshold?: number;
  /** Documents above this many characters count as oversized. Default 500_000. */
  longThreshold?: number;
  /** Repetition ratio above this counts as repetitive. Default 0.5. */
  repetitionThreshold?: number;
  /** A duplicate ratio above this is a warning. Default 0.1. */
  duplicateRatioThreshold?: number;
};

/** Deterministic whitespace-and-case-insensitive key for duplicate detection. */
function duplicateKey(text: string): string {
  return text.toLowerCase().replace(/\s+/g, " ").trim();
}

/**
 * Analyse a dataset and report measurable quality findings. Nothing is
 * modified; the returned counts describe what is actually in the corpus.
 */
export function analyseDatasetQuality(
  dataset: AlphaDataset,
  options: QualityOptions = {},
): QualityReport {
  const shortThreshold = options.shortThreshold ?? 40;
  const longThreshold = options.longThreshold ?? 500_000;
  const repetitionThreshold = options.repetitionThreshold ?? 0.5;
  const duplicateRatioThreshold = options.duplicateRatioThreshold ?? 0.1;

  const findings: QualityFinding[] = [];
  const counts = {
    empty: 0,
    duplicate: 0,
    invalidUnicode: 0,
    repetitive: 0,
    suspiciouslyShort: 0,
    oversized: 0,
    clean: 0,
  };

  const emptyIdx: number[] = [];
  const duplicateIdx: number[] = [];
  const invalidIdx: number[] = [];
  const repetitiveIdx: number[] = [];
  const shortIdx: number[] = [];
  const longIdx: number[] = [];

  const seen = new Map<string, number>();

  dataset.documents.forEach((doc, index) => {
    if (doc.trim().length === 0) {
      counts.empty++;
      emptyIdx.push(index);
      return;
    }
    if (hasInvalidUnicode(doc)) {
      counts.invalidUnicode++;
      invalidIdx.push(index);
      return;
    }
    const key = duplicateKey(doc);
    if (seen.has(key)) {
      counts.duplicate++;
      duplicateIdx.push(index);
    } else {
      seen.set(key, index);
    }
    if (doc.length < shortThreshold) {
      counts.suspiciouslyShort++;
      shortIdx.push(index);
    }
    if (doc.length > longThreshold) {
      counts.oversized++;
      longIdx.push(index);
    }
    if (repetitionRatio(doc) > repetitionThreshold) {
      counts.repetitive++;
      repetitiveIdx.push(index);
    }
  });

  const total = dataset.documents.length;
  counts.clean = Math.max(
    0,
    total - counts.empty - counts.invalidUnicode - counts.duplicate,
  );
  const duplicateRatio = total > 0 ? counts.duplicate / total : 0;

  if (counts.empty > 0) {
    findings.push({
      code: "empty-documents",
      severity: "error",
      message: `${counts.empty} document(s) contain no text`,
      affected: counts.empty,
      documentIndexes: emptyIdx,
    });
  }
  if (counts.invalidUnicode > 0) {
    findings.push({
      code: "invalid-unicode",
      severity: "error",
      message: `${counts.invalidUnicode} document(s) contain lone surrogates or replacement characters, so the bytes were not valid UTF-8`,
      affected: counts.invalidUnicode,
      documentIndexes: invalidIdx,
    });
  }
  if (counts.duplicate > 0) {
    findings.push({
      code: "duplicate-documents",
      severity: duplicateRatio > duplicateRatioThreshold ? "error" : "warning",
      message: `${counts.duplicate} of ${total} documents (${(duplicateRatio * 100).toFixed(1)}%) duplicate an earlier document`,
      affected: counts.duplicate,
      documentIndexes: duplicateIdx,
    });
  }
  if (counts.repetitive > 0) {
    findings.push({
      code: "excessive-repetition",
      severity: "warning",
      message: `${counts.repetitive} document(s) repeat a 24-character window more than ${(repetitionThreshold * 100).toFixed(0)}% of the time`,
      affected: counts.repetitive,
      documentIndexes: repetitiveIdx,
    });
  }
  if (counts.suspiciouslyShort > 0) {
    findings.push({
      code: "suspiciously-short",
      severity: "warning",
      message: `${counts.suspiciouslyShort} document(s) are shorter than ${shortThreshold} characters`,
      affected: counts.suspiciouslyShort,
      documentIndexes: shortIdx,
    });
  }
  if (counts.oversized > 0) {
    findings.push({
      code: "oversized-document",
      severity: "warning",
      message: `${counts.oversized} document(s) exceed ${longThreshold.toLocaleString()} characters`,
      affected: counts.oversized,
      documentIndexes: longIdx,
    });
  }

  const errors = findings.filter((f) => f.severity === "error").length;
  const warnings = findings.filter((f) => f.severity === "warning").length;

  return {
    fingerprint: datasetFingerprint(dataset),
    documents: total,
    characters: dataset.documents.reduce((sum, d) => sum + d.length, 0),
    stats: datasetStats(dataset),
    findings,
    counts,
    duplicateRatio,
    shortThreshold,
    repetitionThreshold,
    splits: [],
    unlicensedSources: [],
    pass: errors === 0,
    errors,
    warnings,
  };
}

/**
 * Extend a quality report with the things only a *versioned* dataset knows:
 * split sizes, leakage between splits, and licence/source provenance.
 */
export function analyseDatasetVersionQuality(
  version: AlphaDatasetVersion,
  options: QualityOptions = {},
): QualityReport {
  const base = analyseDatasetQuality(
    {
      id: version.datasetId,
      name: version.name,
      version: version.version,
      description: version.description,
      license: version.license,
      source: version.source,
      documents: version.documents,
    },
    options,
  );

  const splits = version.manifest.splits.map((s) => ({
    name: s.name,
    documents: s.documents,
  }));
  const unlicensed = version.sources
    .filter((s) => !s.license || s.license.trim().length === 0 || s.origin === "unknown")
    .map((s) => s.id);

  const findings = [...base.findings];

  // Leakage: the same normalised text appearing in two different splits would
  // let evaluation measure something the model has already memorised.
  const keysBySplit = new Map<string, Set<string>>();
  for (const split of version.manifest.splits) {
    const set = new Set<string>();
    for (const i of split.documentIndexes) {
      const doc = version.documents[i];
      if (doc) set.add(duplicateKey(doc));
    }
    keysBySplit.set(split.name, set);
  }
  const names = [...keysBySplit.keys()];
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const a = keysBySplit.get(names[i])!;
      const b = keysBySplit.get(names[j])!;
      const overlap = [...a].filter((k) => b.has(k)).length;
      if (overlap > 0) {
        findings.push({
          code: `split-leakage:${names[i]}-${names[j]}`,
          severity: "error",
          message: `${overlap} document(s) appear in both the "${names[i]}" and "${names[j]}" splits`,
          affected: overlap,
        });
      }
    }
  }

  if (unlicensed.length > 0) {
    findings.push({
      code: "missing-license",
      severity: "error",
      message: `source(s) ${unlicensed.join(", ")} have no licence or an unknown origin; Alpha will not train on them`,
      affected: unlicensed.length,
    });
  }

  const errors = findings.filter((f) => f.severity === "error").length;
  const warnings = findings.filter((f) => f.severity === "warning").length;

  return {
    ...base,
    findings,
    splits,
    unlicensedSources: unlicensed,
    pass: errors === 0,
    errors,
    warnings,
  };
}

/** A compact, printable summary. Numbers only — no judgement calls. */
export function summariseQualityReport(report: QualityReport): string {
  const status = report.pass ? "PASS" : "FAIL";
  const lines = [
    `Dataset quality ${status} — ${report.documents} docs, ${report.characters.toLocaleString()} chars, fp ${report.fingerprint}`,
    `  empty ${report.counts.empty} · duplicate ${report.counts.duplicate} (${(report.duplicateRatio * 100).toFixed(1)}%) · invalid-unicode ${report.counts.invalidUnicode} · repetitive ${report.counts.repetitive} · short ${report.counts.suspiciouslyShort} · oversized ${report.counts.oversized}`,
  ];
  if (report.splits.length) {
    lines.push(`  splits: ${report.splits.map((s) => `${s.name} ${s.documents}`).join(" · ")}`);
  }
  for (const f of report.findings) {
    lines.push(`  [${f.severity}] ${f.code}: ${f.message}`);
  }
  return lines.join("\n");
}
