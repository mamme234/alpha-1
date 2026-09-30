/**
 * Alpha Datasets — corpus representation, validation and splitting.
 *
 * Alpha trains on data that is *in* the project: a document array plus
 * provenance (licence, source, version). Nothing is scraped at runtime and no
 * dataset is downloaded from a model hub.
 *
 * The pipeline never silently discards training data. A document that is empty,
 * whitespace-only, non-string, oversized or repeated to the point of being
 * useless is reported as an issue and the pipeline refuses to start, rather
 * than quietly training on a corpus that is not what the caller thinks.
 */

import { AlphaValidationError } from "../core/errors";
import { ALPHA_RESOURCE_LIMITS, assertResourceLimit } from "../core/limits";

export type AlphaDataset = {
  id: string;
  name: string;
  version: string;
  description: string;
  /** SPDX-ish licence string for the corpus. */
  license: string;
  /** Where the text came from — "authored for Alpha" for the seed corpus. */
  source: string;
  documents: string[];
};

export type DatasetStats = {
  documents: number;
  characters: number;
  averageDocumentLength: number;
  uniqueCharacters: number;
  lines: number;
  shortestDocument: number;
  longestDocument: number;
  /** Documents that are empty after trimming — always 0 for a valid dataset. */
  emptyDocuments: number;
};

export function datasetStats(dataset: AlphaDataset): DatasetStats {
  let characters = 0;
  let lines = 0;
  let empty = 0;
  let shortest = Number.POSITIVE_INFINITY;
  let longest = 0;
  const unique = new Set<string>();
  for (const doc of dataset.documents) {
    const text = typeof doc === "string" ? doc : "";
    if (text.trim().length === 0) empty++;
    characters += text.length;
    lines += text.split("\n").length;
    if (text.length < shortest) shortest = text.length;
    if (text.length > longest) longest = text.length;
    for (const ch of text) unique.add(ch);
  }
  return {
    documents: dataset.documents.length,
    characters,
    averageDocumentLength:
      dataset.documents.length === 0 ? 0 : Math.round(characters / dataset.documents.length),
    uniqueCharacters: unique.size,
    lines,
    shortestDocument: dataset.documents.length === 0 ? 0 : shortest,
    longestDocument: longest,
    emptyDocuments: empty,
  };
}

export type DatasetIssue = {
  /** Index of the offending document, or -1 for whole-dataset problems. */
  index: number;
  problem: string;
};

export type DatasetValidation = {
  valid: boolean;
  issues: DatasetIssue[];
  /** Non-fatal observations, e.g. duplicated documents. */
  warnings: string[];
};

/**
 * Validate a corpus before it is tokenised. Returns every problem found (not
 * just the first) so a caller can fix the whole corpus in one pass.
 */
export function validateDataset(dataset: AlphaDataset): DatasetValidation {
  const issues: DatasetIssue[] = [];
  const warnings: string[] = [];
  if (!Array.isArray(dataset.documents)) {
    return { valid: false, issues: [{ index: -1, problem: "documents must be an array of strings" }], warnings };
  }
  if (dataset.documents.length === 0) {
    issues.push({ index: -1, problem: "a dataset needs at least one document" });
  }
  if (dataset.documents.length > ALPHA_RESOURCE_LIMITS.maxDocuments) {
    issues.push({
      index: -1,
      problem: `${dataset.documents.length} documents exceeds Alpha's limit of ${ALPHA_RESOURCE_LIMITS.maxDocuments}`,
    });
  }
  if (!dataset.name?.trim()) issues.push({ index: -1, problem: "a dataset needs a name" });
  if (!dataset.version?.trim()) issues.push({ index: -1, problem: "a dataset needs a version" });
  if (!dataset.license?.trim()) {
    issues.push({ index: -1, problem: "a dataset needs a licence — untraceable training data is refused" });
  }
  if (!dataset.source?.trim()) issues.push({ index: -1, problem: "a dataset needs a source" });

  let trainableCharacters = 0;
  dataset.documents.forEach((doc, index) => {
    if (typeof doc !== "string") {
      issues.push({ index, problem: `document ${index} is ${typeof doc}, not a string` });
      return;
    }
    if (doc.trim().length === 0) {
      issues.push({ index, problem: `document ${index} is empty or whitespace only` });
      return;
    }
    trainableCharacters += doc.trim().length;
    if (doc.length > ALPHA_RESOURCE_LIMITS.maxDocumentCharacters) {
      issues.push({
        index,
        problem: `document ${index} is ${doc.length} characters, above Alpha's limit of ${ALPHA_RESOURCE_LIMITS.maxDocumentCharacters}`,
      });
    }
  });

  if (trainableCharacters < 32 && issues.length === 0) {
    issues.push({
      index: -1,
      problem: "the corpus is too small to train on: fewer than 32 non-whitespace characters",
    });
  }

  const seen = new Map<string, number>();
  dataset.documents.forEach((doc, index) => {
    if (typeof doc !== "string") return;
    const key = doc.trim();
    if (key.length === 0) return;
    const first = seen.get(key);
    if (first === undefined) {
      seen.set(key, index);
    } else {
      warnings.push(`document ${index} duplicates document ${first}`);
    }
  });

  return { valid: issues.length === 0, issues, warnings };
}

/** Throwing form used by the pipeline entry points. */
export function assertValidDataset(dataset: AlphaDataset): DatasetValidation {
  const validation = validateDataset(dataset);
  if (!validation.valid) {
    const detail = validation.issues
      .slice(0, 6)
      .map((issue) => (issue.index >= 0 ? `#${issue.index}: ${issue.problem}` : issue.problem))
      .join("; ");
    throw new AlphaValidationError(
      "datasets",
      `corpus "${dataset.name ?? "unnamed"}" failed validation: ${detail}`,
      { issues: validation.issues },
    );
  }
  return validation;
}

export function createDataset(
  input: Omit<AlphaDataset, "id"> & { id?: string },
): AlphaDataset {
  const dataset: AlphaDataset = {
    id: input.id ?? `dataset_${input.name}_${input.version}`,
    name: input.name,
    version: input.version,
    description: input.description,
    license: input.license,
    source: input.source,
    documents: [...input.documents],
  };
  assertValidDataset(dataset);
  return dataset;
}

/**
 * Deterministic train/validation split. Documents are interleaved rather than
 * sliced so both halves see every style present in the corpus, and the
 * assignment depends only on the document index — the same corpus always
 * splits the same way, which is what makes a resumed run comparable.
 */
export function splitDocuments(
  dataset: AlphaDataset,
  validationFraction: number,
): { train: AlphaDataset; validation: AlphaDataset } {
  assertValidDataset(dataset);
  if (validationFraction <= 0 || validationFraction >= 1) {
    throw new AlphaValidationError("datasets", "validationFraction must be in (0, 1)");
  }
  const train: string[] = [];
  const validation: string[] = [];
  // Take every `stride`-th document, so both halves see the start and the end
  // and the held-out share is stable rather than random.
  const stride = Math.max(2, Math.round(1 / validationFraction));
  dataset.documents.forEach((doc, index) => {
    const heldOut = index % stride === 0 && validation.length + 1 < dataset.documents.length;
    if (heldOut) validation.push(doc);
    else train.push(doc);
  });
  if (train.length === 0) train.push(dataset.documents[0]);
  if (validation.length === 0) validation.push(dataset.documents[dataset.documents.length - 1]);
  return {
    train: { ...dataset, documents: train },
    validation: { ...dataset, documents: validation },
  };
}

/**
 * Stable 32-bit fingerprint (FNV-1a) of a corpus. Checkpoints record it so a
 * resumed run can prove it is looking at the same documents.
 */
export function datasetFingerprint(dataset: AlphaDataset): string {
  let hash = 0x811c9dc5;
  const feed = (text: string) => {
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
  };
  feed(dataset.name);
  feed("@");
  feed(dataset.version);
  feed("|");
  dataset.documents.forEach((doc, index) => {
    feed(`${index}:`);
    feed(typeof doc === "string" ? doc : "");
    feed("\u0000");
  });
  return `ds_${hash.toString(16).padStart(8, "0")}`;
}

/** Resolve a dataset id that may be missing, keeping ids readable. */
export function datasetReference(dataset: AlphaDataset): string {
  return `${dataset.name}@${dataset.version}#${datasetFingerprint(dataset)}`;
}
