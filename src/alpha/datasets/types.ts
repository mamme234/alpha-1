/**
 * Alpha Datasets — corpus representation.
 *
 * Alpha trains on data that is *in* the project: a document array plus
 * provenance (license, source, version). Nothing is scraped at runtime and no
 * dataset is downloaded from a model hub.
 */

import { AlphaValidationError } from "../core/errors";

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
};

export function datasetStats(dataset: AlphaDataset): DatasetStats {
  let characters = 0;
  let lines = 0;
  const unique = new Set<string>();
  for (const doc of dataset.documents) {
    characters += doc.length;
    lines += doc.split("\n").length;
    for (const ch of doc) unique.add(ch);
  }
  return {
    documents: dataset.documents.length,
    characters,
    averageDocumentLength:
      dataset.documents.length === 0 ? 0 : Math.round(characters / dataset.documents.length),
    uniqueCharacters: unique.size,
    lines,
  };
}

export function createDataset(
  input: Omit<AlphaDataset, "id"> & { id?: string },
): AlphaDataset {
  if (input.documents.length === 0) {
    throw new AlphaValidationError("datasets", "a dataset needs at least one document");
  }
  return {
    id: input.id ?? `dataset_${input.name}_${input.version}`,
    ...input,
  };
}

/**
 * Deterministic train/validation split. Documents are interleaved rather than
 * sliced so both halves see every style present in the corpus.
 */
export function splitDocuments(
  dataset: AlphaDataset,
  validationFraction: number,
): { train: AlphaDataset; validation: AlphaDataset } {
  if (validationFraction <= 0 || validationFraction >= 1) {
    throw new AlphaValidationError("datasets", "validationFraction must be in (0, 1)");
  }
  const train: string[] = [];
  const validation: string[] = [];
  dataset.documents.forEach((doc, index) => {
    const bucket = (index * validationFraction) % 1;
    if (bucket < validationFraction / 2 + 1e-9 && validation.length < dataset.documents.length) {
      validation.push(doc);
    } else {
      train.push(doc);
    }
  });
  if (train.length === 0) train.push(dataset.documents[0]);
  if (validation.length === 0) validation.push(dataset.documents[dataset.documents.length - 1]);
  return {
    train: { ...dataset, documents: train },
    validation: { ...dataset, documents: validation },
  };
}
