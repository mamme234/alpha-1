/**
 * Step 7's training corpus.
 *
 * A corpus is *authored text in this repository*, split across
 * `step7-docs-*.ts`, with metadata attached per document:
 *
 *   source / provenance   which source record the text belongs to
 *   licence               "Alpha-owned" for everything authored here
 *   dataset version       STEP7_CORPUS_VERSION, part of the dataset fingerprint
 *   language              BCP-47-ish code, "en" for English documents
 *   category              one of the eight mixture categories Step 5 defined
 *   topic                 what the document exercises (mathematics, coding,
 *                         reasoning, dialogue, …) — a finer label than category
 *   quality               grade + review flag, so quality is stated per
 *                         document and measured per corpus
 *
 * Nothing here is scraped, downloaded or produced by an external model, and
 * nothing here is sampled from the frozen evaluation suite: the builder is
 * followed by an explicit leakage audit in `alpha:verify-intelligence`.
 */

import {
  createProvenanceDocument,
  documentFingerprint,
  type MixCategory,
  type ProvenanceDocument,
  type ProvenanceSource,
} from "./provenance";
import { STEP7_DOCS_1 } from "./step7-docs-1";
import { STEP7_DOCS_2 } from "./step7-docs-2";
import { STEP7_DOCS_3 } from "./step7-docs-3";

export type Step7Topic =
  | "general-knowledge"
  | "explanation"
  | "dialogue"
  | "instruction"
  | "question-answer"
  | "technical"
  | "mathematics"
  | "reasoning"
  | "structured"
  | "factual"
  | "narrative"
  | "code"
  | "multilingual";

export type Step7Document = {
  id: string;
  category: MixCategory;
  topic: Step7Topic;
  language: string;
  /** Quality metadata recorded per document. */
  quality: {
    grade: "high" | "medium";
    reviewed: boolean;
    note?: string;
  };
  text: string;
};

/** Dataset version: part of the fingerprint, bumped only for content changes. */
export const STEP7_CORPUS_VERSION = "1.0.0";
/** Source id used in every provenance record built from this corpus. */
export const STEP7_SOURCE_ID = "alpha-step7-corpus";
/** Recorded timestamp for reproducible fingerprints (same convention as Step 5). */
export const STEP7_RECORDED_AT = 1_760_000_000_000;

export const STEP7_DOCUMENTS: Step7Document[] = [
  ...STEP7_DOCS_1,
  ...STEP7_DOCS_2,
  ...STEP7_DOCS_3,
];

export const STEP7_SOURCES: ProvenanceSource[] = [
  {
    id: STEP7_SOURCE_ID,
    title: "Alpha Step 7 authored capability corpus",
    origin: "authored",
    license: "Alpha-owned",
    documents: STEP7_DOCUMENTS.length,
    note: "Authored in this repository by src/alpha/datasets/step7-docs-*.ts. No scraped or third-party text, no external model output.",
  },
];

export type Step7CorpusBuild = {
  documents: ProvenanceDocument[];
  sources: ProvenanceSource[];
  duplicatesRemoved: number;
  unreviewed: number;
  topicCounts: Record<string, number>;
  languageCounts: Record<string, number>;
  characters: number;
};

/**
 * Attach provenance to every document, dropping exact duplicates rather than
 * training on the same text twice, and report what the corpus actually is.
 */
export function buildStep7Corpus(): Step7CorpusBuild {
  const seen = new Set<string>();
  const documents: ProvenanceDocument[] = [];
  const topicCounts: Record<string, number> = {};
  const languageCounts: Record<string, number> = {};
  let duplicatesRemoved = 0;
  let unreviewed = 0;
  let characters = 0;

  for (const doc of STEP7_DOCUMENTS) {
    const fingerprint = documentFingerprint(doc.text.trim());
    if (seen.has(fingerprint)) {
      duplicatesRemoved += 1;
      continue;
    }
    seen.add(fingerprint);
    characters += doc.text.length;
    topicCounts[doc.topic] = (topicCounts[doc.topic] ?? 0) + 1;
    languageCounts[doc.language] = (languageCounts[doc.language] ?? 0) + 1;
    if (!doc.quality.reviewed) unreviewed += 1;
    documents.push(
      createProvenanceDocument({
        documentId: doc.id,
        text: doc.text,
        sourceId: STEP7_SOURCE_ID,
        origin: "authored",
        license: "Alpha-owned",
        language: doc.language,
        category: doc.category,
        acquisition: {
          method: "generated-in-repo",
          location: "this-repository",
          acquiredAt: STEP7_RECORDED_AT,
          collectedBy: "alpha:step7-corpus",
          note: doc.quality.note,
        },
        createdAt: STEP7_RECORDED_AT,
        version: STEP7_CORPUS_VERSION,
      }),
    );
  }

  return {
    documents,
    sources: STEP7_SOURCES,
    duplicatesRemoved,
    unreviewed,
    topicCounts,
    languageCounts,
    characters,
  };
}
