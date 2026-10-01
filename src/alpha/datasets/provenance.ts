/**
 * Per-document provenance.
 *
 * Step 4 recorded provenance per *source*: "these 300 documents came from the
 * generator". That is not enough to answer the question an auditor actually
 * asks, which is about one specific paragraph: where did this text come from,
 * under what licence, in what language, in what category, and has it changed
 * since it was written.
 *
 * So provenance here is attached per document, and a dataset version carries
 * the whole record. Every document has:
 *
 *   documentId   stable within its corpus
 *   sourceId     which declared source it belongs to
 *   origin       authored | public-domain | user-owned | licensed
 *   license      SPDX-ish string, required
 *   acquisition  how it was obtained, when, and by whom
 *   language     ISO 639-1 style tag
 *   category     the mixture bucket it counts towards
 *   createdAt    when the text itself was created
 *   version      the revision of that text
 *   fingerprint  content hash, so any edit is detectable
 *
 * Alpha trains only on material it has the right to use. `assertProvenance`
 * refuses a corpus containing any document with an unknown origin or a missing
 * licence, rather than quietly averaging it into a "mostly fine" dataset.
 *
 * Nothing here fetches anything. These are records, not a collector: Alpha does
 * not scrape, and a document only exists here because it was authored in this
 * repository, authored by a user, or imported from material whose licence the
 * operator declared.
 */

import { AlphaValidationError } from "../core/errors";

/** How Alpha is allowed to hold the material. */
export type ProvenanceOrigin =
  /** Written by/for Alpha in this repository. */
  | "authored"
  /** Not subject to copyright, or explicitly dedicated to the public domain. */
  | "public-domain"
  /** Explicitly supplied by the account owner. */
  | "user-owned"
  /** Third-party material under a permissive licence the operator declared. */
  | "licensed";

export const PROVENANCE_ORIGINS: ProvenanceOrigin[] = [
  "authored",
  "public-domain",
  "user-owned",
  "licensed",
];

/** How the bytes actually arrived. Recorded so the route is auditable. */
export type AcquisitionMethod =
  /** Deterministically produced by code in this repository. */
  | "generated-in-repo"
  /** Hand-written in this repository. */
  | "authored-in-repo"
  /** Explicitly supplied by the account owner. */
  | "supplied-by-user"
  /** Imported from a declared public-domain work. */
  | "imported-public-domain"
  /** Imported under a declared permissive licence. */
  | "imported-licensed";

/** Where the text physically came from. Never a scraped website. */
export type AcquisitionLocation =
  | "this-repository"
  | "user-upload"
  | "declared-public-domain-collection"
  | "declared-licensed-dataset";

export type AcquisitionRecord = {
  method: AcquisitionMethod;
  location: AcquisitionLocation;
  /** Epoch milliseconds of acquisition. Passed in, never read from the clock here. */
  acquiredAt: number;
  /** Who did it: "alpha:generated-corpus", "owner:someone", a curator name. */
  collectedBy: string;
  /** Collection date as an ISO string when the source is external. */
  collectedOn?: string;
  /** URL or reference, only for imported material the operator declared. */
  reference?: string;
  note?: string;
};

export const ACQUISITION_LOCATIONS: Record<AcquisitionMethod, AcquisitionLocation[]> = {
  "generated-in-repo": ["this-repository"],
  "authored-in-repo": ["this-repository"],
  "supplied-by-user": ["user-upload"],
  "imported-public-domain": ["declared-public-domain-collection"],
  "imported-licensed": ["declared-licensed-dataset"],
};

/** The declared source a group of documents came from. */
export type ProvenanceSource = {
  id: string;
  title: string;
  origin: ProvenanceOrigin;
  license: string;
  /** Filled in after the corpus is built: how many documents actually landed here. */
  documents: number;
  note?: string;
};

/** Categories Alpha can actually mix. A category with no documents is not claimed. */
export const ALPHA_MIX_CATEGORIES = [
  "general-prose",
  "educational",
  "factual-reference",
  "dialogue",
  "instructions",
  "explanations",
  "structured",
  "multilingual",
] as const;

export type MixCategory = (typeof ALPHA_MIX_CATEGORIES)[number];

/** One document plus everything known about where it came from. */
export type ProvenanceDocument = {
  documentId: string;
  text: string;
  sourceId: string;
  origin: ProvenanceOrigin;
  license: string;
  language: string;
  category: MixCategory;
  acquisition: AcquisitionRecord;
  /** Epoch milliseconds: when this text was created, not when it was imported. */
  createdAt: number;
  /** Revision of the text itself. "1" for a first version. */
  version: string;
  /** Content fingerprint, recomputed on every validation. */
  fingerprint: string;
};

export type ProvenanceCorpus = {
  datasetId: string;
  name: string;
  version: string;
  description: string;
  sources: ProvenanceSource[];
  documents: ProvenanceDocument[];
  createdAt: number;
  previousVersion: string | null;
};

/** FNV-1a over the text, same algorithm family as the dataset fingerprint. */
export function documentFingerprint(text: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return `doc_${hash.toString(16).padStart(8, "0")}`;
}

/** Stable fingerprint over the full provenance record, not just the text. */
export function provenanceFingerprint(document: Omit<ProvenanceDocument, "fingerprint">): string {
  const parts = [
    document.documentId,
    document.sourceId,
    document.origin,
    document.license,
    document.language,
    document.category,
    document.acquisition.method,
    document.acquisition.location,
    document.acquisition.collectedBy,
    String(document.acquisition.acquiredAt),
    String(document.createdAt),
    document.version,
    documentFingerprint(document.text),
  ];
  return documentFingerprint(parts.join("|"));
}

export type CreateProvenanceDocumentInput = {
  documentId: string;
  text: string;
  sourceId: string;
  origin: ProvenanceOrigin;
  license: string;
  language: string;
  category: MixCategory;
  acquisition: Omit<AcquisitionRecord, "location"> & { location?: AcquisitionLocation };
  createdAt: number;
  version?: string;
};

/** Build one provenance record, filling in the only fields that can be derived. */
export function createProvenanceDocument(
  input: CreateProvenanceDocumentInput,
): ProvenanceDocument {
  const allowed = ACQUISITION_LOCATIONS[input.acquisition.method];
  if (!allowed.includes(input.acquisition.location as AcquisitionLocation)) {
    throw new AlphaValidationError(
      "datasets",
      `document "${input.documentId}": acquisition method "${input.acquisition.method}" ` +
        `cannot have location "${String(input.acquisition.location)}"; expected one of ${allowed.join(", ")}`,
      { documentId: input.documentId },
    );
  }
  const base: Omit<ProvenanceDocument, "fingerprint"> = {
    documentId: input.documentId,
    text: input.text,
    sourceId: input.sourceId,
    origin: input.origin,
    license: input.license,
    language: input.language,
    category: input.category,
    acquisition: {
      method: input.acquisition.method,
      location: (input.acquisition.location ?? allowed[0]) as AcquisitionLocation,
      acquiredAt: input.acquisition.acquiredAt,
      collectedBy: input.acquisition.collectedBy,
      ...(input.acquisition.collectedOn ? { collectedOn: input.acquisition.collectedOn } : {}),
      ...(input.acquisition.reference ? { reference: input.acquisition.reference } : {}),
      ...(input.acquisition.note ? { note: input.acquisition.note } : {}),
    },
    createdAt: input.createdAt,
    version: input.version ?? "1",
  };
  return { ...base, fingerprint: provenanceFingerprint(base) };
}

export type ProvenanceIssue = {
  documentId: string;
  problem: string;
};

export type ProvenanceValidation = {
  valid: boolean;
  issues: ProvenanceIssue[];
  /** Documents whose fingerprint no longer matches their text. */
  tampered: string[];
  duplicateDocumentIds: string[];
  duplicateContent: string[];
  /** Categories present with at least one document. Empty means nothing to claim. */
  categoriesPresent: MixCategory[];
  languagesPresent: string[];
};

/**
 * Validate a whole corpus of provenance records.
 *
 * Reports problems; deletes nothing. A caller that wants bad documents dropped
 * says so explicitly through the dataset version's filter settings, and the
 * removal is recorded in the manifest.
 */
export function validateProvenance(
  documents: ProvenanceDocument[],
  sources: ProvenanceSource[],
): ProvenanceValidation {
  const issues: ProvenanceIssue[] = [];
  const tampered: string[] = [];
  const sourceIds = new Set(sources.map((s) => s.id));

  const seenIds = new Map<string, number>();
  /** content fingerprint -> every document id that carries it. */
  const contentOwners = new Map<string, string[]>();
  const repeatedIds = new Set<string>();
  const repeatedContent = new Set<string>();

  for (const document of documents) {
    const fail = (problem: string) => issues.push({ documentId: document.documentId, problem });

    if (!document.documentId) fail("missing documentId");
    if (typeof document.text !== "string" || document.text.trim().length === 0) {
      fail("document text is empty");
    }
    if (!sourceIds.has(document.sourceId)) {
      fail(`sourceId "${document.sourceId}" is not a declared source of this corpus`);
    }
    if (!document.license || document.license.trim().length === 0) {
      fail("no licence declared; Alpha will not train on this document");
    }
    if (!PROVENANCE_ORIGINS.includes(document.origin)) {
      fail(`origin "${String(document.origin)}" is not one Alpha recognises`);
    }
    if (!document.language || document.language.trim().length === 0) {
      fail("no language declared");
    }
    if (!ALPHA_MIX_CATEGORIES.includes(document.category)) {
      fail(`category "${String(document.category)}" is not an Alpha mixture category`);
    }
    if (!Number.isFinite(document.createdAt) || document.createdAt <= 0) {
      fail("createdAt must be a positive epoch timestamp");
    }
    const recomputed = provenanceFingerprint(document);
    if (recomputed !== document.fingerprint) {
      tampered.push(document.documentId);
      fail("fingerprint does not match its own contents — the text was edited after the record was made");
    }

    if (seenIds.has(document.documentId)) {
      repeatedIds.add(document.documentId);
      issues.push({
        documentId: document.documentId,
        problem: `documentId repeats an earlier document (index ${seenIds.get(document.documentId)})`,
      });
    } else {
      seenIds.set(document.documentId, documents.indexOf(document));
    }

    const content = documentFingerprint(document.text);
    const owners = contentOwners.get(content) ?? [];
    owners.push(document.documentId);
    contentOwners.set(content, owners);
    if (owners.length > 1) {
      repeatedContent.add(document.documentId);
      issues.push({
        documentId: document.documentId,
        problem: `identical text to document "${owners[0]}"`,
      });
    }
  }

  return {
    valid: issues.length === 0,
    issues,
    tampered,
    duplicateDocumentIds: [...repeatedIds].sort(),
    duplicateContent: [...repeatedContent].sort(),
    categoriesPresent: ALPHA_MIX_CATEGORIES.filter((category) =>
      documents.some((d) => d.category === category),
    ),
    languagesPresent: [...new Set(documents.map((d) => d.language))].sort(),
  };
}

/** Throwing form. Call this before a corpus is allowed near a training run. */
export function assertProvenance(
  documents: ProvenanceDocument[],
  sources: ProvenanceSource[],
): ProvenanceValidation {
  const validation = validateProvenance(documents, sources);
  if (!validation.valid) {
    throw new AlphaValidationError(
      "datasets",
      `corpus provenance is incomplete: ${validation.issues
        .slice(0, 4)
        .map((i) => `${i.documentId}: ${i.problem}`)
        .join("; ")}`,
      { issues: validation.issues.slice(0, 10) },
    );
  }
  return validation;
}

/** Count documents per category. Used by the mixture report and the capability report. */
export function categoryCounts(documents: ProvenanceDocument[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const category of ALPHA_MIX_CATEGORIES) counts[category] = 0;
  for (const document of documents) {
    counts[document.category] = (counts[document.category] ?? 0) + 1;
  }
  return counts;
}

/** Count documents per language. */
export function languageCounts(documents: ProvenanceDocument[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const document of documents) {
    counts[document.language] = (counts[document.language] ?? 0) + 1;
  }
  return Object.fromEntries(Object.entries(counts).sort((a, b) => b[1] - a[1]));
}

/** Count documents and characters per source. */
export function sourceCounts(
  documents: ProvenanceDocument[],
): Array<{ sourceId: string; documents: number; characters: number }> {
  const rows = new Map<string, { documents: number; characters: number }>();
  for (const document of documents) {
    const row = rows.get(document.sourceId) ?? { documents: 0, characters: 0 };
    row.documents += 1;
    row.characters += document.text.length;
    rows.set(document.sourceId, row);
  }
  return [...rows.entries()]
    .map(([sourceId, row]) => ({ sourceId, ...row }))
    .sort((a, b) => b.characters - a.characters);
}

/** Attach the measured counts to each declared source. */
export function withSourceCounts(
  sources: ProvenanceSource[],
  documents: ProvenanceDocument[],
): ProvenanceSource[] {
  const counts = new Map(sourceCounts(documents).map((row) => [row.sourceId, row]));
  return sources.map((source) => ({ ...source, documents: counts.get(source.id)?.documents ?? 0 }));
}

/** One-line provenance summary for a report. */
export function describeProvenance(documents: ProvenanceDocument[], sources: ProvenanceSource[]): string {
  const characters = documents.reduce((sum, d) => sum + d.text.length, 0);
  const licences = [...new Set(documents.map((d) => d.license))].sort();
  const languages = [...new Set(documents.map((d) => d.language))].sort();
  const categories = categoryCounts(documents);
  const present = Object.entries(categories)
    .filter(([, count]) => count > 0)
    .map(([name, count]) => `${name} ${count}`);
  return (
    `${documents.length} documents · ${characters.toLocaleString()} chars · ` +
    `sources ${sources.map((s) => `${s.id} (${s.license}, ${s.origin})`).join(" + ")} · ` +
    `languages ${languages.join(", ")} · licences ${licences.join(", ")} · categories ${present.join(", ")}`
  );
}
