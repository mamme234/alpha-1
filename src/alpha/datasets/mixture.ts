/**
 * Dataset mixtures.
 *
 * A mixture is a decision about what a model should see more of. Writing it down
 * explicitly, per category, with the requested weights recorded next to what
 * was actually achieved, is the difference between "we trained on a mixture" and
 * a claim that can be checked.
 *
 * This module combines several provenance corpora — each with its own sources,
 * licences and categories — into one mixture under explicit weights or quotas.
 *
 * Two things it deliberately does not do:
 *
 *   1. It does not invent categories. If a caller asks for a category no corpus
 *      supplies, the mixture reports it as unsatisfiable rather than quietly
 *      substituting another.
 *   2. It does not over-sample indefinitely. `oversampleLimit` caps how many
 *      times a component's documents may be repeated, so a tiny category cannot
 *      quietly dominate the token stream through repetition.
 *
 * Determinism: the selection is a seeded shuffle within each component followed
 * by a fixed allocation, so the same inputs always produce the same mixture and
 * therefore the same mixture fingerprint.
 *
 * The output is interleaved rather than concatenated, so a training run sees a
 * mixture of categories from its very first step instead of one category for the
 * first thousand tokens and another category afterwards.
 */

import { AlphaValidationError } from "../core/errors";
import type { MixCategory, ProvenanceCorpus, ProvenanceDocument } from "./provenance";
import { ALPHA_MIX_CATEGORIES } from "./provenance";

/** One input to a mixture. */
export type MixtureComponent = {
  /** Stable id of the component, distinct from the corpus it draws from. */
  id: string;
  /** The corpus this component draws from. */
  corpus: ProvenanceCorpus;
  /** Optional: restrict this component to certain categories of its corpus. */
  categories?: MixCategory[];
  /**
   * Relative weight, normalised across components, so `2` and `1` means the
   * first contributes twice as many documents as the second. Ignored when
   * `quota` is given.
   */
  weight?: number;
  /** Absolute document count. Takes precedence over `weight`. */
  quota?: number;
};

/** What was asked for, and what was actually produced. */
export type MixtureRecord = {
  componentId: string;
  corpusId: string;
  corpusVersion: string;
  license: string;
  origin: string;
  /** Categories this component actually had documents for. */
  categories: MixCategory[];
  requestedWeight: number;
  requestedQuota: number | null;
  /** Share of the requested mixture this component was intended to hold. */
  intendedShare: number;
  /** Unique documents available to it, before any oversampling. */
  available: number;
  /** Documents it contributed, counting repeats. */
  taken: number;
  /** Share of the achieved mixture. */
  achievedShare: number;
  /**
   * taken / (intendedShare * totalDocuments). 1.0 means the component got
   * exactly what it asked for; below 1.0 means it was short.
   */
  fulfilment: number;
};

export type MixtureResult = {
  documents: ProvenanceDocument[];
  records: MixtureRecord[];
  /** Categories the caller expected that no component could supply. */
  unsatisfiableCategories: MixCategory[];
  /** Documents requested but not available anywhere, after oversampling caps. */
  shortfall: number;
  requested: { totalDocuments: number; weights: Record<string, number>; quotas: Record<string, number> };
  achieved: {
    totalDocuments: number;
    /** Document count and share per category. */
    categories: Record<string, { documents: number; share: number }>;
    /** Document count and share per language. */
    languages: Record<string, { documents: number; share: number }>;
  };
  fingerprint: string;
};

export type MixtureInput = {
  components: MixtureComponent[];
  /** Total documents the mixture should contain. */
  totalDocuments: number;
  seed?: number;
  /** Maximum repeats of any one document. Default 3. */
  oversampleLimit?: number;
  /** Categories the caller expects; checked, never invented. */
  expectedCategories?: MixCategory[];
};

function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/** Deterministic shuffle, so a mixture is reproducible from its seed. */
function shuffled<T>(items: T[], rng: () => number): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** Stable fingerprint over the selected documents and their provenance. */
function mixtureFingerprint(documents: ProvenanceDocument[]): string {
  let hash = 0x811c9dc5;
  const feed = (text: string) => {
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
  };
  feed(String(documents.length));
  for (const document of documents) {
    feed(document.fingerprint);
    feed("|");
    feed(document.category);
    feed("\u0001");
  }
  return `mix_${hash.toString(16).padStart(8, "0")}`;
}

/** Licence/origin summary of a component's corpus, for the mixture record. */
function componentLicences(corpus: ProvenanceCorpus): { license: string; origin: string } {
  const licences = [...new Set(corpus.documents.map((d) => d.license))].sort();
  const origins = [...new Set(corpus.documents.map((d) => d.origin))].sort();
  return { license: licences.join("+") || "none", origin: origins.join("+") || "unknown" };
}

/**
 * Build a mixture.
 *
 * Allocation:
 *   - quota components are served exactly, up to their availability.
 *   - the remainder is split across weight components by the largest-remainder
 *     method, so integer rounding never costs a component more than one document.
 *   - anything still unfilled is redistributed in weight order until every
 *     component is at its oversampling cap.
 *   - output is interleaved by target, so categories are mixed from step one.
 */
export function buildMixture(input: MixtureInput): MixtureResult {
  const { components } = input;
  if (components.length === 0) {
    throw new AlphaValidationError("datasets", "a mixture needs at least one component");
  }
  if (!Number.isInteger(input.totalDocuments) || input.totalDocuments < 1) {
    throw new AlphaValidationError("datasets", "totalDocuments must be a positive integer");
  }
  const ids = new Set<string>();
  for (const component of components) {
    if (ids.has(component.id)) {
      throw new AlphaValidationError(
        "datasets",
        `mixture component "${component.id}" appears more than once`,
        { componentId: component.id },
      );
    }
    ids.add(component.id);
    for (const category of component.categories ?? []) {
      if (!ALPHA_MIX_CATEGORIES.includes(category)) {
        throw new AlphaValidationError(
          "datasets",
          `"${component.id}" names category "${category}", which Alpha does not define`,
          { componentId: component.id, category },
        );
      }
    }
  }

  const oversampleLimit = Math.max(1, input.oversampleLimit ?? 3);
  const rng = lcg(input.seed ?? 20260101);
  const total = input.totalDocuments;

  type Pool = {
    component: MixtureComponent;
    ordered: ProvenanceDocument[];
    categories: MixCategory[];
    /** How many times this pool may be drawn from, counting repeats. */
    capacity: number;
    target: number;
  };

  const pools: Pool[] = components.map((component) => {
    const filtered =
      component.categories && component.categories.length > 0
        ? component.corpus.documents.filter((d) => component.categories!.includes(d.category))
        : component.corpus.documents;
    const ordered = shuffled(filtered, rng);
    if (ordered.length === 0) {
      throw new AlphaValidationError(
        "datasets",
        `mixture component "${component.id}" has no documents to draw from ` +
          `(corpus ${component.corpus.name}@${component.corpus.version}` +
          `${component.categories ? `, categories ${component.categories.join(", ")}` : ""})`,
        { componentId: component.id, corpusId: component.corpus.datasetId },
      );
    }
    return {
      component,
      ordered,
      categories: [...new Set(filtered.map((d) => d.category))].sort() as MixCategory[],
      capacity: ordered.length * oversampleLimit,
      target: 0,
    };
  });

  // Quota components first.
  let quotaUsed = 0;
  const weighted: Pool[] = [];
  for (const pool of pools) {
    const quota = pool.component.quota;
    if (quota === undefined) {
      weighted.push(pool);
      continue;
    }
    pool.target = Math.min(Math.max(0, quota), pool.capacity);
    quotaUsed += pool.target;
  }

  // Largest-remainder split of what is left across the weight components.
  const remaining = Math.max(0, total - quotaUsed);
  const weightSum = weighted.reduce((sum, pool) => sum + (pool.component.weight ?? 0), 0);
  if (remaining > 0 && weighted.length > 0) {
    if (weightSum <= 0) {
      // No weights at all: share equally rather than giving everything to one.
      const share = Math.floor(remaining / weighted.length);
      weighted.forEach((pool) => {
        pool.target += share;
      });
      let leftover = remaining - share * weighted.length;
      for (let i = 0; i < leftover; i++) weighted[i % weighted.length].target += 1;
    } else {
      const exact = weighted.map((pool) => (remaining * (pool.component.weight ?? 0)) / weightSum);
      const floors = exact.map((value) => Math.floor(value));
      let assigned = floors.reduce((s, v) => s + v, 0);
      const order = exact
        .map((value, index) => ({ index, remainder: value - Math.floor(value) }))
        .sort((a, b) => b.remainder - a.remainder || a.index - b.index);
      for (const { index } of order) {
        if (assigned >= remaining) break;
        floors[index] += 1;
        assigned += 1;
      }
      weighted.forEach((pool, index) => {
        pool.target += Math.min(floors[index], pool.capacity - pool.target);
      });
    }
  }

  // Redistribute anything still unfilled, in weight order, up to each cap.
  const allocated = () => pools.reduce((sum, pool) => sum + pool.target, 0);
  let guard = 0;
  while (allocated() < total && guard < 100_000) {
    guard += 1;
    let progressed = false;
    for (const pool of pools) {
      if (allocated() >= total) break;
      if (pool.target >= pool.capacity) continue;
      pool.target += 1;
      progressed = true;
    }
    if (!progressed) break;
  }

  // Interleave: walk the components round-robin until every target is met.
  const cursors = new Map<string, number>();
  for (const pool of pools) cursors.set(pool.component.id, 0);
  const documents: ProvenanceDocument[] = [];
  let round = 0;
  while (documents.length < total) {
    let progressed = false;
    for (const pool of pools) {
      if (documents.length >= total) break;
      const cursor = cursors.get(pool.component.id)!;
      if (cursor >= pool.target) continue;
      documents.push(pool.ordered[cursor % pool.ordered.length]);
      cursors.set(pool.component.id, cursor + 1);
      progressed = true;
    }
    if (!progressed) break;
    round += 1;
    if (round > total + pools.length + 1) break;
  }

  const achievedTotal = documents.length;
  const categoryCounts = new Map<string, number>();
  const languageCounts = new Map<string, number>();
  for (const document of documents) {
    categoryCounts.set(document.category, (categoryCounts.get(document.category) ?? 0) + 1);
    languageCounts.set(document.language, (languageCounts.get(document.language) ?? 0) + 1);
  }
  const categories: Record<string, { documents: number; share: number }> = {};
  for (const category of ALPHA_MIX_CATEGORIES) {
    const count = categoryCounts.get(category) ?? 0;
    categories[category] = {
      documents: count,
      share: achievedTotal === 0 ? 0 : count / achievedTotal,
    };
  }
  const languages: Record<string, { documents: number; share: number }> = {};
  for (const [language, count] of [...languageCounts.entries()].sort((a, b) => b[1] - a[1])) {
    languages[language] = { documents: count, share: achievedTotal === 0 ? 0 : count / achievedTotal };
  }

  const records: MixtureRecord[] = pools.map((pool) => {
    const { license, origin } = componentLicences(pool.component.corpus);
    const intendedCount = pool.component.quota ?? Math.round(((pool.component.weight ?? 0) / Math.max(1e-9, weightSum)) * remaining);
    const intendedShare = total === 0 ? 0 : intendedCount / total;
    const achievedShare = achievedTotal === 0 ? 0 : pool.target / achievedTotal;
    return {
      componentId: pool.component.id,
      corpusId: pool.component.corpus.datasetId,
      corpusVersion: pool.component.corpus.version,
      license,
      origin,
      categories: pool.categories,
      requestedWeight: pool.component.weight ?? 0,
      requestedQuota: pool.component.quota ?? null,
      intendedShare,
      available: pool.ordered.length,
      taken: pool.target,
      achievedShare,
      fulfilment: intendedShare === 0 ? (pool.target === 0 ? 1 : 1) : achievedShare / intendedShare,
    };
  });

  const unsatisfiable = (input.expectedCategories ?? []).filter(
    (category) => (categories[category]?.documents ?? 0) === 0,
  );

  return {
    documents,
    records,
    unsatisfiableCategories: unsatisfiable,
    shortfall: Math.max(0, total - achievedTotal),
    requested: {
      totalDocuments: total,
      weights: Object.fromEntries(components.map((c) => [c.id, c.weight ?? 0])),
      quotas: Object.fromEntries(components.map((c) => [c.id, c.quota ?? 0])),
    },
    achieved: {
      totalDocuments: achievedTotal,
      categories,
      languages,
    },
    fingerprint: mixtureFingerprint(documents),
  };
}

/** One-line mixture summary for a report. */
export function describeMixture(result: MixtureResult): string {
  const parts = result.records.map(
    (r) => `${r.componentId} ${r.taken}/${r.available} (${(r.achievedShare * 100).toFixed(1)}%)`,
  );
  const categories = Object.entries(result.achieved.categories)
    .filter(([, value]) => value.documents > 0)
    .map(([name, value]) => `${name} ${value.documents}`);
  const lines = [
    `${result.achieved.totalDocuments}/${result.requested.totalDocuments} documents · ${result.fingerprint} · components ${parts.join(", ")} · categories ${categories.join(", ")}`,
  ];
  if (result.unsatisfiableCategories.length > 0) {
    lines.push(`  UNSATISFIABLE categories (requested, none available): ${result.unsatisfiableCategories.join(", ")}`);
  }
  if (result.shortfall > 0) {
    lines.push(`  SHORTFALL: ${result.shortfall} document(s) could not be filled at the oversampling limit`);
  }
  return lines.join("\n");
}
