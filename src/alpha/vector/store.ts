/**
 * Alpha Vector Store — Alpha's own vector layer.
 *
 * In-memory, deterministic, and serialisable. There is no hosted vector
 * database behind it: collections are namespaces, vectors carry metadata,
 * search is an exact similarity scan, and the whole store can be exported to
 * and imported from JSON so it can be persisted anywhere (the workspace stores
 * it in Alpha's own Convex tables).
 *
 * An exact scan is the right choice at this scale and is honest about its cost:
 * search is O(records x dimension).
 */

import { AlphaValidationError } from "../core/errors";
import { cosineSimilarity, dotProduct, euclideanDistance } from "../embeddings/embedder";

export type VectorMetric = "cosine" | "dot" | "euclidean";

export type VectorRecord<TMetadata = Record<string, unknown>> = {
  id: string;
  collection: string;
  vector: number[];
  text: string;
  metadata: TMetadata;
  /** Optional grouping key, e.g. the document a chunk came from. */
  sourceId?: string;
  createdAt: number;
  updatedAt: number;
};

export type VectorCollectionInfo = {
  name: string;
  dimension: number;
  metric: VectorMetric;
  description: string;
  createdAt: number;
  recordCount: number;
};

export type VectorSearchHit = {
  record: VectorRecord;
  score: number;
  rank: number;
};

export type VectorSearchRequest = {
  collection: string;
  vector: number[];
  topK?: number;
  /** Drop hits below this score (cosine/dot) or above this distance (euclidean). */
  minScore?: number;
  filter?: (record: VectorRecord) => boolean;
};

export type VectorStoreSnapshot = {
  version: string;
  collections: VectorCollectionInfo[];
  records: VectorRecord[];
  exportedAt: number;
};

function scoreFor(metric: VectorMetric, query: number[], vector: number[]): number {
  switch (metric) {
    case "cosine":
      return cosineSimilarity(query, vector);
    case "dot":
      return dotProduct(query, vector);
    case "euclidean":
      // Convert distance to a descending "higher is better" score.
      return 1 / (1 + euclideanDistance(query, vector));
  }
}

export class AlphaVectorStore {
  private collections = new Map<string, VectorCollectionInfo>();
  private records = new Map<string, VectorRecord>();

  createCollection(input: {
    name: string;
    dimension: number;
    metric?: VectorMetric;
    description?: string;
  }): VectorCollectionInfo {
    if (!input.name) throw new AlphaValidationError("vector", "collection name is required");
    if (input.dimension < 1) {
      throw new AlphaValidationError("vector", "collection dimension must be positive");
    }
    const existing = this.collections.get(input.name);
    if (existing) {
      if (existing.dimension !== input.dimension) {
        throw new AlphaValidationError(
          "vector",
          `collection "${input.name}" already exists with dimension ${existing.dimension}`,
        );
      }
      return existing;
    }
    const info: VectorCollectionInfo = {
      name: input.name,
      dimension: input.dimension,
      metric: input.metric ?? "cosine",
      description: input.description ?? "",
      createdAt: Date.now(),
      recordCount: 0,
    };
    this.collections.set(info.name, info);
    return info;
  }

  ensureCollection(name: string, dimension: number, metric: VectorMetric = "cosine"): VectorCollectionInfo {
    return this.createCollection({ name, dimension, metric });
  }

  listCollections(): VectorCollectionInfo[] {
    return Array.from(this.collections.values()).map((info) => ({
      ...info,
      recordCount: this.count(info.name),
    }));
  }

  getCollection(name: string): VectorCollectionInfo | null {
    const info = this.collections.get(name);
    if (!info) return null;
    return { ...info, recordCount: this.count(name) };
  }

  deleteCollection(name: string): number {
    const info = this.collections.get(name);
    if (!info) return 0;
    let removed = 0;
    for (const [id, record] of this.records) {
      if (record.collection === name) {
        this.records.delete(id);
        removed++;
      }
    }
    this.collections.delete(name);
    return removed;
  }

  insert(input: {
    id: string;
    collection: string;
    vector: number[];
    text: string;
    metadata?: Record<string, unknown>;
    sourceId?: string;
  }): VectorRecord {
    const info = this.collections.get(input.collection);
    if (!info) {
      throw new AlphaValidationError(
        "vector",
        `collection "${input.collection}" does not exist; create it first`,
      );
    }
    if (input.vector.length !== info.dimension) {
      throw new AlphaValidationError(
        "vector",
        `vector has ${input.vector.length} dimensions but collection "${input.collection}" expects ${info.dimension}`,
      );
    }
    if (this.records.has(input.id)) {
      throw new AlphaValidationError("vector", `record "${input.id}" already exists`);
    }
    const now = Date.now();
    const record: VectorRecord = {
      id: input.id,
      collection: input.collection,
      vector: [...input.vector],
      text: input.text,
      metadata: input.metadata ?? {},
      sourceId: input.sourceId,
      createdAt: now,
      updatedAt: now,
    };
    this.records.set(record.id, record);
    return record;
  }

  /** Insert or replace. */
  upsert(input: Parameters<AlphaVectorStore["insert"]>[0]): VectorRecord {
    if (this.records.has(input.id)) return this.update(input.id, input);
    return this.insert(input);
  }

  update(
    id: string,
    patch: Partial<Pick<VectorRecord, "vector" | "text" | "metadata" | "sourceId">>,
  ): VectorRecord {
    const record = this.records.get(id);
    if (!record) throw new AlphaValidationError("vector", `record "${id}" not found`);
    const info = this.collections.get(record.collection);
    if (patch.vector) {
      if (!info) throw new AlphaValidationError("vector", "collection missing for record");
      if (patch.vector.length !== info.dimension) {
        throw new AlphaValidationError(
          "vector",
          `vector has ${patch.vector.length} dimensions but collection expects ${info.dimension}`,
        );
      }
      record.vector = [...patch.vector];
    }
    if (patch.text !== undefined) record.text = patch.text;
    if (patch.metadata !== undefined) record.metadata = patch.metadata;
    if (patch.sourceId !== undefined) record.sourceId = patch.sourceId;
    record.updatedAt = Date.now();
    return record;
  }

  delete(id: string): boolean {
    return this.records.delete(id);
  }

  deleteBySource(collection: string, sourceId: string): number {
    let removed = 0;
    for (const [id, record] of this.records) {
      if (record.collection === collection && record.sourceId === sourceId) {
        this.records.delete(id);
        removed++;
      }
    }
    return removed;
  }

  get(id: string): VectorRecord | null {
    return this.records.get(id) ?? null;
  }

  count(collection?: string): number {
    if (!collection) return this.records.size;
    let total = 0;
    for (const record of this.records.values()) if (record.collection === collection) total++;
    return total;
  }

  list(collection?: string): VectorRecord[] {
    const out: VectorRecord[] = [];
    for (const record of this.records.values()) {
      if (!collection || record.collection === collection) out.push(record);
    }
    return out.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  }

  /** Exact similarity search with deterministic tie-breaking on record id. */
  search(request: VectorSearchRequest): VectorSearchHit[] {
    const info = this.collections.get(request.collection);
    if (!info) {
      throw new AlphaValidationError("vector", `collection "${request.collection}" does not exist`);
    }
    if (request.vector.length !== info.dimension) {
      throw new AlphaValidationError(
        "vector",
        `query has ${request.vector.length} dimensions but collection expects ${info.dimension}`,
      );
    }
    const scored: { record: VectorRecord; score: number }[] = [];
    for (const record of this.records.values()) {
      if (record.collection !== request.collection) continue;
      if (request.filter && !request.filter(record)) continue;
      const score = scoreFor(info.metric, request.vector, record.vector);
      if (request.minScore !== undefined && score < request.minScore) continue;
      scored.push({ record, score });
    }
    scored.sort((a, b) => (b.score === a.score ? (a.record.id < b.record.id ? -1 : 1) : b.score - a.score));
    const topK = request.topK ?? 5;
    return scored.slice(0, topK).map((entry, index) => ({
      record: entry.record,
      score: entry.score,
      rank: index + 1,
    }));
  }

  exportSnapshot(): VectorStoreSnapshot {
    return {
      version: "0.1.0",
      collections: this.listCollections(),
      records: this.list(),
      exportedAt: Date.now(),
    };
  }

  importSnapshot(snapshot: VectorStoreSnapshot): { collections: number; records: number } {
    for (const info of snapshot.collections) {
      if (!this.collections.has(info.name)) {
        this.collections.set(info.name, { ...info, recordCount: 0 });
      }
    }
    let records = 0;
    for (const record of snapshot.records) {
      if (!this.collections.has(record.collection)) {
        this.collections.set(record.collection, {
          name: record.collection,
          dimension: record.vector.length,
          metric: "cosine",
          description: "recovered from snapshot",
          createdAt: record.createdAt,
          recordCount: 0,
        });
      }
      this.records.set(record.id, { ...record });
      records++;
    }
    return { collections: snapshot.collections.length, records };
  }

  clear(): void {
    this.collections.clear();
    this.records.clear();
  }
}
