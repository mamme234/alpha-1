/**
 * Alpha Memory — Alpha's own memory subsystem.
 *
 * Three scopes, deliberately different in lifetime:
 *   - `conversation` — one exchange; discarded when the turn ends
 *   - `session`      — one workspace session; discarded when the session ends
 *   - `long-term`    — persists, and is only written with explicit approval
 *
 * Relevance is computed locally: embedding similarity, decayed by age, boosted
 * by importance and by how often the memory has been useful. No external
 * service decides what Alpha remembers.
 */

import { alphaId, type AlphaStatus } from "../core/types";
import { AlphaPermissionError, AlphaValidationError } from "../core/errors";
import { cosineSimilarity, type AlphaEmbedder } from "../embeddings/embedder";

export type MemoryScope = "conversation" | "session" | "long-term";

export type MemoryRecord = {
  id: string;
  scope: MemoryScope;
  /** Session this memory belongs to (conversation/session scopes). */
  sessionId: string | null;
  /** Short machine-readable key, e.g. "user.name". */
  key: string;
  content: string;
  embedding: number[];
  tags: string[];
  /** 0..1 — caller-supplied importance. */
  importance: number;
  /** Long-term memories must be approved before they can be recalled. */
  approved: boolean;
  source: "agent" | "user" | "system" | "tool";
  createdAt: number;
  updatedAt: number;
  lastAccessedAt: number;
  accessCount: number;
};

export type MemoryWriteInput = {
  scope: MemoryScope;
  key: string;
  content: string;
  sessionId?: string | null;
  tags?: string[];
  importance?: number;
  source?: MemoryRecord["source"];
  /** Required for `long-term` writes. */
  approved?: boolean;
};

export type MemoryRetrieval = {
  record: MemoryRecord;
  /** Weighted relevance: similarity x decay x importance x usage. */
  relevance: number;
  similarity: number;
  ageHours: number;
};

export type MemoryStoreConfig = {
  /** Half-life of the recency decay, in hours. */
  recencyHalfLifeHours: number;
  /** Weight of embedding similarity in the relevance score. */
  similarityWeight: number;
  /** Weight of importance in the relevance score. */
  importanceWeight: number;
  /** Weight of usage (log-scaled access count). */
  usageWeight: number;
  /** Maximum memories retained per scope. */
  maxPerScope: number;
};

export const DEFAULT_MEMORY_CONFIG: MemoryStoreConfig = {
  recencyHalfLifeHours: 72,
  similarityWeight: 0.6,
  importanceWeight: 0.25,
  usageWeight: 0.15,
  maxPerScope: 512,
};

export type MemoryStoreSnapshot = {
  version: string;
  records: MemoryRecord[];
  exportedAt: number;
};

export class AlphaMemoryStore {
  private records = new Map<string, MemoryRecord>();
  private readonly embedder: AlphaEmbedder;
  readonly config: MemoryStoreConfig;
  readonly status: AlphaStatus = "ready";

  constructor(options: { embedder: AlphaEmbedder; config?: Partial<MemoryStoreConfig> }) {
    this.embedder = options.embedder;
    this.config = { ...DEFAULT_MEMORY_CONFIG, ...options.config };
  }

  /**
   * Write a memory. Long-term writes require `approved: true` — an agent cannot
   * quietly remember something across sessions without a human saying yes.
   */
  write(input: MemoryWriteInput): MemoryRecord {
    if (!input.key.trim()) {
      throw new AlphaValidationError("memory", "a memory needs a key");
    }
    if (input.scope === "long-term" && input.approved !== true) {
      throw new AlphaPermissionError(
        "memory",
        "writing long-term memory requires explicit user approval",
        { key: input.key, scope: input.scope },
      );
    }
    if ((input.scope === "conversation" || input.scope === "session") && !input.sessionId) {
      throw new AlphaValidationError(
        "memory",
        `scope "${input.scope}" requires a sessionId`,
      );
    }
    const embedded = this.embedder.embed(input.content);
    const now = Date.now();
    const existing = this.findByKey(input.scope, input.key, input.sessionId ?? null);
    if (existing) {
      existing.content = input.content;
      existing.embedding = embedded.vector;
      existing.tags = input.tags ?? existing.tags;
      existing.importance = input.importance ?? existing.importance;
      existing.updatedAt = now;
      existing.approved = existing.approved || input.approved === true;
      return existing;
    }
    const record: MemoryRecord = {
      id: alphaId("mem"),
      scope: input.scope,
      sessionId: input.sessionId ?? null,
      key: input.key,
      content: input.content,
      embedding: embedded.vector,
      tags: input.tags ?? [],
      importance: input.importance ?? 0.5,
      approved: input.scope === "long-term" ? true : input.approved ?? true,
      source: input.source ?? "system",
      createdAt: now,
      updatedAt: now,
      lastAccessedAt: now,
      accessCount: 0,
    };
    this.records.set(record.id, record);
    this.enforceLimit(input.scope);
    return record;
  }

  private findByKey(scope: MemoryScope, key: string, sessionId: string | null): MemoryRecord | null {
    for (const record of this.records.values()) {
      if (record.scope === scope && record.key === key && record.sessionId === sessionId) return record;
    }
    return null;
  }

  private enforceLimit(scope: MemoryScope): void {
    const scoped = [...this.records.values()]
      .filter((record) => record.scope === scope)
      .sort((a, b) => a.updatedAt - b.updatedAt);
    while (scoped.length > this.config.maxPerScope) {
      const oldest = scoped.shift();
      if (oldest) this.records.delete(oldest.id);
    }
  }

  get(id: string): MemoryRecord | null {
    return this.records.get(id) ?? null;
  }

  list(options: { scope?: MemoryScope; sessionId?: string } = {}): MemoryRecord[] {
    return [...this.records.values()]
      .filter((record) => (options.scope ? record.scope === options.scope : true))
      .filter((record) =>
        options.sessionId ? record.sessionId === options.sessionId || record.sessionId === null : true,
      )
      .sort((a, b) => b.updatedAt - a.updatedAt);
  }

  /**
   * Relevance scoring, fully local and inspectable:
   *   relevance = similarityWeight * cosine
   *             + importanceWeight * importance
   *             + usageWeight * log1p(accessCount)/log1p(10)
   *   then multiplied by a 2^(-age/halfLife) recency factor.
   */
  score(record: MemoryRecord, queryVector: number[], now = Date.now()): MemoryRetrieval {
    const similarity = cosineSimilarity(queryVector, record.embedding);
    const ageHours = (now - record.updatedAt) / 3_600_000;
    const decay = 2 ** (-ageHours / this.config.recencyHalfLifeHours);
    const usage = Math.log1p(record.accessCount) / Math.log1p(10);
    const weighted =
      this.config.similarityWeight * similarity +
      this.config.importanceWeight * record.importance +
      this.config.usageWeight * Math.min(1, usage);
    return {
      record,
      relevance: Number((weighted * decay).toFixed(6)),
      similarity: Number(similarity.toFixed(6)),
      ageHours: Number(ageHours.toFixed(3)),
    };
  }

  /** Retrieve memories relevant to a query, ranked by the score above. */
  retrieve(
    query: string,
    options: {
      scopes?: MemoryScope[];
      sessionId?: string | null;
      topK?: number;
      minRelevance?: number;
      includeUnapproved?: boolean;
      tags?: string[];
    } = {},
  ): MemoryRetrieval[] {
    const scopes = options.scopes ?? ["conversation", "session", "long-term"];
    const queryVector = this.embedder.embedQuery(query).vector;
    const now = Date.now();
    const results: MemoryRetrieval[] = [];
    for (const record of this.records.values()) {
      if (!scopes.includes(record.scope)) continue;
      if (!options.includeUnapproved && record.scope === "long-term" && !record.approved) continue;
      if (options.sessionId && record.scope !== "long-term" && record.sessionId !== options.sessionId) {
        continue;
      }
      if (options.tags && options.tags.length > 0) {
        if (!options.tags.some((tag) => record.tags.includes(tag))) continue;
      }
      const scored = this.score(record, queryVector, now);
      if (options.minRelevance !== undefined && scored.relevance < options.minRelevance) continue;
      results.push(scored);
    }
    results.sort((a, b) =>
      b.relevance === a.relevance ? (a.record.id < b.record.id ? -1 : 1) : b.relevance - a.relevance,
    );
    const top = results.slice(0, options.topK ?? 5);
    for (const entry of top) {
      entry.record.accessCount++;
      entry.record.lastAccessedAt = now;
    }
    return top;
  }

  /** Context block for the model: memory as data, with keys for traceability. */
  buildContext(entries: MemoryRetrieval[], maxCharacters = 1200): string {
    const lines: string[] = [];
    let used = 0;
    for (const entry of entries) {
      const line = `- ${entry.record.key}: ${entry.record.content}`;
      if (used + line.length > maxCharacters) break;
      lines.push(line);
      used += line.length;
    }
    return lines.join("\n");
  }

  update(
    id: string,
    patch: Partial<Pick<MemoryRecord, "content" | "importance" | "tags" | "approved" | "key">>,
  ): MemoryRecord {
    const record = this.records.get(id);
    if (!record) throw new AlphaValidationError("memory", `memory "${id}" not found`);
    if (patch.key !== undefined) record.key = patch.key;
    if (patch.content !== undefined) {
      record.content = patch.content;
      record.embedding = this.embedder.embed(patch.content).vector;
    }
    if (patch.importance !== undefined) record.importance = patch.importance;
    if (patch.tags !== undefined) record.tags = patch.tags;
    if (patch.approved !== undefined) record.approved = patch.approved;
    record.updatedAt = Date.now();
    return record;
  }

  /** Delete a single memory. Deletion is immediate and irreversible by design. */
  forget(id: string): boolean {
    return this.records.delete(id);
  }

  /** Drop every memory in a session (conversation/session scopes). */
  forgetSession(sessionId: string): number {
    let removed = 0;
    for (const [id, record] of this.records) {
      if (record.sessionId === sessionId) {
        this.records.delete(id);
        removed++;
      }
    }
    return removed;
  }

  /** Drop every memory in a scope. */
  forgetScope(scope: MemoryScope): number {
    let removed = 0;
    for (const [id, record] of this.records) {
      if (record.scope === scope) {
        this.records.delete(id);
        removed++;
      }
    }
    return removed;
  }

  stats(): Record<MemoryScope, number> {
    const counts: Record<MemoryScope, number> = { conversation: 0, session: 0, "long-term": 0 };
    for (const record of this.records.values()) counts[record.scope]++;
    return counts;
  }

  exportSnapshot(): MemoryStoreSnapshot {
    return {
      version: "0.1.0",
      records: [...this.records.values()].map((record) => ({ ...record })),
      exportedAt: Date.now(),
    };
  }

  importSnapshot(snapshot: MemoryStoreSnapshot): number {
    let imported = 0;
    for (const record of snapshot.records) {
      this.records.set(record.id, { ...record });
      imported++;
    }
    return imported;
  }
}

/**
 * Conversation memory helper: keeps the rolling window for one chat session.
 * It stores what was said, not what was inferred.
 */
export class ConversationMemory {
  readonly sessionId: string;
  readonly maxTurns: number;
  private turns: { role: "user" | "alpha"; text: string; at: number }[] = [];

  constructor(sessionId: string, maxTurns = 12) {
    this.sessionId = sessionId;
    this.maxTurns = maxTurns;
  }

  append(role: "user" | "alpha", text: string): void {
    this.turns.push({ role, text, at: Date.now() });
    if (this.turns.length > this.maxTurns) this.turns = this.turns.slice(-this.maxTurns);
  }

  transcript(): string {
    return this.turns.map((turn) => `${turn.role}: ${turn.text}`).join("\n");
  }

  clear(): void {
    this.turns = [];
  }

  get length(): number {
    return this.turns.length;
  }
}
