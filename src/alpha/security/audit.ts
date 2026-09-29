/**
 * Alpha Security — audit log.
 *
 * Append-only and hash-chained: each record stores the hash of the previous
 * record plus its own contents, so a deleted or edited entry is detectable by
 * replaying the chain. This is what makes "never allow an agent to execute a
 * powerful action without permission" a verifiable claim rather than a promise.
 */

import { alphaId } from "../core/types";
import { redactSecrets } from "./validation";

export type AuditDecision = "allow" | "deny" | "error" | "info";

export type AuditRecord = {
  id: string;
  at: number;
  actor: string;
  module: string;
  action: string;
  resource: string | null;
  decision: AuditDecision;
  reason: string;
  traceId: string | null;
  data: Record<string, unknown>;
  prevHash: string;
  hash: string;
};

/** FNV-1a over the record payload — dependency-free and stable. */
function hashPayload(payload: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < payload.length; i++) {
    hash ^= payload.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

export class AlphaAuditLog {
  private records: AuditRecord[] = [];
  private readonly maxRecords: number;

  constructor(options: { maxRecords?: number } = {}) {
    this.maxRecords = options.maxRecords ?? 2000;
  }

  append(input: {
    actor: string;
    module: string;
    action: string;
    resource?: string | null;
    decision: AuditDecision;
    reason?: string;
    traceId?: string | null;
    data?: Record<string, unknown>;
  }): AuditRecord {
    const previous = this.records[this.records.length - 1];
    const prevHash = previous ? previous.hash : "genesis";
    const body = {
      id: alphaId("audit"),
      at: Date.now(),
      actor: input.actor,
      module: input.module,
      action: input.action,
      resource: input.resource ?? null,
      decision: input.decision,
      reason: redactSecrets(input.reason ?? ""),
      traceId: input.traceId ?? null,
      data: JSON.parse(redactSecrets(JSON.stringify(input.data ?? {}))) as Record<string, unknown>,
      prevHash,
    };
    const hash = hashPayload(`${prevHash}|${JSON.stringify(body)}`);
    const record: AuditRecord = { ...body, hash };
    this.records.push(record);
    if (this.records.length > this.maxRecords) {
      this.records = this.records.slice(-this.maxRecords);
    }
    return record;
  }

  list(options: { limit?: number; actor?: string; module?: string; decision?: AuditDecision } = {}): AuditRecord[] {
    return this.records
      .filter((record) => (options.actor ? record.actor === options.actor : true))
      .filter((record) => (options.module ? record.module === options.module : true))
      .filter((record) => (options.decision ? record.decision === options.decision : true))
      .slice(-(options.limit ?? 100))
      .reverse();
  }

  /** Verify the chain is intact; returns the first broken record id, if any. */
  verifyChain(): { intact: boolean; checked: number; brokenAt: string | null } {
    let prevHash = "genesis";
    for (const record of this.records) {
      const { hash, ...body } = record;
      const expected = hashPayload(`${prevHash}|${JSON.stringify(body)}`);
      if (record.prevHash !== prevHash || expected !== hash) {
        return { intact: false, checked: this.records.length, brokenAt: record.id };
      }
      prevHash = record.hash;
    }
    return { intact: true, checked: this.records.length, brokenAt: null };
  }

  get size(): number {
    return this.records.length;
  }
}
