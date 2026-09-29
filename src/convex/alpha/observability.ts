import { v } from "convex/values";
import { mutation, query } from "../_generated/server";
import { requireActorId } from "./helpers";

/** Run records: inference, training, rag, agent, workflow. */
export const recordRun = mutation({
  args: {
    sessionToken: v.string(),
    kind: v.string(),
    status: v.string(),
    traceId: v.string(),
    modelStage: v.string(),
    input: v.string(),
    output: v.string(),
    metrics: v.any(),
    error: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    return await ctx.db.insert("alphaRuns", { ...args, actorId, createdAt: Date.now() });
  },
});

export const listRuns = query({
  args: { sessionToken: v.string(), kind: v.optional(v.string()), limit: v.optional(v.number()) },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = args.kind
      ? await ctx.db
          .query("alphaRuns")
          .withIndex("by_kind", (q) => q.eq("actorId", actorId).eq("kind", args.kind!))
          .collect()
      : await ctx.db
          .query("alphaRuns")
          .withIndex("by_actor", (q) => q.eq("actorId", actorId))
          .collect();
    return rows.sort((a, b) => b.createdAt - a.createdAt).slice(0, args.limit ?? 50);
  },
});

export const runStats = query({
  args: { sessionToken: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = await ctx.db
      .query("alphaRuns")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
    const byKind = new Map<string, number>();
    let latencySum = 0;
    let latencyCount = 0;
    let generatedTokens = 0;
    for (const row of rows) {
      byKind.set(row.kind, (byKind.get(row.kind) ?? 0) + 1);
      const metrics = (row.metrics ?? {}) as { latencyMs?: number; generatedTokens?: number };
      if (typeof metrics.latencyMs === "number") {
        latencySum += metrics.latencyMs;
        latencyCount++;
      }
      if (typeof metrics.generatedTokens === "number") generatedTokens += metrics.generatedTokens;
    }
    return {
      total: rows.length,
      byKind: [...byKind.entries()].map(([kind, count]) => ({ kind, count })),
      averageLatencyMs: latencyCount > 0 ? latencySum / latencyCount : 0,
      generatedTokens,
    };
  },
});

/** Spans written by Alpha's tracer. */
export const recordSpans = mutation({
  args: {
    sessionToken: v.string(),
    spans: v.array(
      v.object({
        traceId: v.string(),
        spanId: v.string(),
        parentId: v.optional(v.string()),
        name: v.string(),
        kind: v.string(),
        module: v.string(),
        status: v.string(),
        startMs: v.number(),
        durationMs: v.optional(v.number()),
        attributes: v.any(),
      }),
    ),
  },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    for (const span of args.spans) {
      await ctx.db.insert("alphaSpans", { ...span, actorId, createdAt: Date.now() });
    }
    return { written: args.spans.length };
  },
});

export const listSpans = query({
  args: { sessionToken: v.string(), limit: v.optional(v.number()) },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = await ctx.db
      .query("alphaSpans")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
    return rows.sort((a, b) => b.startMs - a.startMs).slice(0, args.limit ?? 40);
  },
});

/** Hash-chained audit records. */
export const appendAudit = mutation({
  args: {
    sessionToken: v.string(),
    records: v.array(
      v.object({
        recordId: v.string(),
        actor: v.string(),
        module: v.string(),
        action: v.string(),
        resource: v.optional(v.string()),
        decision: v.string(),
        reason: v.string(),
        traceId: v.optional(v.string()),
        data: v.any(),
        hash: v.string(),
        prevHash: v.string(),
        at: v.number(),
      }),
    ),
  },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    let written = 0;
    for (const record of args.records) {
      const existing = await ctx.db
        .query("alphaAuditLogs")
        .withIndex("by_record", (q) => q.eq("recordId", record.recordId))
        .first();
      if (existing) continue;
      await ctx.db.insert("alphaAuditLogs", { ...record, actorId });
      written++;
    }
    return { written };
  },
});

export const listAudit = query({
  args: { sessionToken: v.string(), limit: v.optional(v.number()) },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = await ctx.db
      .query("alphaAuditLogs")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
    return rows.sort((a, b) => b.at - a.at).slice(0, args.limit ?? 50);
  },
});
