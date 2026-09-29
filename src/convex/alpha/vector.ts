import { v } from "convex/values";
import { mutation, query } from "../_generated/server";
import { requireActorId } from "./helpers";

/**
 * Alpha's vectors, stored in Alpha's own database.
 *
 * Embeddings come from Alpha's transformer hidden states (see
 * `src/alpha/embeddings`). No hosted vector database is involved.
 */
export const upsertVectors = mutation({
  args: {
    sessionToken: v.string(),
    collection: v.string(),
    dimension: v.number(),
    records: v.array(
      v.object({
        recordId: v.string(),
        text: v.string(),
        embedding: v.array(v.float64()),
        metadata: v.any(),
        sourceId: v.optional(v.string()),
      }),
    ),
  },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    let written = 0;
    for (const record of args.records) {
      const existing = await ctx.db
        .query("alphaVectors")
        .withIndex("by_record", (q) => q.eq("recordId", record.recordId))
        .first();
      const now = Date.now();
      if (existing && existing.actorId === actorId) {
        await ctx.db.patch(existing._id, {
          text: record.text,
          embedding: record.embedding,
          metadata: record.metadata,
          sourceId: record.sourceId,
          dimension: args.dimension,
          collection: args.collection,
          updatedAt: now,
        });
      } else {
        await ctx.db.insert("alphaVectors", {
          actorId,
          recordId: record.recordId,
          collection: args.collection,
          dimension: args.dimension,
          text: record.text,
          embedding: record.embedding,
          metadata: record.metadata,
          sourceId: record.sourceId,
          createdAt: now,
          updatedAt: now,
        });
      }
      written++;
    }
    return { written };
  },
});

export const listVectors = query({
  args: { sessionToken: v.string(), collection: v.optional(v.string()) },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    if (args.collection) {
      return await ctx.db
        .query("alphaVectors")
        .withIndex("by_actor_collection", (q) => q.eq("actorId", actorId).eq("collection", args.collection!))
        .collect();
    }
    return await ctx.db
      .query("alphaVectors")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
  },
});

export const stats = query({
  args: { sessionToken: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = await ctx.db
      .query("alphaVectors")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
    const collections = new Map<string, { collection: string; records: number; dimension: number }>();
    for (const row of rows) {
      const entry = collections.get(row.collection) ?? {
        collection: row.collection,
        records: 0,
        dimension: row.dimension,
      };
      entry.records++;
      collections.set(row.collection, entry);
    }
    return { total: rows.length, collections: [...collections.values()] };
  },
});

export const clearCollection = mutation({
  args: { sessionToken: v.string(), collection: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = await ctx.db
      .query("alphaVectors")
      .withIndex("by_actor_collection", (q) => q.eq("actorId", actorId).eq("collection", args.collection))
      .collect();
    for (const row of rows) await ctx.db.delete(row._id);
    return { deleted: rows.length };
  },
});

export const deleteBySource = mutation({
  args: { sessionToken: v.string(), collection: v.string(), sourceId: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = await ctx.db
      .query("alphaVectors")
      .withIndex("by_source", (q) => q.eq("actorId", actorId).eq("sourceId", args.sourceId))
      .collect();
    let deleted = 0;
    for (const row of rows) {
      if (row.collection !== args.collection) continue;
      await ctx.db.delete(row._id);
      deleted++;
    }
    return { deleted };
  },
});
