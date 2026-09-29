import { v } from "convex/values";
import { mutation, query } from "../_generated/server";
import { requireActorId } from "./helpers";

/** Alpha's memory records, scoped to the signed-in user. */
export const upsertMemories = mutation({
  args: {
    sessionToken: v.string(),
    records: v.array(
      v.object({
        memoryId: v.string(),
        scope: v.string(),
        sessionId: v.optional(v.string()),
        key: v.string(),
        content: v.string(),
        embedding: v.array(v.float64()),
        tags: v.array(v.string()),
        importance: v.number(),
        approved: v.boolean(),
        source: v.string(),
        accessCount: v.number(),
        createdAt: v.number(),
        updatedAt: v.number(),
      }),
    ),
  },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    let written = 0;
    for (const record of args.records) {
      const existing = await ctx.db
        .query("alphaMemories")
        .withIndex("by_memory", (q) => q.eq("memoryId", record.memoryId))
        .first();
      if (existing && existing.actorId === actorId) {
        await ctx.db.patch(existing._id, record);
      } else {
        await ctx.db.insert("alphaMemories", { ...record, actorId });
      }
      written++;
    }
    return { written };
  },
});

export const list = query({
  args: { sessionToken: v.string(), scope: v.optional(v.string()) },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = args.scope
      ? await ctx.db
          .query("alphaMemories")
          .withIndex("by_scope", (q) => q.eq("actorId", actorId).eq("scope", args.scope!))
          .collect()
      : await ctx.db
          .query("alphaMemories")
          .withIndex("by_actor", (q) => q.eq("actorId", actorId))
          .collect();
    return rows.sort((a, b) => b.updatedAt - a.updatedAt);
  },
});

export const remove = mutation({
  args: { sessionToken: v.string(), memoryId: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const existing = await ctx.db
      .query("alphaMemories")
      .withIndex("by_memory", (q) => q.eq("memoryId", args.memoryId))
      .first();
    if (!existing || existing.actorId !== actorId) return { deleted: 0 };
    await ctx.db.delete(existing._id);
    return { deleted: 1 };
  },
});

export const clearScope = mutation({
  args: { sessionToken: v.string(), scope: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = await ctx.db
      .query("alphaMemories")
      .withIndex("by_scope", (q) => q.eq("actorId", actorId).eq("scope", args.scope))
      .collect();
    for (const row of rows) await ctx.db.delete(row._id);
    return { deleted: rows.length };
  },
});
