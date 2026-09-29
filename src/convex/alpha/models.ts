import { v } from "convex/values";
import { mutation, query } from "../_generated/server";
import { requireActorId } from "./helpers";

/**
 * Model records.
 *
 * `stage` is written by the workspace from what actually exists (a checkpoint
 * with step > 0), so a record can never claim to be trained without one.
 */
export const record = mutation({
  args: {
    name: v.string(),
    version: v.string(),
    stage: v.string(),
    parameterCount: v.number(),
    config: v.any(),
    checkpointId: v.optional(v.string()),
    trainedTokens: v.optional(v.number()),
    validationLoss: v.optional(v.number()),
    notes: v.array(v.string()),
  },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx);
    const existing = await ctx.db
      .query("alphaModels")
      .withIndex("by_actor_name", (q) => q.eq("actorId", actorId).eq("name", args.name))
      .first();
    const now = Date.now();
    if (existing) {
      await ctx.db.patch(existing._id, { ...args, updatedAt: now });
      return existing._id;
    }
    return await ctx.db.insert("alphaModels", { ...args, actorId, createdAt: now, updatedAt: now });
  },
});

export const current = query({
  args: {},
  handler: async (ctx) => {
    const actorId = await requireActorId(ctx);
    const models = await ctx.db
      .query("alphaModels")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
    if (models.length === 0) return null;
    return models.sort((a, b) => b.updatedAt - a.updatedAt)[0];
  },
});

export const list = query({
  args: {},
  handler: async (ctx) => {
    const actorId = await requireActorId(ctx);
    const models = await ctx.db
      .query("alphaModels")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
    return models.sort((a, b) => b.updatedAt - a.updatedAt);
  },
});
