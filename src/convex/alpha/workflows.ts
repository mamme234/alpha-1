import { v } from "convex/values";
import { mutation, query } from "../_generated/server";
import { requireActorId } from "./helpers";

/** Automation workflows (trigger + conditions + actions, stored as data). */
export const saveWorkflow = mutation({
  args: {
    workflowId: v.string(),
    name: v.string(),
    description: v.string(),
    trigger: v.any(),
    conditions: v.any(),
    actions: v.any(),
    enabled: v.boolean(),
    actor: v.string(),
  },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx);
    const existing = await ctx.db
      .query("alphaWorkflows")
      .withIndex("by_workflow", (q) => q.eq("workflowId", args.workflowId))
      .first();
    const now = Date.now();
    if (existing && existing.actorId === actorId) {
      await ctx.db.patch(existing._id, { ...args, updatedAt: now });
      return existing._id;
    }
    return await ctx.db.insert("alphaWorkflows", { ...args, actorId, createdAt: now, updatedAt: now });
  },
});

export const listWorkflows = query({
  args: {},
  handler: async (ctx) => {
    const actorId = await requireActorId(ctx);
    const rows = await ctx.db
      .query("alphaWorkflows")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
    return rows.sort((a, b) => b.updatedAt - a.updatedAt);
  },
});

/** Workflow executions, including retries and per-action outcomes. */
export const saveJob = mutation({
  args: {
    jobId: v.string(),
    workflowId: v.string(),
    workflowName: v.string(),
    status: v.string(),
    attempts: v.number(),
    outcomes: v.any(),
    errors: v.array(v.string()),
    durationMs: v.number(),
    traceId: v.string(),
  },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx);
    const rows = await ctx.db
      .query("alphaJobs")
      .withIndex("by_workflow", (q) => q.eq("workflowId", args.workflowId))
      .collect();
    const existing = rows.find((row) => row.jobId === args.jobId && row.actorId === actorId);
    if (existing) {
      await ctx.db.patch(existing._id, { ...args });
      return existing._id;
    }
    return await ctx.db.insert("alphaJobs", { ...args, actorId, createdAt: Date.now() });
  },
});

export const listJobs = query({
  args: { limit: v.optional(v.number()) },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx);
    const rows = await ctx.db
      .query("alphaJobs")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
    return rows.sort((a, b) => b.createdAt - a.createdAt).slice(0, args.limit ?? 25);
  },
});
