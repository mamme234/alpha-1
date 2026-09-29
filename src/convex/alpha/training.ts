import { v } from "convex/values";
import { mutation, query } from "../_generated/server";
import { requireActorId } from "./helpers";

/** Trained tokenizer vocabularies. */
export const saveTokenizer = mutation({
  args: {
    sessionToken: v.string(),
    version: v.string(),
    trainedOn: v.string(),
    vocabSize: v.number(),
    mergeSteps: v.number(),
    documents: v.number(),
    characters: v.number(),
    snapshot: v.any(),
  },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const existing = await ctx.db
      .query("alphaTokenizers")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
    const match = existing.find((row) => row.version === args.version && row.trainedOn === args.trainedOn);
    if (match) {
      await ctx.db.patch(match._id, { ...args, createdAt: Date.now() });
      return match._id;
    }
    return await ctx.db.insert("alphaTokenizers", { ...args, actorId, createdAt: Date.now() });
  },
});

export const currentTokenizer = query({
  args: { sessionToken: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = await ctx.db
      .query("alphaTokenizers")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
    if (rows.length === 0) return null;
    return rows.sort((a, b) => b.createdAt - a.createdAt)[0];
  },
});

/** Datasets available to Alpha. */
export const saveDataset = mutation({
  args: {
    sessionToken: v.string(),
    datasetId: v.string(),
    name: v.string(),
    version: v.string(),
    description: v.string(),
    license: v.string(),
    source: v.string(),
    documents: v.number(),
    characters: v.number(),
    text: v.array(v.string()),
  },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const existing = await ctx.db
      .query("alphaDatasets")
      .withIndex("by_dataset", (q) => q.eq("datasetId", args.datasetId))
      .filter((q) => q.eq(q.field("actorId"), actorId))
      .first();
    if (existing) {
      await ctx.db.patch(existing._id, { ...args });
      return existing._id;
    }
    return await ctx.db.insert("alphaDatasets", { ...args, actorId, createdAt: Date.now() });
  },
});

export const listDatasets = query({
  args: { sessionToken: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    return await ctx.db
      .query("alphaDatasets")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
  },
});

/** Checkpoints carry the weights Alpha actually trained. */
export const saveCheckpoint = mutation({
  args: {
    sessionToken: v.string(),
    checkpointId: v.string(),
    label: v.string(),
    modelName: v.string(),
    modelVersion: v.string(),
    datasetName: v.string(),
    datasetLicense: v.string(),
    stage: v.string(),
    step: v.number(),
    tokensSeen: v.number(),
    trainLoss: v.number(),
    validationLoss: v.optional(v.number()),
    learningRate: v.number(),
    sizeBytes: v.number(),
    parameterCount: v.number(),
    weights: v.string(),
    optimizer: v.string(),
    rng: v.any(),
    config: v.any(),
  },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const existing = await ctx.db
      .query("alphaCheckpoints")
      .withIndex("by_checkpoint", (q) => q.eq("checkpointId", args.checkpointId))
      .first();
    if (existing && existing.actorId === actorId) {
      await ctx.db.patch(existing._id, { ...args });
      return existing._id;
    }
    return await ctx.db.insert("alphaCheckpoints", { ...args, actorId, createdAt: Date.now() });
  },
});

/** Checkpoint summaries for the UI — the weight payload stays on the server. */
export const listCheckpoints = query({
  args: { sessionToken: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = await ctx.db
      .query("alphaCheckpoints")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
    return rows
      .sort((a, b) => b.createdAt - a.createdAt)
      .map((row) => ({
        _id: row._id,
        checkpointId: row.checkpointId,
        label: row.label,
        modelName: row.modelName,
        stage: row.stage,
        step: row.step,
        tokensSeen: row.tokensSeen,
        trainLoss: row.trainLoss,
        validationLoss: row.validationLoss,
        learningRate: row.learningRate,
        sizeBytes: row.sizeBytes,
        parameterCount: row.parameterCount,
        datasetName: row.datasetName,
        createdAt: row.createdAt,
      }));
  },
});

/** Full checkpoint including weights — used to resume a run. */
export const latestCheckpoint = query({
  args: { sessionToken: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = await ctx.db
      .query("alphaCheckpoints")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
    if (rows.length === 0) return null;
    return rows.sort((a, b) => b.createdAt - a.createdAt)[0];
  },
});
