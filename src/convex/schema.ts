import { authTables } from "@convex-dev/auth/server";
import { defineSchema, defineTable } from "convex/server";
import { Infer, v } from "convex/values";

// default user roles. can add / remove based on the project as needed
export const ROLES = {
  ADMIN: "admin",
  USER: "user",
  MEMBER: "member",
} as const;

export const roleValidator = v.union(
  v.literal(ROLES.ADMIN),
  v.literal(ROLES.USER),
  v.literal(ROLES.MEMBER),
);
export type Role = Infer<typeof roleValidator>;

/**
 * Alpha persistence tables.
 *
 * These store Alpha's *own* artifacts: model records, trained vocabularies,
 * checkpoints produced by Alpha's training engine, vectors from Alpha's
 * embeddings, memories, run records, spans, the audit chain and workflow jobs.
 *
 * Convex is the application database, not an AI provider: no table here holds
 * output from an external model, and nothing in the Alpha stack calls out to
 * one. Every row is scoped to the signed-in user.
 */
const schema = defineSchema(
  {
    // default auth tables using convex auth.
    ...authTables, // do not remove or modify

    // the users table is the default users table that is brought in by the authTables
    users: defineTable({
      name: v.optional(v.string()), // name of the user. do not remove
      image: v.optional(v.string()), // image of the user. do not remove
      email: v.optional(v.string()), // email of the user. do not remove
      emailVerificationTime: v.optional(v.number()), // email verification time. do not remove
      isAnonymous: v.optional(v.boolean()), // is the user anonymous. do not remove

      role: v.optional(roleValidator), // role of the user. do not remove
    }).index("email", ["email"]), // index for the email. do not remove or modify

    /**
     * Model records. `stage` is one of architecture | untrained | trained |
     * fine-tuned | production and is derived from artefacts, never asserted.
     */
    alphaModels: defineTable({
      actorId: v.string(),
      name: v.string(),
      version: v.string(),
      stage: v.string(),
      parameterCount: v.number(),
      config: v.any(),
      checkpointId: v.optional(v.string()),
      trainedTokens: v.optional(v.number()),
      validationLoss: v.optional(v.number()),
      notes: v.array(v.string()),
      createdAt: v.number(),
      updatedAt: v.number(),
    })
      .index("by_actor", ["actorId"])
      .index("by_actor_name", ["actorId", "name"]),

    /** Trained tokenizer vocabularies (merge table + stats + provenance). */
    alphaTokenizers: defineTable({
      actorId: v.string(),
      version: v.string(),
      trainedOn: v.string(),
      vocabSize: v.number(),
      mergeSteps: v.number(),
      documents: v.number(),
      characters: v.number(),
      snapshot: v.any(),
      createdAt: v.number(),
    }).index("by_actor", ["actorId"]),

    /** Checkpoints from Alpha's training engine (weights + optimiser state). */
    alphaCheckpoints: defineTable({
      actorId: v.string(),
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
      createdAt: v.number(),
    })
      .index("by_actor", ["actorId"])
      .index("by_checkpoint", ["checkpointId"]),

    /** Datasets available to Alpha (seed corpus plus anything the user adds). */
    alphaDatasets: defineTable({
      actorId: v.string(),
      datasetId: v.string(),
      name: v.string(),
      version: v.string(),
      description: v.string(),
      license: v.string(),
      source: v.string(),
      documents: v.number(),
      characters: v.number(),
      text: v.array(v.string()),
      createdAt: v.number(),
    })
      .index("by_actor", ["actorId"])
      .index("by_dataset", ["datasetId"]),

    /** Vectors produced by Alpha's own embeddings. */
    alphaVectors: defineTable({
      actorId: v.string(),
      recordId: v.string(),
      collection: v.string(),
      dimension: v.number(),
      text: v.string(),
      embedding: v.array(v.float64()),
      metadata: v.any(),
      sourceId: v.optional(v.string()),
      createdAt: v.number(),
      updatedAt: v.number(),
    })
      .index("by_actor", ["actorId"])
      .index("by_actor_collection", ["actorId", "collection"])
      .index("by_record", ["recordId"])
      .index("by_source", ["actorId", "sourceId"]),

    /** Alpha's memory records across the three scopes. */
    alphaMemories: defineTable({
      actorId: v.string(),
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
    })
      .index("by_actor", ["actorId"])
      .index("by_scope", ["actorId", "scope"])
      .index("by_memory", ["memoryId"]),

    /** Run records: inference, training, rag, agent, workflow. */
    alphaRuns: defineTable({
      actorId: v.string(),
      kind: v.string(),
      status: v.string(),
      traceId: v.string(),
      modelStage: v.string(),
      input: v.string(),
      output: v.string(),
      metrics: v.any(),
      error: v.optional(v.string()),
      createdAt: v.number(),
    })
      .index("by_actor", ["actorId"])
      .index("by_kind", ["actorId", "kind"])
      .index("by_trace", ["traceId"]),

    /** Spans for Alpha's own tracing. */
    alphaSpans: defineTable({
      actorId: v.string(),
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
      createdAt: v.number(),
    })
      .index("by_actor", ["actorId"])
      .index("by_trace", ["traceId"]),

    /** Append-only, hash-chained audit records. */
    alphaAuditLogs: defineTable({
      actorId: v.string(),
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
    })
      .index("by_actor", ["actorId"])
      .index("by_record", ["recordId"]),

    /** Tool descriptors registered by the workspace, for discovery and review. */
    alphaTools: defineTable({
      actorId: v.string(),
      name: v.string(),
      description: v.string(),
      module: v.string(),
      permission: v.string(),
      requiresApproval: v.boolean(),
      dangerous: v.boolean(),
      source: v.string(),
      inputSchema: v.any(),
      stats: v.any(),
      updatedAt: v.number(),
    })
      .index("by_actor", ["actorId"])
      .index("by_actor_name", ["actorId", "name"]),

    /** Automation workflows. */
    alphaWorkflows: defineTable({
      actorId: v.string(),
      workflowId: v.string(),
      name: v.string(),
      description: v.string(),
      trigger: v.any(),
      conditions: v.any(),
      actions: v.any(),
      enabled: v.boolean(),
      actor: v.string(),
      createdAt: v.number(),
      updatedAt: v.number(),
    })
      .index("by_actor", ["actorId"])
      .index("by_workflow", ["workflowId"]),

    /** Workflow executions, including retries and outcomes. */
    alphaJobs: defineTable({
      actorId: v.string(),
      jobId: v.string(),
      workflowId: v.string(),
      workflowName: v.string(),
      status: v.string(),
      attempts: v.number(),
      outcomes: v.any(),
      errors: v.array(v.string()),
      durationMs: v.number(),
      traceId: v.string(),
      createdAt: v.number(),
    })
      .index("by_actor", ["actorId"])
      .index("by_workflow", ["workflowId"]),
  },
  {
    schemaValidation: false,
  },
);

export default schema;
