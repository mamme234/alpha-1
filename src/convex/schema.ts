import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

/**
 * Alpha's database schema.
 *
 * Every table here belongs to Alpha. There is no provider-owned table, no
 * federated identity table and nothing that stores output from an external
 * model. Convex is the application database — Alpha's own backend runtime —
 * not an AI provider.
 *
 * Responsibilities are kept apart:
 *   alphaUsers, alphaSessions   → identity and sessions (Alpha Authentication)
 *   alphaConversations, alphaMessages → conversations
 *   alphaModels, alphaTokenizers, alphaDatasets, alphaCheckpoints → model management
 *   alphaVectors, alphaMemories → retrieval and memory
 *   alphaRuns, alphaSpans, alphaAuditLogs → observability and audit
 *   alphaTools, alphaWorkflows, alphaJobs → tools and automation
 *
 * `actorId` is the id of the `alphaUsers` row that owns the record. It is the
 * only value the API layer trusts for ownership, and it is resolved from a
 * session on every call — never taken from the client.
 */

export const ALPHA_ROLES = ["user", "admin"] as const;
export type AlphaRole = (typeof ALPHA_ROLES)[number];

export const ALPHA_ACCOUNT_STATUSES = ["active", "suspended"] as const;
export type AlphaAccountStatus = (typeof ALPHA_ACCOUNT_STATUSES)[number];

export const alphaRoleValidator = v.union(v.literal("user"), v.literal("admin"));
export const alphaAccountStatusValidator = v.union(v.literal("active"), v.literal("suspended"));

const schema = defineSchema(
  {
    /**
     * Alpha accounts.
     *
     * Passwords are never stored: only a PBKDF2-HMAC-SHA256 derivation with a
     * per-account salt. `passwordIterations` is stored alongside so the work
     * factor can be raised later without invalidating existing accounts.
     */
    alphaUsers: defineTable({
      /** The address as the account holder typed it, for display. */
      email: v.string(),
      /** Lowercased, trimmed address. The unique key lookups use. */
      emailKey: v.string(),
      displayName: v.string(),
      passwordHash: v.string(),
      passwordSalt: v.string(),
      passwordIterations: v.number(),
      passwordAlgorithm: v.string(),
      role: alphaRoleValidator,
      status: alphaAccountStatusValidator,
      createdAt: v.number(),
      updatedAt: v.number(),
      lastSignInAt: v.optional(v.number()),
      failedSignIns: v.number(),
      /** Set while a burst of failed sign-ins is cooling down. */
      lockedUntil: v.optional(v.number()),
      /**
       * Bumped whenever every existing session must stop working (a password
       * change, or "sign out everywhere"). Sessions carry the epoch they were
       * issued under; a mismatch means the session is dead.
       */
      sessionEpoch: v.number(),
    }).index("by_email", ["emailKey"]),

    /**
     * Alpha sessions.
     *
     * The row stores `sha256(token)` — never the token itself — so a database
     * dump does not hand over working sessions. Sessions expire absolutely
     * (`expiresAt`), go idle (`lastSeenAt` + the API's idle window), and can be
     * revoked individually or all at once.
     */
    alphaSessions: defineTable({
      userId: v.id("alphaUsers"),
      tokenHash: v.string(),
      epoch: v.number(),
      createdAt: v.number(),
      expiresAt: v.number(),
      lastSeenAt: v.number(),
      revokedAt: v.optional(v.number()),
      revokedReason: v.optional(v.string()),
      userAgent: v.optional(v.string()),
    })
      .index("by_token", ["tokenHash"])
      .index("by_user", ["userId"]),

    /** A conversation between a user and Alpha. */
    alphaConversations: defineTable({
      actorId: v.string(),
      conversationId: v.string(),
      title: v.string(),
      /** generate | rag | agent */
      kind: v.string(),
      messageCount: v.number(),
      lastMessageAt: v.number(),
      createdAt: v.number(),
      updatedAt: v.number(),
    })
      .index("by_actor", ["actorId"])
      .index("by_conversation", ["conversationId"])
      .index("by_actor_updated", ["actorId", "updatedAt"]),

    /**
     * One turn in a conversation. `content` is Alpha's own output (or the user's
     * message) with `modelStage` recorded next to it, so a transcript can never
     * be mistaken for a finished model's answer.
     */
    alphaMessages: defineTable({
      actorId: v.string(),
      conversationId: v.string(),
      messageId: v.string(),
      /** user | assistant | system */
      role: v.string(),
      content: v.string(),
      /** Retrieved sources cited by an assistant message, if any. */
      sources: v.array(v.object({ title: v.string(), score: v.float64(), chunkId: v.string() })),
      modelStage: v.optional(v.string()),
      tokens: v.optional(v.number()),
      traceId: v.optional(v.string()),
      createdAt: v.number(),
    })
      .index("by_conversation", ["actorId", "conversationId"])
      .index("by_message", ["messageId"]),

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

    /**
     * Security events raised by the API layer: sign-in outcomes, lockouts,
     * revoked sessions and rejected authorisation checks.
     */
    alphaSecurityEvents: defineTable({
      actorId: v.optional(v.string()),
      emailKey: v.optional(v.string()),
      kind: v.string(),
      outcome: v.string(),
      detail: v.string(),
      at: v.number(),
    })
      .index("by_actor", ["actorId"])
      .index("by_kind", ["kind"])
      .index("by_email", ["emailKey"]),

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
