/**
 * Alpha's chat API — the server side of the product.
 *
 * This module is where the verified Step 7 model becomes a real chat service.
 * A Convex action cannot push to a browser, so the shape is:
 *
 *   client action call ──► load the serving runtime (per account)
 *                          hydrate that account's memories and vectors
 *                          write a row into `alphaStreams`
 *                          run Alpha's own transformer, token by token
 *                          append each token to the stream row
 *        browser ◄──────── subscribe to `activeStream` (a query)
 *                          persist the assistant message
 *                          finalize the stream row last
 *
 * Every step is Alpha's own code: the weights come from the checked-in Step 7
 * serving artefact, embeddings come from Alpha's transformer, retrieval comes
 * from Alpha's vector store. There is no external model and no provider key
 * anywhere in this file.
 *
 * Honesty rules the implementation keeps:
 *   - A bad request is refused with the measured numbers (token counts), not
 *     silently truncated.
 *   - A load failure is raised as `model-unavailable`; nothing is generated.
 *   - A generation failure is stored on the assistant message and the stream
 *     row (`status: "error"`) and returned — never replaced with fallback text.
 *   - The final stream row is written *after* the assistant message, so a
 *     client that sees a finished stream finds the answer in the transcript.
 */

import { ConvexError, v } from "convex/values";
import {
  action,
  internalMutation,
  internalQuery,
  mutation,
  query,
  type ActionCtx,
} from "../_generated/server";
import { api, internal } from "../_generated/api";
import { alphaId } from "../../alpha/core/types";
import {
  createServingRuntime,
  type AlphaServingRuntime,
} from "../../alpha/serving/runtime";
import { buildChatRequest, type ChatHistoryTurn, type ChatTurnSettings } from "../../alpha/serving/chat";
import type { RespondRequest, RespondResult } from "../../alpha/runtime/orchestrator";
import { authReject } from "../alphaAuth/validation";
import { requireActorId } from "./helpers";
import STEP7_ARTIFACT from "../../alpha/serving/step7-artifact.json";

/* -------------------------------------------------------------------------- */
/* Runtime cache                                                               */
/* -------------------------------------------------------------------------- */

/**
 * Runtimes are cached per account, not globally.
 *
 * A serving runtime keeps the account's hydrated memories and vectors in
 * memory. One shared instance would let two concurrent turns from different
 * accounts overwrite each other's hydration, so each account gets its own.
 * The cache is bounded; the least recently used runtime is dropped when a new
 * account arrives. The status runtime is separate because it must never hold
 * any account's data.
 */
const MAX_ACTOR_RUNTIMES = 3;
const actorRuntimes = new Map<string, AlphaServingRuntime>();
let statusRuntime: AlphaServingRuntime | null = null;

export const ALPHA_ARTIFACT_IMPORT_PROVENANCE =
  "src/alpha/serving/step7-artifact.json — built by scripts/alpha-build-step7-artifact.ts from the verified step-7 checkpoint ckpt_mutfvpv45sqe7 (frozen suite evl_b58eacf7; gate gate_9ab33311 not met, recorded as measured)";

function buildRuntime(): AlphaServingRuntime {
  return createServingRuntime({ artifact: STEP7_ARTIFACT, logLevel: "warn" });
}

/** Load (or reuse) the serving runtime for an account. Throws on a bad artefact. */
function runtimeFor(actorId: string): AlphaServingRuntime {
  const cached = actorRuntimes.get(actorId);
  if (cached) {
    // Refresh LRU position.
    actorRuntimes.delete(actorId);
    actorRuntimes.set(actorId, cached);
    return cached;
  }
  const runtime = buildRuntime();
  actorRuntimes.set(actorId, runtime);
  while (actorRuntimes.size > MAX_ACTOR_RUNTIMES) {
    const oldest = actorRuntimes.keys().next().value as string | undefined;
    if (oldest === undefined) break;
    actorRuntimes.delete(oldest);
  }
  return runtime;
}

/* -------------------------------------------------------------------------- */
/* Errors                                                                      */
/* -------------------------------------------------------------------------- */

/** Data carried by every serving failure, so the client can act on the code. */
export type AlphaServingErrorData = {
  alphaServing: true;
  code: string;
  message: string;
};

function servingReject(code: string, message: string): never {
  throw new ConvexError<AlphaServingErrorData>({ alphaServing: true, code, message });
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "string" && error) return error;
  return "Alpha could not complete that request.";
}

/** Resolve a session token to an actor id *inside an action*. */
async function actorFromToken(ctx: ActionCtx, sessionToken: string): Promise<string> {
  if (typeof sessionToken !== "string" || sessionToken.length === 0) {
    authReject("not-signed-in", "Alpha's chat API requires a signed-in account. Sign in and try again.");
  }
  const resolved = await ctx.runQuery(internal.alphaAuth.sessions.readAccountForToken, {
    token: sessionToken,
  });
  if (!resolved) {
    authReject("not-signed-in", "Alpha's chat API requires a signed-in account. Sign in and try again.");
  }
  return resolved.user._id as unknown as string;
}

/* -------------------------------------------------------------------------- */
/* Validators and shared shapes                                                */
/* -------------------------------------------------------------------------- */

const settingsValidator = v.object({
  temperature: v.optional(v.number()),
  topK: v.optional(v.number()),
  topP: v.optional(v.number()),
  maxNewTokens: v.optional(v.number()),
  repetitionPenalty: v.optional(v.number()),
  deterministic: v.optional(v.boolean()),
  seed: v.optional(v.number()),
  useMemory: v.optional(v.boolean()),
  useRetrieval: v.optional(v.boolean()),
  allowedTools: v.optional(v.array(v.string())),
});

type SettingsInput = {
  temperature?: number;
  topK?: number;
  topP?: number;
  maxNewTokens?: number;
  repetitionPenalty?: number;
  deterministic?: boolean;
  seed?: number;
  useMemory?: boolean;
  useRetrieval?: boolean;
  allowedTools?: string[];
};

/** The public shape of a stream row. `chunks` is what a streaming UI renders. */
function streamView(row: {
  streamId: string;
  status: string;
  stopRequested: boolean;
  chunks: string[];
  text: string;
  modelStage: string;
  outputTokens?: number;
  promptTokens?: number;
  error?: string;
  stopReason?: string;
  messageId?: string;
  route?: string;
  requestId: string;
  latencyMs?: number;
  createdAt: number;
  updatedAt: number;
}) {
  return {
    streamId: row.streamId,
    status: row.status,
    stopRequested: row.stopRequested,
    text: row.text,
    chunks: row.chunks,
    outputTokens: row.outputTokens ?? 0,
    promptTokens: row.promptTokens ?? null,
    error: row.error ?? null,
    stopReason: row.stopReason ?? null,
    messageId: row.messageId ?? null,
    route: row.route ?? null,
    requestId: row.requestId,
    modelStage: row.modelStage,
    latencyMs: row.latencyMs ?? null,
    createdAt: row.createdAt,
    updatedAt: row.updatedAt,
  };
}

/* -------------------------------------------------------------------------- */
/* Public queries                                                              */
/* -------------------------------------------------------------------------- */

/**
 * The live stream for a conversation, or null when there is none.
 *
 * This is the subscription that makes streaming work: while the action runs,
 * `chunks` grows; when the action finishes, `status` becomes `done` | `stopped`
 * | `error` and `messageId` points at the persisted assistant message.
 */
export const activeStream = query({
  args: { sessionToken: v.string(), conversationId: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = await ctx.db
      .query("alphaStreams")
      .withIndex("by_actor_conversation", (q) =>
        q.eq("actorId", actorId).eq("conversationId", args.conversationId),
      )
      .collect();
    const latest = rows.sort((a, b) => b.createdAt - a.createdAt)[0];
    return latest ? streamView(latest) : null;
  },
});

/** Everything a status panel needs about the served model, straight from the artefact. */
export const modelStatus = action({
  args: {},
  handler: async () => {
    try {
      if (!statusRuntime) statusRuntime = buildRuntime();
      return {
        available: true as const,
        info: statusRuntime.info,
        limits: statusRuntime.limits,
        tools: statusRuntime.tools.describe().map((tool) => ({
          name: tool.name,
          description: tool.description,
        })),
      };
    } catch (error) {
      return { available: false as const, error: errorMessage(error) };
    }
  },
});

/* -------------------------------------------------------------------------- */
/* Public mutations                                                            */
/* -------------------------------------------------------------------------- */

/**
 * Set the stop flag for a live generation. The action checks it between
 * tokens; the partial answer is kept and marked `cancelled`, not discarded.
 */
export const stop = mutation({
  args: { sessionToken: v.string(), streamId: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const row = await ctx.db
      .query("alphaStreams")
      .withIndex("by_stream", (q) => q.eq("streamId", args.streamId))
      .first();
    if (!row || row.actorId !== actorId) {
      return { stopped: false, reason: "that stream does not belong to this account" };
    }
    if (row.status !== "streaming") {
      return { stopped: false, reason: `the stream already finished with status "${row.status}"` };
    }
    await ctx.db.patch(row._id, { stopRequested: true, updatedAt: Date.now() });
    return { stopped: true, reason: "stop requested; generation will stop at the next token boundary" };
  },
});

/**
 * Stop whatever is generating in a conversation right now.
 *
 * The stream-id variant above is what a subscribed client uses; this variant
 * is for callers that know the conversation but not the stream id (and for the
 * production verification, where the stream id is discovered server-side in
 * one round trip).
 */
export const stopLatest = mutation({
  args: { sessionToken: v.string(), conversationId: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const streams = await ctx.db
      .query("alphaStreams")
      .withIndex("by_actor_conversation", (q) =>
        q.eq("actorId", actorId).eq("conversationId", args.conversationId),
      )
      .collect();
    const latest = streams.sort((a, b) => b.createdAt - a.createdAt)[0];
    if (!latest) {
      return { stopped: false, streamId: null, reason: "no generation has run in this conversation" };
    }
    if (latest.status !== "streaming") {
      return {
        stopped: false,
        streamId: latest.streamId,
        reason: `the stream already finished with status "${latest.status}"`,
      };
    }
    await ctx.db.patch(latest._id, { stopRequested: true, updatedAt: Date.now() });
    return {
      stopped: true,
      streamId: latest.streamId,
      reason: "stop requested; generation will stop at the next token boundary",
    };
  },
});

/**
 * Clear a conversation's transcript. Refused while a reply is being generated,
 * because deleting messages mid-generation would leave the action writing into
 * a transcript that no longer exists.
 */
export const clearMessages = mutation({
  args: { sessionToken: v.string(), conversationId: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const conversation = await ctx.db
      .query("alphaConversations")
      .withIndex("by_conversation", (q) => q.eq("conversationId", args.conversationId))
      .first();
    if (!conversation || conversation.actorId !== actorId) {
      return { cleared: 0, conversationMissing: true };
    }
    const streams = await ctx.db
      .query("alphaStreams")
      .withIndex("by_actor_conversation", (q) =>
        q.eq("actorId", actorId).eq("conversationId", args.conversationId),
      )
      .collect();
    if (streams.some((row) => row.status === "streaming")) {
      servingReject(
        "stream-in-progress",
        "A reply is still being generated. Stop it before clearing this conversation.",
      );
    }
    const messages = await ctx.db
      .query("alphaMessages")
      .withIndex("by_conversation", (q) =>
        q.eq("actorId", actorId).eq("conversationId", args.conversationId),
      )
      .collect();
    for (const message of messages) await ctx.db.delete(message._id);
    for (const stream of streams) await ctx.db.delete(stream._id);
    await ctx.db.patch(conversation._id, {
      messageCount: 0,
      updatedAt: Date.now(),
    });
    return { cleared: messages.length, conversationMissing: false };
  },
});

/* -------------------------------------------------------------------------- */
/* Internal helpers used by the actions                                        */
/* -------------------------------------------------------------------------- */

/** What the generation loop polls to learn whether the user pressed stop. */
export const readStreamControl = internalQuery({
  args: { streamId: v.string() },
  handler: async (ctx, args) => {
    const row = await ctx.db
      .query("alphaStreams")
      .withIndex("by_stream", (q) => q.eq("streamId", args.streamId))
      .first();
    if (!row) return null;
    return { stopRequested: row.stopRequested, status: row.status };
  },
});

/**
 * One account's memory and vector rows, shaped for the serving runtime.
 * Bounded: a serving turn is not a full-database export, and the truncation is
 * reported rather than hidden.
 */
export const readActorStores = internalQuery({
  args: { actorId: v.string() },
  handler: async (ctx, args) => {
    const memoryLimit = 500;
    const vectorLimit = 1000;
    const memoryRows = await ctx.db
      .query("alphaMemories")
      .withIndex("by_actor", (q) => q.eq("actorId", args.actorId))
      .take(memoryLimit + 1);
    const vectorRows = await ctx.db
      .query("alphaVectors")
      .withIndex("by_actor", (q) => q.eq("actorId", args.actorId))
      .take(vectorLimit + 1);
    return {
      memories: memoryRows.slice(0, memoryLimit).map((row) => ({
        memoryId: row.memoryId,
        scope: row.scope,
        sessionId: row.sessionId ?? null,
        key: row.key,
        content: row.content,
        embedding: row.embedding,
        tags: row.tags,
        importance: row.importance,
        approved: row.approved,
        source: row.source,
        accessCount: row.accessCount,
        createdAt: row.createdAt,
        updatedAt: row.updatedAt,
      })),
      vectors: vectorRows.slice(0, vectorLimit).map((row) => ({
        recordId: row.recordId,
        collection: row.collection,
        dimension: row.dimension,
        text: row.text,
        embedding: row.embedding,
        metadata: (row.metadata ?? {}) as Record<string, unknown>,
        sourceId: row.sourceId ?? null,
        createdAt: row.createdAt,
        updatedAt: row.updatedAt,
      })),
      truncated: {
        memories: memoryRows.length > memoryLimit,
        vectors: vectorRows.length > vectorLimit,
      },
    };
  },
});

export const createStream = internalMutation({
  args: {
    actorId: v.string(),
    conversationId: v.string(),
    streamId: v.string(),
    requestId: v.string(),
    modelId: v.string(),
    modelStage: v.string(),
    generationConfig: v.any(),
  },
  handler: async (ctx, args) => {
    const now = Date.now();
    await ctx.db.insert("alphaStreams", {
      actorId: args.actorId,
      conversationId: args.conversationId,
      streamId: args.streamId,
      requestId: args.requestId,
      status: "streaming",
      stopRequested: false,
      chunks: [],
      text: "",
      modelId: args.modelId,
      modelStage: args.modelStage,
      generationConfig: args.generationConfig,
      outputTokens: 0,
      createdAt: now,
      updatedAt: now,
    });
    return { streamId: args.streamId };
  },
});

export const appendStreamChunk = internalMutation({
  args: { streamId: v.string(), delta: v.string() },
  handler: async (ctx, args) => {
    const row = await ctx.db
      .query("alphaStreams")
      .withIndex("by_stream", (q) => q.eq("streamId", args.streamId))
      .first();
    if (!row) return { updated: false };
    await ctx.db.patch(row._id, {
      chunks: [...row.chunks, args.delta],
      text: row.text + args.delta,
      outputTokens: (row.outputTokens ?? 0) + 1,
      updatedAt: Date.now(),
    });
    return { updated: true };
  },
});

export const noteStreamProgress = internalMutation({
  args: {
    streamId: v.string(),
    route: v.optional(v.string()),
    promptTokens: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const row = await ctx.db
      .query("alphaStreams")
      .withIndex("by_stream", (q) => q.eq("streamId", args.streamId))
      .first();
    if (!row) return { updated: false };
    await ctx.db.patch(row._id, {
      updatedAt: Date.now(),
      ...(args.route !== undefined ? { route: args.route } : {}),
      ...(args.promptTokens !== undefined ? { promptTokens: args.promptTokens } : {}),
    });
    return { updated: true };
  },
});

export const finalizeStream = internalMutation({
  args: {
    streamId: v.string(),
    status: v.string(),
    text: v.optional(v.string()),
    stopReason: v.optional(v.string()),
    outputTokens: v.optional(v.number()),
    error: v.optional(v.string()),
    messageId: v.optional(v.string()),
    latencyMs: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const row = await ctx.db
      .query("alphaStreams")
      .withIndex("by_stream", (q) => q.eq("streamId", args.streamId))
      .first();
    if (!row) return { finalized: false };
    const now = Date.now();
    await ctx.db.patch(row._id, {
      status: args.status,
      updatedAt: now,
      finishedAt: now,
      ...(args.text !== undefined ? { text: args.text } : {}),
      ...(args.stopReason !== undefined ? { stopReason: args.stopReason } : {}),
      ...(args.outputTokens !== undefined ? { outputTokens: args.outputTokens } : {}),
      ...(args.error !== undefined ? { error: args.error } : {}),
      ...(args.messageId !== undefined ? { messageId: args.messageId } : {}),
      ...(args.latencyMs !== undefined ? { latencyMs: args.latencyMs } : {}),
    });
    return { finalized: true };
  },
});

/** Remove one message (used by regenerate, before the new answer is produced). */
export const deleteMessage = internalMutation({
  args: { actorId: v.string(), conversationId: v.string(), messageId: v.string() },
  handler: async (ctx, args) => {
    const row = await ctx.db
      .query("alphaMessages")
      .withIndex("by_message", (q) => q.eq("messageId", args.messageId))
      .first();
    if (!row || row.actorId !== args.actorId || row.conversationId !== args.conversationId) {
      return { deleted: 0 };
    }
    await ctx.db.delete(row._id);
    const conversation = await ctx.db
      .query("alphaConversations")
      .withIndex("by_conversation", (q) => q.eq("conversationId", args.conversationId))
      .first();
    if (conversation) {
      await ctx.db.patch(conversation._id, {
        messageCount: Math.max(0, conversation.messageCount - 1),
        updatedAt: Date.now(),
      });
    }
    return { deleted: 1 };
  },
});

/* -------------------------------------------------------------------------- */
/* Turn plumbing                                                               */
/* -------------------------------------------------------------------------- */

type TranscriptMessage = { messageId: string; role: string; content: string };

/** Only user and assistant text is offered to the context engine as history. */
function toHistory(messages: TranscriptMessage[]): ChatHistoryTurn[] {
  const history: ChatHistoryTurn[] = [];
  for (const message of messages) {
    if (message.role !== "user" && message.role !== "assistant") continue;
    if (!message.content || message.content.trim() === "") continue;
    history.push({ role: message.role, content: message.content });
  }
  return history;
}

async function fetchTranscript(
  ctx: ActionCtx,
  sessionToken: string,
  conversationId: string,
): Promise<{ messages: TranscriptMessage[] } | null> {
  const transcript = await ctx.runQuery(api.alpha.conversations.get, {
    sessionToken,
    conversationId,
  });
  if (!transcript) return null;
  return {
    messages: transcript.messages.map((message) => ({
      messageId: message.messageId,
      role: message.role,
      content: message.content,
    })),
  };
}

/** What `readActorStores` returns: one account's serving inputs, bounded. */
type ActorStores = {
  memories: {
    memoryId: string;
    scope: string;
    sessionId: string | null;
    key: string;
    content: string;
    embedding: number[];
    tags: string[];
    importance: number;
    approved: boolean;
    source: string;
    accessCount: number;
    createdAt: number;
    updatedAt: number;
  }[];
  vectors: {
    recordId: string;
    collection: string;
    dimension: number;
    text: string;
    embedding: number[];
    metadata: Record<string, unknown>;
    sourceId: string | null;
    createdAt: number;
    updatedAt: number;
  }[];
  truncated: { memories: boolean; vectors: boolean };
};

export type ChatTurnOutcome = {
  conversationId: string;
  streamId: string;
  userMessageId: string | null;
  assistantMessageId: string | null;
  requestId: string;
  traceId: string;
  status: "done" | "stopped" | "error";
  stopReason: string | null;
  latencyMs: number;
  tokens: number;
  text: string;
  error: string | null;
};

type TurnPlan = {
  sessionToken: string;
  actorId: string;
  conversationId: string;
  message: string;
  history: ChatHistoryTurn[];
  settings?: SettingsInput;
  /** When set, this assistant message is removed and replaced by the new turn. */
  replaceMessageId?: string;
};

/** One real turn: hydrate, generate, persist, finalize. Nothing is faked here. */
async function runTurn(ctx: ActionCtx, turn: TurnPlan): Promise<ChatTurnOutcome> {
  /* 1. The model. A load failure is a load failure — no generation happens. */
  let runtime: AlphaServingRuntime;
  try {
    runtime = runtimeFor(turn.actorId);
  } catch (error) {
    servingReject("model-unavailable", `Alpha's serving model could not be loaded: ${errorMessage(error)}`);
  }

  /* 2. This account's memory and vectors, with dimension mismatches reported. */
  let stores: ActorStores;
  try {
    stores = await ctx.runQuery(internal.alpha.chat.readActorStores, { actorId: turn.actorId });
  } catch (error) {
    servingReject("hydration-failed", `Alpha could not read this account's memory: ${errorMessage(error)}`);
  }
  const hydrationReport = runtime.hydrateFor(turn.actorId, {
    memories: stores.memories,
    vectors: stores.vectors,
  });

  /* 3. The request, refused with real numbers when it cannot fit the window. */
  let request: RespondRequest;
  try {
    const settings: ChatTurnSettings | undefined = turn.settings as ChatTurnSettings | undefined;
    request = buildChatRequest(runtime, {
      actorId: turn.actorId,
      conversationId: turn.conversationId,
      message: turn.message,
      history: turn.history,
      settings,
    });
  } catch (error) {
    servingReject("invalid-request", errorMessage(error));
  }

  /* 4. The stream row exists before any output, so the client can subscribe. */
  const streamId = alphaId("stream");
  const requestRowId = alphaId("req");
  const sampling = request.sampling ?? runtime.generationDefaults;
  await ctx.runMutation(internal.alpha.chat.createStream, {
    actorId: turn.actorId,
    conversationId: turn.conversationId,
    streamId,
    requestId: requestRowId,
    modelId: runtime.modelId,
    modelStage: runtime.artifact.stage,
    generationConfig: sampling,
  });

  let streamedText = "";
  let userMessageId: string | null = null;
  try {
    if (turn.replaceMessageId) {
      await ctx.runMutation(internal.alpha.chat.deleteMessage, {
        actorId: turn.actorId,
        conversationId: turn.conversationId,
        messageId: turn.replaceMessageId,
      });
    } else {
      const appended = await ctx.runMutation(api.alpha.conversations.appendMessage, {
        sessionToken: turn.sessionToken,
        conversationId: turn.conversationId,
        role: "user",
        content: request.message,
      });
      userMessageId = appended.messageId;
    }

    /* 5. Generation. Deltas are batched into writes; stop is polled in-band. */
    const events = runtime.respondStream(request, {
      shouldStop: async () => {
        const control = await ctx.runQuery(internal.alpha.chat.readStreamControl, { streamId });
        return control?.stopRequested === true;
      },
      shouldStopEvery: 8,
    });

    let pending: string[] = [];
    const flush = async () => {
      if (pending.length === 0) return;
      const delta = pending.join("");
      pending = [];
      streamedText += delta;
      await ctx.runMutation(internal.alpha.chat.appendStreamChunk, { streamId, delta });
    };

    let step = await events.next();
    while (!step.done) {
      const event = step.value;
      if (event.type === "delta") {
        pending.push(event.text);
        // Batch three tokens per write: a write is a round trip, and a batch
        // this small still reads as live typing in the UI.
        if (pending.length >= 3) await flush();
      } else if (event.type === "route") {
        await flush();
        await ctx.runMutation(internal.alpha.chat.noteStreamProgress, {
          streamId,
          route: event.route.decision,
        });
      } else if (event.type === "context") {
        await flush();
        await ctx.runMutation(internal.alpha.chat.noteStreamProgress, {
          streamId,
          promptTokens: event.context.usedTokens,
        });
      }
      step = await events.next();
    }
    await flush();
    const result: RespondResult = step.value;

    /* 6. Persist the assistant message first, then finalize the stream. */
    const tokens = result.generation?.generatedTokens ?? 0;
    const stopReason = result.generation?.stopReason ?? (result.error ? "error" : "done");
    const status: ChatTurnOutcome["status"] = result.error
      ? "error"
      : stopReason === "cancelled"
        ? "stopped"
        : "done";
    const assistant = await ctx.runMutation(api.alpha.conversations.appendMessage, {
      sessionToken: turn.sessionToken,
      conversationId: turn.conversationId,
      role: "assistant",
      content: result.response,
      sources: result.sources.map((source) => ({
        title: source.title,
        score: source.score,
        chunkId: source.chunkId,
      })),
      modelStage: result.modelStage ?? runtime.artifact.stage,
      tokens,
      traceId: result.traceId,
      modelId: runtime.modelId,
      modelVersion: runtime.model.config.version,
      generationConfig: result.generation?.sampling ?? sampling,
      requestId: result.requestId,
      latencyMs: result.durationMs,
      stopReason,
      error: result.error?.message,
    });
    await ctx.runMutation(internal.alpha.chat.finalizeStream, {
      streamId,
      status,
      text: result.response,
      stopReason,
      outputTokens: tokens,
      messageId: assistant.messageId,
      latencyMs: result.durationMs,
      error: result.error?.message,
    });

    /* 7. A run record, so the turn is reviewable after the fact. */
    await ctx.runMutation(api.alpha.observability.recordRun, {
      sessionToken: turn.sessionToken,
      kind: "chat-turn",
      status,
      traceId: result.traceId,
      modelStage: result.modelStage ?? runtime.artifact.stage,
      input: request.message,
      output: result.response,
      metrics: {
        latencyMs: result.durationMs,
        generatedTokens: tokens,
        promptTokens: result.generation?.promptTokens ?? null,
        contextUsedTokens: result.context.usedTokens,
        contextBudgetTokens: result.context.budgetTokens,
        contextDroppedTokens: result.context.droppedTokens,
        route: result.route.decision,
        memories: result.memories.length,
        sources: result.sources.length,
        toolCalls: result.toolCalls.length,
        grounded: result.verification.grounded,
        hydration: {
          ...hydrationReport,
          truncated: stores.truncated,
        },
      },
      error: result.error?.message,
    });

    return {
      conversationId: turn.conversationId,
      streamId,
      userMessageId,
      assistantMessageId: assistant.messageId,
      requestId: result.requestId,
      traceId: result.traceId,
      status,
      stopReason,
      latencyMs: result.durationMs,
      tokens,
      text: result.response,
      error: result.error?.message ?? null,
    };
  } catch (thrown) {
    /* A failed turn is stored as a failed turn. */
    const message = errorMessage(thrown);
    let assistantMessageId: string | null = null;
    try {
      const assistant = await ctx.runMutation(api.alpha.conversations.appendMessage, {
        sessionToken: turn.sessionToken,
        conversationId: turn.conversationId,
        role: "assistant",
        content: streamedText,
        modelStage: runtime.artifact.stage,
        modelId: runtime.modelId,
        modelVersion: runtime.model.config.version,
        requestId: requestRowId,
        stopReason: "error",
        error: message,
      });
      assistantMessageId = assistant.messageId;
      await ctx.runMutation(internal.alpha.chat.finalizeStream, {
        streamId,
        status: "error",
        text: streamedText,
        stopReason: "error",
        messageId: assistantMessageId,
        error: message,
      });
    } catch {
      // Storage itself failed; the stream row may stay "streaming". There is
      // nothing honest left to write, so the failure is returned below.
      try {
        await ctx.runMutation(internal.alpha.chat.finalizeStream, {
          streamId,
          status: "error",
          text: streamedText,
          stopReason: "error",
          error: message,
        });
      } catch {
        // Give up on persistence entirely; the error still reaches the caller.
      }
    }
    return {
      conversationId: turn.conversationId,
      streamId,
      userMessageId,
      assistantMessageId,
      requestId: requestRowId,
      traceId: "",
      status: "error",
      stopReason: "error",
      latencyMs: 0,
      tokens: 0,
      text: streamedText,
      error: message,
    };
  }
}

/* -------------------------------------------------------------------------- */
/* Public actions                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Send one message. Streams through `alphaStreams`; returns when the turn is
 * fully persisted. The client subscribes to `activeStream` while it runs.
 */
export const send = action({
  args: {
    sessionToken: v.string(),
    conversationId: v.optional(v.string()),
    message: v.string(),
    settings: v.optional(settingsValidator),
  },
  handler: async (ctx, args): Promise<ChatTurnOutcome> => {
    const actorId = await actorFromToken(ctx, args.sessionToken);

    let conversationId = args.conversationId;
    let history: ChatHistoryTurn[] = [];
    if (!conversationId) {
      const started = await ctx.runMutation(api.alpha.conversations.start, {
        sessionToken: args.sessionToken,
        kind: "chat",
      });
      conversationId = started.conversationId;
    } else {
      const transcript = await fetchTranscript(ctx, args.sessionToken, conversationId);
      if (!transcript) {
        servingReject("conversation-not-found", "That conversation does not exist for this account.");
      }
      history = toHistory(transcript.messages);
    }

    return await runTurn(ctx, {
      sessionToken: args.sessionToken,
      actorId,
      conversationId,
      message: args.message,
      history,
      settings: args.settings,
    });
  },
});

/**
 * Regenerate the last answer: the previous assistant message is removed and
 * the same user turns are replayed. The old reply is only deleted after the
 * request has been validated, so a refused regeneration changes nothing.
 */
export const regenerate = action({
  args: {
    sessionToken: v.string(),
    conversationId: v.string(),
    settings: v.optional(settingsValidator),
  },
  handler: async (ctx, args): Promise<ChatTurnOutcome> => {
    const actorId = await actorFromToken(ctx, args.sessionToken);
    const transcript = await fetchTranscript(ctx, args.sessionToken, args.conversationId);
    if (!transcript) {
      servingReject("conversation-not-found", "That conversation does not exist for this account.");
    }
    const messages = transcript.messages;
    const lastAssistant = [...messages].reverse().find((message) => message.role === "assistant");
    if (!lastAssistant) {
      servingReject(
        "nothing-to-regenerate",
        "There is no assistant reply in this conversation to regenerate.",
      );
    }
    const assistantIndex = messages.findIndex((m) => m.messageId === lastAssistant.messageId);
    const exchange = messages.slice(0, assistantIndex);
    const userMessage = [...exchange].reverse().find((message) => message.role === "user");
    if (!userMessage) {
      servingReject("nothing-to-regenerate", "There is no user message to answer again.");
    }
    const userIndex = exchange.findIndex((m) => m.messageId === userMessage.messageId);

    return await runTurn(ctx, {
      sessionToken: args.sessionToken,
      actorId,
      conversationId: args.conversationId,
      message: userMessage.content,
      history: toHistory(exchange.slice(0, userIndex)),
      settings: args.settings,
      replaceMessageId: lastAssistant.messageId,
    });
  },
});

