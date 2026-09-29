/**
 * Conversations.
 *
 * A conversation is a stored transcript between an account and Alpha. It is
 * Alpha's own record: the messages are produced by Alpha's inference engine
 * running in the workspace, and each assistant turn carries the `modelStage`
 * that produced it. A transcript therefore cannot be read later as though a
 * finished model had written it when the weights were still random.
 *
 * Ownership comes from the session on every call, so one account can never read
 * or delete another account's conversations.
 */

import { v } from "convex/values";
import { mutation, query } from "../_generated/server";
import { requireActorId } from "./helpers";

/**
 * A readable identifier for a conversation or message. It is a label, not a
 * security boundary — ownership is always the session's `actorId`.
 */
function conversationId(): string {
  return `conv_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function messageId(): string {
  return `msg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

/** Open a conversation. */
export const start = mutation({
  args: {
    sessionToken: v.string(),
    title: v.optional(v.string()),
    kind: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const now = Date.now();
    const id = conversationId();
    await ctx.db.insert("alphaConversations", {
      actorId,
      conversationId: id,
      title: args.title?.trim() || "Untitled conversation",
      kind: args.kind ?? "generate",
      messageCount: 0,
      lastMessageAt: now,
      createdAt: now,
      updatedAt: now,
    });
    return { conversationId: id };
  },
});

/**
 * Append one turn. The first user message names the conversation, so a list of
 * transcripts is readable without opening each one.
 */
export const appendMessage = mutation({
  args: {
    sessionToken: v.string(),
    conversationId: v.string(),
    role: v.union(v.literal("user"), v.literal("assistant"), v.literal("system")),
    content: v.string(),
    sources: v.optional(
      v.array(v.object({ title: v.string(), score: v.float64(), chunkId: v.string() })),
    ),
    modelStage: v.optional(v.string()),
    tokens: v.optional(v.number()),
    traceId: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const conversation = await ctx.db
      .query("alphaConversations")
      .withIndex("by_conversation", (q) => q.eq("conversationId", args.conversationId))
      .first();
    if (!conversation || conversation.actorId !== actorId) {
      throw new Error("That conversation does not belong to this account.");
    }

    const now = Date.now();
    const id = messageId();
    await ctx.db.insert("alphaMessages", {
      actorId,
      conversationId: args.conversationId,
      messageId: id,
      role: args.role,
      content: args.content,
      sources: args.sources ?? [],
      modelStage: args.modelStage,
      tokens: args.tokens,
      traceId: args.traceId,
      createdAt: now,
    });

    const firstUserMessage =
      conversation.messageCount === 0 && args.role === "user" && conversation.title === "Untitled conversation";
    await ctx.db.patch(conversation._id, {
      messageCount: conversation.messageCount + 1,
      lastMessageAt: now,
      updatedAt: now,
      title: firstUserMessage ? args.content.slice(0, 80) : conversation.title,
    });

    return { messageId: id, messageCount: conversation.messageCount + 1 };
  },
});

export const list = query({
  args: { sessionToken: v.string(), limit: v.optional(v.number()) },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = await ctx.db
      .query("alphaConversations")
      .withIndex("by_actor_updated", (q) => q.eq("actorId", actorId))
      .collect();
    return rows
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, args.limit ?? 25)
      .map((row) => ({
        conversationId: row.conversationId,
        title: row.title,
        kind: row.kind,
        messageCount: row.messageCount,
        createdAt: row.createdAt,
        updatedAt: row.updatedAt,
        lastMessageAt: row.lastMessageAt,
      }));
  },
});

export const get = query({
  args: { sessionToken: v.string(), conversationId: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const conversation = await ctx.db
      .query("alphaConversations")
      .withIndex("by_conversation", (q) => q.eq("conversationId", args.conversationId))
      .first();
    if (!conversation || conversation.actorId !== actorId) return null;
    const messages = await ctx.db
      .query("alphaMessages")
      .withIndex("by_conversation", (q) =>
        q.eq("actorId", actorId).eq("conversationId", args.conversationId),
      )
      .collect();
    return {
      conversation: {
        conversationId: conversation.conversationId,
        title: conversation.title,
        kind: conversation.kind,
        createdAt: conversation.createdAt,
        updatedAt: conversation.updatedAt,
      },
      messages: messages.sort((a, b) => a.createdAt - b.createdAt),
    };
  },
});

export const rename = mutation({
  args: { sessionToken: v.string(), conversationId: v.string(), title: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const title = args.title.trim();
    if (!title) throw new Error("A conversation needs a title.");
    const conversation = await ctx.db
      .query("alphaConversations")
      .withIndex("by_conversation", (q) => q.eq("conversationId", args.conversationId))
      .first();
    if (!conversation || conversation.actorId !== actorId) {
      throw new Error("That conversation does not belong to this account.");
    }
    await ctx.db.patch(conversation._id, { title: title.slice(0, 120), updatedAt: Date.now() });
    return { renamed: true };
  },
});

/** Delete a conversation and every message in it. */
export const remove = mutation({
  args: { sessionToken: v.string(), conversationId: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const conversation = await ctx.db
      .query("alphaConversations")
      .withIndex("by_conversation", (q) => q.eq("conversationId", args.conversationId))
      .first();
    if (!conversation || conversation.actorId !== actorId) return { deleted: 0 };
    const messages = await ctx.db
      .query("alphaMessages")
      .withIndex("by_conversation", (q) =>
        q.eq("actorId", actorId).eq("conversationId", args.conversationId),
      )
      .collect();
    for (const message of messages) await ctx.db.delete(message._id);
    await ctx.db.delete(conversation._id);
    return { deleted: messages.length + 1 };
  },
});
