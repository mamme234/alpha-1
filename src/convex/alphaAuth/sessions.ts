/**
 * Alpha authentication — accounts and sessions.
 *
 * This is the only place that decides who an API call is. `requireActorId`
 * (see `../alpha/helpers`) is built on `resolveSession`, so every Alpha record
 * is scoped to a session-verified account rather than to anything the client
 * claims.
 *
 * Sessions are opaque bearer tokens:
 *   - the database stores `sha256(token)`, never the token;
 *   - `expiresAt` gives every session an absolute end;
 *   - `lastSeenAt` plus the idle window stops a forgotten tab living forever;
 *   - `revokedAt` ends one session, and the account's `sessionEpoch` ends all of
 *     them at once (a password change, or "sign out everywhere").
 *
 * Revocation is stored, not implied, so a revoked session shows up in the
 * account's session list with the reason it ended.
 */

import { v } from "convex/values";
import type { Doc } from "../_generated/dataModel";
import { internalMutation, internalQuery, mutation, query, type MutationCtx, type QueryCtx } from "../_generated/server";
import { ALPHA_SESSION_IDLE_MS, ALPHA_SESSION_TTL_MS, authReject, validateSessionTokenShape } from "./validation";
import { hashSessionToken } from "./crypto";

/** What the client is allowed to see about an account. Never the password material. */
export type AlphaPublicUser = {
  id: string;
  email: string;
  displayName: string;
  role: "user" | "admin";
  status: "active" | "suspended";
  createdAt: number;
  lastSignInAt: number | null;
};

export type AlphaSessionSummary = {
  id: string;
  createdAt: number;
  expiresAt: number;
  lastSeenAt: number;
  userAgent: string | null;
  current: boolean;
};

export function toPublicUser(user: Doc<"alphaUsers">): AlphaPublicUser {
  return {
    id: user._id as unknown as string,
    email: user.email,
    displayName: user.displayName,
    role: user.role,
    status: user.status,
    createdAt: user.createdAt,
    lastSignInAt: user.lastSignInAt ?? null,
  };
}

export type ResolvedSession = {
  user: Doc<"alphaUsers">;
  session: Doc<"alphaSessions">;
};

/**
 * Resolve a bearer token to an account, or `null`.
 *
 * Deterministic and read-only, so it is safe in a query: it hashes the token
 * with a pure function and only ever reads. Every reason a session can stop
 * working is checked in one place.
 */
export async function resolveSession(
  ctx: QueryCtx | MutationCtx,
  token: string | undefined,
  now: number = Date.now(),
): Promise<ResolvedSession | null> {
  if (typeof token !== "string" || token.length === 0) return null;
  const session = await ctx.db
    .query("alphaSessions")
    .withIndex("by_token", (q) => q.eq("tokenHash", hashSessionToken(token)))
    .first();
  if (!session) return null;
  if (session.revokedAt !== undefined) return null;
  if (session.expiresAt <= now) return null;
  if (now - session.lastSeenAt > ALPHA_SESSION_IDLE_MS) return null;

  const user = await ctx.db.get(session.userId);
  if (!user) return null;
  // A password change or "sign out everywhere" retires every older session.
  if (session.epoch !== user.sessionEpoch) return null;
  if (user.status !== "active") return null;

  return { user, session };
}

/* -------------------------------------------------------------------------- */
/* Internal API — called by the sign-in/registration actions, never by a client */
/* -------------------------------------------------------------------------- */

export const readAccountForSignIn = internalQuery({
  args: { emailKey: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("alphaUsers")
      .withIndex("by_email", (q) => q.eq("emailKey", args.emailKey))
      .first();
  },
});

/** Resolve a token from inside an action, where `ctx.db` is not available. */
export const readAccountForToken = internalQuery({
  args: { token: v.string() },
  handler: async (ctx, args) => {
    return await resolveSession(ctx, args.token);
  },
});

/**
 * Raise an account's password work factor without ending its sessions. Used when
 * the stored iteration count is below the current setting; verification already
 * used the stored value, so the account keeps working meanwhile.
 */
export const upgradePasswordHash = internalMutation({
  args: {
    userId: v.id("alphaUsers"),
    passwordHash: v.string(),
    passwordSalt: v.string(),
    passwordIterations: v.number(),
    passwordAlgorithm: v.string(),
    now: v.number(),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.userId, {
      passwordHash: args.passwordHash,
      passwordSalt: args.passwordSalt,
      passwordIterations: args.passwordIterations,
      passwordAlgorithm: args.passwordAlgorithm,
      updatedAt: args.now,
    });
  },
});

export const createAccount = internalMutation({
  args: {
    email: v.string(),
    emailKey: v.string(),
    displayName: v.string(),
    passwordHash: v.string(),
    passwordSalt: v.string(),
    passwordIterations: v.number(),
    passwordAlgorithm: v.string(),
    role: v.union(v.literal("user"), v.literal("admin")),
    now: v.number(),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("alphaUsers")
      .withIndex("by_email", (q) => q.eq("emailKey", args.emailKey))
      .first();
    if (existing) authReject("email-taken", "An account already exists for that email address.");

    const userId = await ctx.db.insert("alphaUsers", {
      email: args.email,
      emailKey: args.emailKey,
      displayName: args.displayName,
      passwordHash: args.passwordHash,
      passwordSalt: args.passwordSalt,
      passwordIterations: args.passwordIterations,
      passwordAlgorithm: args.passwordAlgorithm,
      role: args.role,
      status: "active",
      createdAt: args.now,
      updatedAt: args.now,
      failedSignIns: 0,
      sessionEpoch: 1,
    });
    await ctx.db.insert("alphaSecurityEvents", {
      actorId: userId as unknown as string,
      emailKey: args.emailKey,
      kind: "account.created",
      outcome: "succeeded",
      detail: `account created for ${args.emailKey}`,
      at: args.now,
    });
    return userId;
  },
});

export const createSession = internalMutation({
  args: {
    userId: v.id("alphaUsers"),
    tokenHash: v.string(),
    userAgent: v.optional(v.string()),
    now: v.number(),
    ttlMs: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const user = await ctx.db.get(args.userId);
    if (!user) authReject("session-invalid", "No valid account was supplied.");
    const ttl = args.ttlMs ?? ALPHA_SESSION_TTL_MS;
    const sessionId = await ctx.db.insert("alphaSessions", {
      userId: args.userId,
      tokenHash: args.tokenHash,
      epoch: user.sessionEpoch,
      createdAt: args.now,
      expiresAt: args.now + ttl,
      lastSeenAt: args.now,
      userAgent: args.userAgent,
    });
    return { sessionId, expiresAt: args.now + ttl };
  },
});

export const noteSignInSuccess = internalMutation({
  args: { userId: v.id("alphaUsers"), now: v.number() },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.userId, {
      lastSignInAt: args.now,
      updatedAt: args.now,
      failedSignIns: 0,
      lockedUntil: undefined,
    });
  },
});

/**
 * Record a failed attempt and lock the account once the burst is long enough.
 * The lock is stored on the account, so it survives a reload and cannot be
 * cleared by a client that simply retries quickly.
 */
export const noteSignInFailure = internalMutation({
  args: { userId: v.id("alphaUsers"), maxFailures: v.number(), lockoutMs: v.number(), now: v.number(), emailKey: v.string() },
  handler: async (ctx, args) => {
    const user = await ctx.db.get(args.userId);
    if (!user) return { failedSignIns: 0, lockedUntil: null as number | null };
    const failedSignIns = user.failedSignIns + 1;
    const lockedUntil = failedSignIns >= args.maxFailures ? args.now + args.lockoutMs : undefined;
    await ctx.db.patch(args.userId, { failedSignIns, lockedUntil, updatedAt: args.now });
    await ctx.db.insert("alphaSecurityEvents", {
      actorId: args.userId as unknown as string,
      emailKey: args.emailKey,
      kind: lockedUntil ? "account.locked" : "signin.failed",
      outcome: "denied",
      detail: lockedUntil
        ? `locked for ${Math.round(args.lockoutMs / 60000)} minutes after ${failedSignIns} failed sign-ins`
        : `failed sign-in ${failedSignIns}`,
      at: args.now,
    });
    return { failedSignIns, lockedUntil: lockedUntil ?? null };
  },
});

export const replacePassword = internalMutation({
  args: {
    userId: v.id("alphaUsers"),
    passwordHash: v.string(),
    passwordSalt: v.string(),
    passwordIterations: v.number(),
    passwordAlgorithm: v.string(),
    now: v.number(),
  },
  handler: async (ctx, args) => {
    const user = await ctx.db.get(args.userId);
    if (!user) authReject("session-invalid", "No valid account was supplied.");
    // A new epoch retires every session issued before this moment.
    const sessionEpoch = user.sessionEpoch + 1;
    await ctx.db.patch(args.userId, {
      passwordHash: args.passwordHash,
      passwordSalt: args.passwordSalt,
      passwordIterations: args.passwordIterations,
      passwordAlgorithm: args.passwordAlgorithm,
      sessionEpoch,
      updatedAt: args.now,
      failedSignIns: 0,
      lockedUntil: undefined,
    });
    await revokeAllRows(ctx, args.userId, "password changed", args.now);
    await ctx.db.insert("alphaSecurityEvents", {
      actorId: args.userId as unknown as string,
      emailKey: user.emailKey,
      kind: "password.changed",
      outcome: "succeeded",
      detail: "password replaced; all previous sessions revoked",
      at: args.now,
    });
    return { sessionEpoch };
  },
});

export const recordSecurityEvent = internalMutation({
  args: {
    actorId: v.optional(v.string()),
    emailKey: v.optional(v.string()),
    kind: v.string(),
    outcome: v.string(),
    detail: v.string(),
    at: v.number(),
  },
  handler: async (ctx, args) => {
    await ctx.db.insert("alphaSecurityEvents", args);
  },
});

/* -------------------------------------------------------------------------- */
/* Shared revoke helpers                                                       */
/* -------------------------------------------------------------------------- */

async function revokeAllRows(ctx: MutationCtx, userId: Doc<"alphaUsers">["_id"], reason: string, now: number): Promise<number> {
  const rows = await ctx.db
    .query("alphaSessions")
    .withIndex("by_user", (q) => q.eq("userId", userId))
    .collect();
  let revoked = 0;
  for (const row of rows) {
    if (row.revokedAt !== undefined) continue;
    await ctx.db.patch(row._id, { revokedAt: now, revokedReason: reason });
    revoked += 1;
  }
  return revoked;
}

/* -------------------------------------------------------------------------- */
/* Client-facing API                                                           */
/* -------------------------------------------------------------------------- */

/**
 * The current account for a token, or `null`.
 *
 * Deliberately soft: an expired or revoked token is not an error, it is simply
 * not signed in, so the client can drop it and show the sign-in screen.
 */
export const current = query({
  args: { token: v.string() },
  handler: async (ctx, args) => {
    if (!args.token) return null;
    const resolved = await resolveSession(ctx, args.token);
    if (!resolved) return null;
    return {
      user: toPublicUser(resolved.user),
      session: {
        createdAt: resolved.session.createdAt,
        expiresAt: resolved.session.expiresAt,
        lastSeenAt: resolved.session.lastSeenAt,
        userAgent: resolved.session.userAgent ?? null,
      },
    };
  },
});

/** Active sessions for the signed-in account, for the account controls. */
export const listMine = query({
  args: { token: v.string() },
  handler: async (ctx, args): Promise<AlphaSessionSummary[]> => {
    const resolved = await resolveSession(ctx, args.token);
    if (!resolved) return [];
    const now = Date.now();
    const rows = await ctx.db
      .query("alphaSessions")
      .withIndex("by_user", (q) => q.eq("userId", resolved.user._id))
      .collect();
    return rows
      .filter((row) => row.revokedAt === undefined && row.expiresAt > now && row.epoch === resolved.user.sessionEpoch)
      .sort((a, b) => b.lastSeenAt - a.lastSeenAt)
      .map((row) => ({
        id: row._id as unknown as string,
        createdAt: row.createdAt,
        expiresAt: row.expiresAt,
        lastSeenAt: row.lastSeenAt,
        userAgent: row.userAgent ?? null,
        current: row._id === resolved.session._id,
      }));
  },
});

/** Keep the idle window honest without a write on every single call. */
export const touch = mutation({
  args: { token: v.string() },
  handler: async (ctx, args) => {
    const resolved = await resolveSession(ctx, args.token);
    if (!resolved) return { touched: false };
    const now = Date.now();
    if (now - resolved.session.lastSeenAt < 60_000) return { touched: false };
    await ctx.db.patch(resolved.session._id, { lastSeenAt: now });
    return { touched: true };
  },
});

/** End this session. Idempotent: signing out twice is not an error. */
export const signOut = mutation({
  args: { token: v.string() },
  handler: async (ctx, args) => {
    const token = validateSessionTokenShape(args.token);
    const session = await ctx.db
      .query("alphaSessions")
      .withIndex("by_token", (q) => q.eq("tokenHash", hashSessionToken(token)))
      .first();
    if (!session) return { signedOut: true, alreadyEnded: true };
    if (session.revokedAt !== undefined) return { signedOut: true, alreadyEnded: true };
    const now = Date.now();
    await ctx.db.patch(session._id, { revokedAt: now, revokedReason: "signed out" });
    return { signedOut: true, alreadyEnded: false };
  },
});

/** End every session on the account, including any the user has forgotten. */
export const signOutEverywhere = mutation({
  args: { token: v.string() },
  handler: async (ctx, args) => {
    const resolved = await resolveSession(ctx, args.token);
    if (!resolved) authReject("not-signed-in", "Sign in again to end your other sessions.");
    const now = Date.now();
    const nextEpoch = resolved.user.sessionEpoch + 1;
    await ctx.db.patch(resolved.user._id, { sessionEpoch: nextEpoch, updatedAt: now });
    const revoked = await revokeAllRows(ctx, resolved.user._id, "signed out everywhere", now);
    await ctx.db.insert("alphaSecurityEvents", {
      actorId: resolved.user._id as unknown as string,
      emailKey: resolved.user.emailKey,
      kind: "sessions.revoked",
      outcome: "succeeded",
      detail: `${revoked} session(s) ended from the account controls`,
      at: now,
    });
    return { revoked };
  },
});

/** End one other session by id, for the account controls. */
export const revokeSession = mutation({
  args: { token: v.string(), sessionId: v.string() },
  handler: async (ctx, args) => {
    const resolved = await resolveSession(ctx, args.token);
    if (!resolved) authReject("not-signed-in", "Sign in again to manage your sessions.");
    const rows = await ctx.db
      .query("alphaSessions")
      .withIndex("by_user", (q) => q.eq("userId", resolved.user._id))
      .collect();
    const target = rows.find((row) => (row._id as unknown as string) === args.sessionId);
    if (!target) authReject("forbidden", "That session does not belong to this account.");
    if (target.revokedAt !== undefined) return { revoked: false };
    await ctx.db.patch(target._id, { revokedAt: Date.now(), revokedReason: "revoked from account controls" });
    return { revoked: true };
  },
});
