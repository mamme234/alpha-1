/**
 * User data.
 *
 * Alpha's account boundary for the client: the profile behind the current
 * session, and the profile fields an account holder may change about themselves.
 *
 * Identity itself — passwords, sessions, revocation — lives in
 * `alphaAuth/`. This file only reads and edits the account, and it resolves the
 * account from the session on every call, so it can never operate on an id the
 * client supplied.
 */

import { v } from "convex/values";
import type { Id } from "./_generated/dataModel";
import { mutation, query } from "./_generated/server";
import { requireActorId } from "./alpha/helpers";
import { toPublicUser } from "./alphaAuth/sessions";
import { authReject, validateDisplayName } from "./alphaAuth/validation";

/** The signed-in account, or `null` when the session is not valid. */
export const currentUser = query({
  args: { sessionToken: v.string() },
  handler: async (ctx, args) => {
    if (!args.sessionToken) return null;
    try {
      const actorId = await requireActorId(ctx, args.sessionToken);
      const user = await ctx.db.get(actorId as Id<"alphaUsers">);
      return user ? toPublicUser(user) : null;
    } catch {
      // A query that answers "who am I" should report "nobody" rather than
      // failing; the client uses that to drop a stale token.
      return null;
    }
  },
});

/** Change the display name on the signed-in account. */
export const updateDisplayName = mutation({
  args: { sessionToken: v.string(), displayName: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const user = await ctx.db.get(actorId as Id<"alphaUsers">);
    if (!user) authReject("session-invalid", "No valid account was supplied.");
    const displayName = validateDisplayName(args.displayName, user.emailKey);
    await ctx.db.patch(user._id, { displayName, updatedAt: Date.now() });
    return { displayName };
  },
});
