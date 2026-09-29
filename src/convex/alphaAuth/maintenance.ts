/**
 * Alpha authentication — operator maintenance.
 *
 * These are `internalMutation`s, so no client can reach them: they are callable
 * only from another Convex function or from the deployment CLI with the
 * operator's own credentials. They exist because a system that stores sessions
 * needs a way to clear the ones that have ended, and because operating a
 * deployment occasionally requires removing an account outright.
 *
 * Nothing here is wired into the UI, and nothing here is a back door: there is
 * no path from the Alpha app to either function.
 */

import { v } from "convex/values";
import { internalMutation } from "../_generated/server";

/** Delete session rows that ended or expired more than `olderThanMs` ago. */
export const purgeEndedSessions = internalMutation({
  args: { olderThanMs: v.optional(v.number()) },
  handler: async (ctx, args) => {
    const cutoff = Date.now() - (args.olderThanMs ?? 30 * 24 * 60 * 60 * 1000);
    const rows = await ctx.db.query("alphaSessions").collect();
    let deleted = 0;
    for (const row of rows) {
      const ended = row.revokedAt !== undefined ? row.revokedAt : undefined;
      const expired = row.expiresAt <= cutoff;
      if ((ended !== undefined && ended <= cutoff) || expired) {
        await ctx.db.delete(row._id);
        deleted += 1;
      }
    }
    return { deleted, cutoff };
  },
});

/**
 * Remove an account and everything authentication-related that points at it.
 *
 * Deliberately narrow: it clears the account row, its sessions and its security
 * events. It does **not** walk Alpha's other tables (models, checkpoints,
 * vectors, memories, conversations), so this is not a complete account
 * deletion — it is the authentication-side half, and the docs say so.
 */
export const removeAccount = internalMutation({
  args: { emailKey: v.string() },
  handler: async (ctx, args) => {
    const user = await ctx.db
      .query("alphaUsers")
      .withIndex("by_email", (q) => q.eq("emailKey", args.emailKey))
      .first();
    if (!user) return { removed: false, sessions: 0 };

    const sessions = await ctx.db
      .query("alphaSessions")
      .withIndex("by_user", (q) => q.eq("userId", user._id))
      .collect();
    for (const session of sessions) await ctx.db.delete(session._id);

    const events = await ctx.db.query("alphaSecurityEvents").collect();
    let eventsRemoved = 0;
    for (const event of events) {
      if (event.emailKey === args.emailKey || event.actorId === (user._id as unknown as string)) {
        await ctx.db.delete(event._id);
        eventsRemoved += 1;
      }
    }

    await ctx.db.delete(user._id);
    return { removed: true, sessions: sessions.length, eventsRemoved };
  },
});
