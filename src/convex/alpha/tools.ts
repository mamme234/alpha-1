import { v } from "convex/values";
import { mutation, query } from "../_generated/server";
import { requireActorId } from "./helpers";

/**
 * Registered tools.
 *
 * The registry lives in the client-side Alpha workspace; this table mirrors it
 * so the tool surface (name, description, permission, approval requirement) is
 * reviewable across sessions and visible in the workspace's tool panel.
 */
export const syncTools = mutation({
  args: {
    sessionToken: v.string(),
    tools: v.array(
      v.object({
        name: v.string(),
        description: v.string(),
        module: v.string(),
        permission: v.string(),
        requiresApproval: v.boolean(),
        dangerous: v.boolean(),
        source: v.string(),
        inputSchema: v.any(),
        stats: v.any(),
      }),
    ),
  },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    let written = 0;
    for (const tool of args.tools) {
      const existing = await ctx.db
        .query("alphaTools")
        .withIndex("by_actor_name", (q) => q.eq("actorId", actorId).eq("name", tool.name))
        .first();
      if (existing) {
        await ctx.db.patch(existing._id, { ...tool, updatedAt: Date.now() });
      } else {
        await ctx.db.insert("alphaTools", { ...tool, actorId, updatedAt: Date.now() });
      }
      written++;
    }
    return { written };
  },
});

export const listTools = query({
  args: { sessionToken: v.string() },
  handler: async (ctx, args) => {
    const actorId = await requireActorId(ctx, args.sessionToken);
    const rows = await ctx.db
      .query("alphaTools")
      .withIndex("by_actor", (q) => q.eq("actorId", actorId))
      .collect();
    return rows.sort((a, b) => (a.name < b.name ? -1 : 1));
  },
});
