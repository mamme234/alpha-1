import { getAuthUserId } from "@convex-dev/auth/server";
import type { MutationCtx, QueryCtx } from "../_generated/server";

/**
 * Every Alpha record is scoped to the signed-in user. This helper is the single
 * place that decision is made, so no Alpha query can accidentally read another
 * user's model, checkpoint, vectors, memories or audit records.
 */
export async function requireActorId(ctx: QueryCtx | MutationCtx): Promise<string> {
  const userId = await getAuthUserId(ctx);
  if (userId === null) {
    throw new Error("Alpha records require a signed-in user");
  }
  return userId as unknown as string;
}

/** Latest timestamp helper so every table stores the same shape of `updatedAt`. */
export function nowMs(): number {
  return Date.now();
}
