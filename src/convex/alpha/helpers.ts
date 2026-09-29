import { authReject } from "../alphaAuth/validation";
import { resolveSession } from "../alphaAuth/sessions";
import type { MutationCtx, QueryCtx } from "../_generated/server";

/**
 * Alpha's API ownership rule.
 *
 * Every record Alpha stores is scoped to an account, and that account comes
 * from a session — never from an argument the client chooses. `requireActorId`
 * is the single place the decision is made, so no Alpha query can accidentally
 * read another account's model, checkpoint, vectors, memories or audit records.
 *
 * If the session is missing, expired, revoked or retired by a password change,
 * the call is rejected with a code the client can act on.
 */
export async function requireActorId(
  ctx: QueryCtx | MutationCtx,
  sessionToken: string | undefined,
): Promise<string> {
  const resolved = await resolveSession(ctx, sessionToken);
  if (!resolved) {
    authReject("not-signed-in", "Alpha's API requires a signed-in account. Sign in and try again.");
  }
  return resolved.user._id as unknown as string;
}

/**
 * Same resolution, plus a role check. Used for operations that a normal account
 * must not perform — anything that changes how the deployment itself behaves.
 */
export async function requireRole(
  ctx: QueryCtx | MutationCtx,
  sessionToken: string | undefined,
  roles: readonly ("user" | "admin")[],
): Promise<string> {
  const resolved = await resolveSession(ctx, sessionToken);
  if (!resolved) {
    authReject("not-signed-in", "Alpha's API requires a signed-in account. Sign in and try again.");
  }
  if (!roles.includes(resolved.user.role)) {
    authReject("forbidden", "This account does not have permission to perform that operation.");
  }
  return resolved.user._id as unknown as string;
}

/** Latest timestamp helper so every table stores the same shape of `updatedAt`. */
export function nowMs(): number {
  return Date.now();
}
