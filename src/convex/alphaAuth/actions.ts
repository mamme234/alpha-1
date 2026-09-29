/**
 * Alpha authentication — the operations that touch passwords.
 *
 * These are actions because a password derivation is deliberately expensive and
 * uses a fresh random salt; Convex allows that in an action and forbids it in a
 * deterministic query or mutation. The actual writes go through internal
 * mutations in `sessions.ts`, so there is exactly one place that stores an
 * account or a session.
 *
 * What this file will not do:
 *   - create a privileged account for a special email or a first registrant;
 *   - reveal whether an address has an account when a sign-in fails;
 *   - accept a password that is short, breached, or contains the email address;
 *   - hand out a session without a verified password.
 */

import { v } from "convex/values";
import { action } from "../_generated/server";
import { internal } from "../_generated/api";
import {
  ALPHA_PASSWORD_ALGORITHM,
  ALPHA_PASSWORD_ITERATIONS,
  createPasswordRecord,
  hashSessionToken,
  newSessionToken,
  verifyPassword,
} from "./crypto";
import {
  ALPHA_LOCKOUT_MS,
  ALPHA_MAX_FAILED_SIGN_INS,
  ALPHA_SESSION_TTL_MS,
  authReject,
  isLockedOut,
  normalizeEmail,
  validateDisplayName,
  validateEmail,
  validatePassword,
  validateSessionTokenShape,
  validateUserAgent,
  isValidEmail,
} from "./validation";

/** Same shape as a real record, so a missing account costs the same as a wrong password. */
const DUMMY_PASSWORD_RECORD = {
  hash: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  salt: "AAAAAAAAAAAAAAAAAAAAAA",
  iterations: ALPHA_PASSWORD_ITERATIONS,
  algorithm: ALPHA_PASSWORD_ALGORITHM,
};

const GENERIC_SIGN_IN_FAILURE = "Email or password is incorrect.";

export type AlphaAuthResult = {
  token: string;
  expiresAt: number;
  user: {
    id: string;
    email: string;
    displayName: string;
    role: "user" | "admin";
    status: "active" | "suspended";
    createdAt: number;
    lastSignInAt: number | null;
  };
};

/** Create an account and sign it in. */
export const register = action({
  args: {
    email: v.string(),
    password: v.string(),
    displayName: v.optional(v.string()),
    userAgent: v.optional(v.string()),
  },
  handler: async (ctx, args): Promise<AlphaAuthResult> => {
    const emailKey = validateEmail(args.email);
    const displayName = validateDisplayName(args.displayName, emailKey);
    const password = validatePassword(args.password, emailKey);
    const userAgent = validateUserAgent(args.userAgent);
    const now = Date.now();

    const record = await createPasswordRecord(password);
    const userId = await ctx.runMutation(internal.alphaAuth.sessions.createAccount, {
      email: args.email.trim(),
      emailKey,
      displayName,
      passwordHash: record.hash,
      passwordSalt: record.salt,
      passwordIterations: record.iterations,
      passwordAlgorithm: record.algorithm,
      role: "user",
      now,
    });

    const token = newSessionToken();
    const session = await ctx.runMutation(internal.alphaAuth.sessions.createSession, {
      userId,
      tokenHash: hashSessionToken(token),
      userAgent,
      now,
      ttlMs: ALPHA_SESSION_TTL_MS,
    });
    await ctx.runMutation(internal.alphaAuth.sessions.noteSignInSuccess, { userId, now });

    const current = await ctx.runQuery(internal.alphaAuth.sessions.readAccountForToken, { token });
    if (!current) authReject("session-invalid", "The new session could not be confirmed.");

    return {
      token,
      expiresAt: session.expiresAt,
      user: {
        id: current.user._id as unknown as string,
        email: current.user.email,
        displayName: current.user.displayName,
        role: current.user.role,
        status: current.user.status,
        createdAt: current.user.createdAt,
        lastSignInAt: current.user.lastSignInAt ?? null,
      },
    };
  },
});

/** Sign in with an existing account. */
export const signIn = action({
  args: {
    email: v.string(),
    password: v.string(),
    userAgent: v.optional(v.string()),
  },
  handler: async (ctx, args): Promise<AlphaAuthResult> => {
    const candidate = typeof args.email === "string" ? normalizeEmail(args.email) : "";
    const password = typeof args.password === "string" ? args.password : "";
    const userAgent = validateUserAgent(args.userAgent);
    const now = Date.now();

    // A malformed address cannot match anything, so it fails like a wrong
    // password rather than announcing that the format is the problem.
    if (!isValidEmail(candidate) || password.length === 0) {
      await verifyPassword(password, DUMMY_PASSWORD_RECORD);
      authReject("invalid-credentials", GENERIC_SIGN_IN_FAILURE);
    }

    const account = await ctx.runQuery(internal.alphaAuth.sessions.readAccountForSignIn, { emailKey: candidate });
    if (!account) {
      // Burn the same work a real verification would, then say the same thing.
      await verifyPassword(password, DUMMY_PASSWORD_RECORD);
      authReject("invalid-credentials", GENERIC_SIGN_IN_FAILURE);
    }

    if (account.status !== "active") {
      authReject("account-suspended", "This account is suspended. Contact the operator of this deployment.");
    }
    if (isLockedOut(account.lockedUntil, now)) {
      const minutes = Math.max(1, Math.ceil(((account.lockedUntil ?? now) - now) / 60_000));
      authReject("account-locked", `Too many failed attempts. Try again in ${minutes} minute(s).`);
    }

    const verification = await verifyPassword(password, {
      hash: account.passwordHash,
      salt: account.passwordSalt,
      iterations: account.passwordIterations,
      algorithm: account.passwordAlgorithm,
    });

    if (!verification.valid) {
      const failure = await ctx.runMutation(internal.alphaAuth.sessions.noteSignInFailure, {
        userId: account._id,
        maxFailures: ALPHA_MAX_FAILED_SIGN_INS,
        lockoutMs: ALPHA_LOCKOUT_MS,
        now,
        emailKey: account.emailKey,
      });
      if (failure.lockedUntil) {
        authReject("account-locked", "Too many failed attempts. This account is locked for 15 minutes.");
      }
      authReject("invalid-credentials", GENERIC_SIGN_IN_FAILURE);
    }

    // The password was correct: raise the work factor if ours has moved on.
    if (verification.needsRehash) {
      const upgraded = await createPasswordRecord(password);
      await ctx.runMutation(internal.alphaAuth.sessions.upgradePasswordHash, {
        userId: account._id,
        passwordHash: upgraded.hash,
        passwordSalt: upgraded.salt,
        passwordIterations: upgraded.iterations,
        passwordAlgorithm: upgraded.algorithm,
        now,
      });
    }

    const token = newSessionToken();
    const session = await ctx.runMutation(internal.alphaAuth.sessions.createSession, {
      userId: account._id,
      tokenHash: hashSessionToken(token),
      userAgent,
      now,
      ttlMs: ALPHA_SESSION_TTL_MS,
    });
    await ctx.runMutation(internal.alphaAuth.sessions.noteSignInSuccess, { userId: account._id, now });
    await ctx.runMutation(internal.alphaAuth.sessions.recordSecurityEvent, {
      actorId: account._id as unknown as string,
      emailKey: account.emailKey,
      kind: "signin.succeeded",
      outcome: "allowed",
      detail: userAgent ? `signed in from ${userAgent}` : "signed in",
      at: now,
    });

    return {
      token,
      expiresAt: session.expiresAt,
      user: {
        id: account._id as unknown as string,
        email: account.email,
        displayName: account.displayName,
        role: account.role,
        status: account.status,
        createdAt: account.createdAt,
        lastSignInAt: now,
      },
    };
  },
});

/**
 * Change the password of the signed-in account.
 *
 * Every other session dies with the old password: the account's session epoch
 * is bumped, so a session stolen before the change cannot outlive it. This
 * device is handed a brand new session so the change is not also a sign-out.
 */
export const changePassword = action({
  args: {
    token: v.string(),
    currentPassword: v.string(),
    newPassword: v.string(),
    userAgent: v.optional(v.string()),
  },
  handler: async (ctx, args): Promise<AlphaAuthResult> => {
    const token = validateSessionTokenShape(args.token);
    const resolved = await ctx.runQuery(internal.alphaAuth.sessions.readAccountForToken, { token });
    if (!resolved) authReject("not-signed-in", "Sign in again before changing your password.");

    const account = resolved.user;
    const verification = await verifyPassword(args.currentPassword, {
      hash: account.passwordHash,
      salt: account.passwordSalt,
      iterations: account.passwordIterations,
      algorithm: account.passwordAlgorithm,
    });
    if (!verification.valid) {
      authReject("invalid-credentials", "The current password is incorrect.");
    }

    const nextPassword = validatePassword(args.newPassword, account.emailKey);
    if (nextPassword === args.currentPassword) {
      authReject("password-mismatch", "Choose a password different from the current one.");
    }

    const now = Date.now();
    const record = await createPasswordRecord(nextPassword);
    await ctx.runMutation(internal.alphaAuth.sessions.replacePassword, {
      userId: account._id,
      passwordHash: record.hash,
      passwordSalt: record.salt,
      passwordIterations: record.iterations,
      passwordAlgorithm: record.algorithm,
      now,
    });

    const nextToken = newSessionToken();
    const session = await ctx.runMutation(internal.alphaAuth.sessions.createSession, {
      userId: account._id,
      tokenHash: hashSessionToken(nextToken),
      userAgent: validateUserAgent(args.userAgent),
      now,
      ttlMs: ALPHA_SESSION_TTL_MS,
    });

    return {
      token: nextToken,
      expiresAt: session.expiresAt,
      user: {
        id: account._id as unknown as string,
        email: account.email,
        displayName: account.displayName,
        role: account.role,
        status: account.status,
        createdAt: account.createdAt,
        lastSignInAt: account.lastSignInAt ?? null,
      },
    };
  },
});
