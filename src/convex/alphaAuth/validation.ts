/**
 * Alpha authentication — input validation and account policy.
 *
 * Everything the client sends is checked here before it reaches the database,
 * and every rejection comes back as a structured `ConvexError` with a stable
 * code, so the client can show an exact message without the server guessing.
 *
 * Two rules shape the messages:
 *   1. A failed sign-in never says which half was wrong. "Email or password is
 *      incorrect" for both a missing account and a bad password, so the endpoint
 *      cannot be used to enumerate who has an account.
 *   2. Registration may say an address is taken, because it has to.
 */

import { ConvexError } from "convex/values";

export const ALPHA_AUTH_ERROR_CODES = [
  "invalid-email",
  "invalid-password",
  "weak-password",
  "password-mismatch",
  "username-invalid",
  "email-taken",
  "invalid-credentials",
  "account-suspended",
  "account-locked",
  "session-invalid",
  "session-expired",
  "not-signed-in",
  "forbidden",
] as const;

export type AlphaAuthErrorCode = (typeof ALPHA_AUTH_ERROR_CODES)[number];

export type AlphaAuthErrorData = {
  alphaAuth: true;
  code: AlphaAuthErrorCode;
  message: string;
};

/**
 * Reject with a code the client can act on. Using `ConvexError` means the code
 * and message survive the wire intact instead of being flattened into a generic
 * server error.
 */
export function authReject(code: AlphaAuthErrorCode, message: string): never {
  throw new ConvexError<AlphaAuthErrorData>({ alphaAuth: true, code, message });
}

export const ALPHA_PASSWORD_MIN_LENGTH = 10;
export const ALPHA_PASSWORD_MAX_LENGTH = 200;
export const ALPHA_EMAIL_MAX_LENGTH = 254;
export const ALPHA_DISPLAY_NAME_MAX_LENGTH = 80;

/** Failed sign-ins tolerated before an account is temporarily locked. */
export const ALPHA_MAX_FAILED_SIGN_INS = 8;
/** How long a lockout lasts. */
export const ALPHA_LOCKOUT_MS = 15 * 60 * 1000;
/** Absolute session lifetime. */
export const ALPHA_SESSION_TTL_MS = 30 * 24 * 60 * 60 * 1000;
/** A session with no activity for this long stops working. */
export const ALPHA_SESSION_IDLE_MS = 7 * 24 * 60 * 60 * 1000;

/**
 * Passwords that show up at the top of every breach list. Length rules alone let
 * these through, and they are the first thing anyone tries.
 */
const BLOCKED_PASSWORDS = new Set([
  "password",
  "password1",
  "password123",
  "passw0rd",
  "1234567890",
  "12345678901",
  "123456789012",
  "qwertyuiop",
  "qwerty12345",
  "letmein123",
  "iloveyou123",
  "welcome12345",
  "administrator",
  "changeme123",
  "alphaalpha",
  "alphapassword",
  "correcthorsebatterystaple",
]);

/**
 * Deliberately permissive: one `@`, something either side, at least one dot in
 * the domain, no spaces or control characters. Addresses that match this but are
 * undeliverable are the account holder's problem; inventing stricter rules only
 * locks out real people.
 */
export function isValidEmail(email: string): boolean {
  if (email.length === 0 || email.length > ALPHA_EMAIL_MAX_LENGTH) return false;
  if (/\s/.test(email)) return false;
  const parts = email.split("@");
  if (parts.length !== 2) return false;
  const [local, domain] = parts;
  if (local.length === 0 || domain.length === 0) return false;
  if (!domain.includes(".")) return false;
  if (domain.startsWith(".") || domain.endsWith(".") || domain.includes("..")) return false;
  return /^[^\s@]+@[^\s@.]+(\.[^\s@.]+)+$/.test(email);
}

/** Lowercased + trimmed. The value every lookup and uniqueness check uses. */
export function normalizeEmail(email: string): string {
  return email.trim().toLowerCase();
}

export function validateEmail(email: unknown): string {
  if (typeof email !== "string") authReject("invalid-email", "An email address is required.");
  const normalized = normalizeEmail(email as string);
  if (!isValidEmail(normalized)) {
    authReject("invalid-email", "That does not look like a valid email address.");
  }
  return normalized;
}

export function validatePassword(password: unknown, emailKey: string): string {
  if (typeof password !== "string") {
    authReject("invalid-password", "A password is required.");
  }
  const value = password as string;
  const localPart = emailKey.split("@")[0] ?? "";
  if (value.length < ALPHA_PASSWORD_MIN_LENGTH) {
    authReject("weak-password", `Use at least ${ALPHA_PASSWORD_MIN_LENGTH} characters.`);
  }
  if (value.length > ALPHA_PASSWORD_MAX_LENGTH) {
    authReject("weak-password", `Use at most ${ALPHA_PASSWORD_MAX_LENGTH} characters.`);
  }
  if (value.trim().length === 0) {
    authReject("weak-password", "A password cannot be only whitespace.");
  }
  if (BLOCKED_PASSWORDS.has(value.trim().toLowerCase())) {
    authReject("weak-password", "That password appears in breach lists. Choose another.");
  }
  if (emailKey.length > 0 && value.toLowerCase().includes(emailKey)) {
    authReject("weak-password", "The password must not contain the email address.");
  }
  if (localPart.length >= 3 && value.toLowerCase().includes(localPart.toLowerCase())) {
    authReject("weak-password", "The password must not contain the email address.");
  }
  return value;
}

export function validateDisplayName(value: unknown, emailKey: string): string {
  if (value === undefined || value === null) {
    return emailKey.split("@")[0] || "Alpha user";
  }
  if (typeof value !== "string") {
    authReject("username-invalid", "A display name must be text.");
  }
  const trimmed = (value as string).trim();
  if (trimmed.length === 0) return emailKey.split("@")[0] || "Alpha user";
  if (trimmed.length > ALPHA_DISPLAY_NAME_MAX_LENGTH) {
    authReject("username-invalid", `Keep the display name under ${ALPHA_DISPLAY_NAME_MAX_LENGTH} characters.`);
  }
  // Control characters would corrupt logs and the audit lines that quote names.
  if (/[\u0000-\u001f\u007f]/.test(trimmed)) {
    authReject("username-invalid", "A display name cannot contain control characters.");
  }
  return trimmed;
}

/** A session token is opaque base64url of a known length; anything else is junk. */
export function validateSessionTokenShape(token: unknown): string {
  if (typeof token !== "string" || token.length === 0 || token.length > 512) {
    authReject("session-invalid", "No valid session was supplied.");
  }
  if (!/^[A-Za-z0-9_-]+$/.test(token as string)) {
    authReject("session-invalid", "No valid session was supplied.");
  }
  return token as string;
}

export function validateUserAgent(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim().replace(/[\u0000-\u001f\u007f]/g, " ");
  if (trimmed.length === 0) return undefined;
  return trimmed.slice(0, 200);
}

/** True while a failed-sign-in burst is still cooling down. */
export function isLockedOut(lockedUntil: number | undefined, now: number): boolean {
  return typeof lockedUntil === "number" && lockedUntil > now;
}
