/**
 * Alpha authentication — cryptographic primitives.
 *
 * Two different jobs, two deliberately different tools:
 *
 *  - **Session tokens** are hashed with a pure-JS SHA-256 from `@oslojs/crypto`.
 *    Resolving a session happens inside queries, which Convex requires to be
 *    deterministic. A synchronous pure function keeps that guarantee, needs no
 *    runtime capability, and behaves identically under Node in the test suite.
 *  - **Passwords** are derived with PBKDF2-HMAC-SHA256 through the Web Crypto
 *    API, which the Convex runtime provides natively. Derivation only ever runs
 *    inside an action, where using a fresh random salt is allowed.
 *
 * Nothing here is a stand-in. A password is stored as a PBKDF2 derivation with
 * a per-account random salt, and the iteration count and algorithm are written
 * next to the hash so verification uses the account's own work factor and the
 * cost can be raised later without invalidating anyone.
 */

import { sha256 } from "@oslojs/crypto/sha2";

/** Recorded on every account alongside the hash. */
export const ALPHA_PASSWORD_ALGORITHM = "PBKDF2-HMAC-SHA256";
/**
 * OWASP's 2023 floor for PBKDF2-HMAC-SHA256. Web Crypto is native in the Convex
 * runtime, so this stays well inside a function's time budget.
 */
export const ALPHA_PASSWORD_ITERATIONS = 600_000;
export const ALPHA_PASSWORD_SALT_BYTES = 16;
export const ALPHA_PASSWORD_KEY_BITS = 256;
/** 32 bytes of entropy; the token is the only copy, the database keeps the hash. */
export const ALPHA_SESSION_TOKEN_BYTES = 32;

const BASE64_URL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
const BASE64_URL_LOOKUP = (() => {
  const table = new Int16Array(128).fill(-1);
  for (let i = 0; i < BASE64_URL_ALPHABET.length; i += 1) {
    table[BASE64_URL_ALPHABET.charCodeAt(i)] = i;
  }
  return table;
})();

/** Unpadded base64url. URL-safe so a token can travel in a query string. */
export function toBase64Url(bytes: Uint8Array): string {
  let out = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i];
    const b1 = i + 1 < bytes.length ? bytes[i + 1] : undefined;
    const b2 = i + 2 < bytes.length ? bytes[i + 2] : undefined;
    out += BASE64_URL_ALPHABET[b0 >> 2];
    out += BASE64_URL_ALPHABET[((b0 & 0x03) << 4) | ((b1 ?? 0) >> 4)];
    if (b1 === undefined) break;
    out += BASE64_URL_ALPHABET[((b1 & 0x0f) << 2) | ((b2 ?? 0) >> 6)];
    if (b2 === undefined) break;
    out += BASE64_URL_ALPHABET[b2 & 0x3f];
  }
  return out;
}

export function fromBase64Url(value: string): Uint8Array<ArrayBuffer> {
  const clean = value.replace(/=+$/, "");
  const out: number[] = [];
  let buffer = 0;
  let bits = 0;
  for (let i = 0; i < clean.length; i += 1) {
    const code = clean.charCodeAt(i);
    const decoded = code < 128 ? BASE64_URL_LOOKUP[code] : -1;
    if (decoded < 0) throw new Error("invalid base64url input");
    buffer = (buffer << 6) | decoded;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push((buffer >> bits) & 0xff);
    }
  }
  return new Uint8Array(out);
}

export function toHex(bytes: Uint8Array): string {
  let out = "";
  for (let i = 0; i < bytes.length; i += 1) {
    out += bytes[i].toString(16).padStart(2, "0");
  }
  return out;
}

/**
 * Cryptographic randomness. Only valid inside an action: Convex's query and
 * mutation runtimes are deterministic by contract, and are not allowed to be
 * the source of a secret.
 */
export function randomBytes(length: number): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  return bytes;
}

/** A fresh session token. Returned to the client once, then never stored raw. */
export function newSessionToken(): string {
  return toBase64Url(randomBytes(ALPHA_SESSION_TOKEN_BYTES));
}

/**
 * The value actually stored for a session. Deterministic and synchronous, so it
 * can be computed while resolving a session inside a query.
 */
export function hashSessionToken(token: string): string {
  return toHex(sha256(new TextEncoder().encode(token)));
}

/**
 * Comparison that does not stop at the first differing character, so the time
 * taken says nothing about how much of a value was guessed correctly.
 */
export function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let difference = 0;
  for (let i = 0; i < a.length; i += 1) {
    difference |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return difference === 0;
}

/**
 * The derivation itself. Exported so a record at an older work factor can be
 * built and verified in tests; `createPasswordRecord` and `verifyPassword` are
 * the two intended entry points.
 */
export async function derivePasswordKey(password: string, salt: string, iterations: number): Promise<string> {
  const saltBytes = fromBase64Url(salt);
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(password),
    "PBKDF2",
    false,
    ["deriveBits"],
  );
  const bits = await crypto.subtle.deriveBits(
    { name: "PBKDF2", hash: "SHA-256", salt: saltBytes, iterations },
    key,
    ALPHA_PASSWORD_KEY_BITS,
  );
  return toBase64Url(new Uint8Array(bits));
}

/** New salt + hash pair for a new or changed password. Actions only. */
export async function createPasswordRecord(
  password: string,
): Promise<{ hash: string; salt: string; iterations: number; algorithm: string }> {
  const salt = toBase64Url(randomBytes(ALPHA_PASSWORD_SALT_BYTES));
  const hash = await derivePasswordKey(password, salt, ALPHA_PASSWORD_ITERATIONS);
  return {
    hash,
    salt,
    iterations: ALPHA_PASSWORD_ITERATIONS,
    algorithm: ALPHA_PASSWORD_ALGORITHM,
  };
}

/**
 * Verify a password against a stored record. The stored iteration count and
 * algorithm decide the work, so accounts created under an older cost still
 * verify (and can be upgraded on next sign-in).
 */
export async function verifyPassword(
  password: string,
  record: { hash: string; salt: string; iterations: number; algorithm: string },
): Promise<{ valid: boolean; needsRehash: boolean }> {
  if (record.algorithm !== ALPHA_PASSWORD_ALGORITHM) {
    throw new Error(`unsupported password algorithm: ${record.algorithm}`);
  }
  const candidate = await derivePasswordKey(password, record.salt, record.iterations);
  return {
    valid: constantTimeEqual(candidate, record.hash),
    needsRehash: record.iterations < ALPHA_PASSWORD_ITERATIONS,
  };
}
