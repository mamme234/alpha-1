/**
 * Alpha Authentication — primitive-level tests.
 *
 * These cover the parts of the auth boundary that must be right regardless of
 * transport: password derivation and verification, session token generation and
 * hashing, and the input rules the API enforces.
 *
 * The tests exercise the real functions — real PBKDF2 through Web Crypto, real
 * SHA-256, real validation — rather than stand-ins. They live outside
 * `src/convex` on purpose: Convex bundles everything under its functions
 * directory, and test-only imports have no business in a deployment.
 */

import { describe, expect, it } from "vitest";
import {
  ALPHA_PASSWORD_ALGORITHM,
  ALPHA_PASSWORD_ITERATIONS,
  constantTimeEqual,
  createPasswordRecord,
  derivePasswordKey,
  fromBase64Url,
  hashSessionToken,
  newSessionToken,
  toBase64Url,
  verifyPassword,
} from "../../convex/alphaAuth/crypto";
import {
  isValidEmail,
  isLockedOut,
  normalizeEmail,
  validateDisplayName,
  validateEmail,
  validatePassword,
  validateSessionTokenShape,
} from "../../convex/alphaAuth/validation";

/** Assert a call rejects with a specific Alpha auth code. */
function expectRejection(run: () => unknown, code: string): void {
  try {
    run();
  } catch (error) {
    const data = (error as { data?: { code?: string; message?: string } }).data;
    expect(data?.code, `expected rejection code ${code}`).toBe(code);
    expect((data?.message ?? "").length).toBeGreaterThan(0);
    return;
  }
  throw new Error(`expected a rejection with code ${code}`);
}

describe("alpha auth crypto", () => {
  it("round-trips base64url without padding", () => {
    for (const length of [0, 1, 2, 3, 4, 5, 16, 31, 32, 33]) {
      const bytes = new Uint8Array(length);
      for (let i = 0; i < length; i += 1) bytes[i] = (i * 37 + 11) % 256;
      const encoded = toBase64Url(bytes);
      expect(encoded).not.toContain("=");
      expect(encoded).toMatch(/^[A-Za-z0-9_-]*$/);
      expect(Array.from(fromBase64Url(encoded))).toEqual(Array.from(bytes));
    }
  });

  it("hashes a session token deterministically, and never to the token", () => {
    const token = newSessionToken();
    const hash = hashSessionToken(token);
    expect(hash).toBe(hashSessionToken(token));
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
    expect(hash).not.toContain(token);
    expect(hashSessionToken(newSessionToken())).not.toBe(hash);
  });

  it("issues unique, URL-safe session tokens", () => {
    const tokens = new Set<string>();
    for (let i = 0; i < 64; i += 1) {
      const token = newSessionToken();
      expect(token).toMatch(/^[A-Za-z0-9_-]+$/);
      // 32 random bytes → 43 base64url characters.
      expect(token.length).toBeGreaterThanOrEqual(42);
      tokens.add(token);
    }
    expect(tokens.size).toBe(64);
  });

  it("stores a derivation, not the password, with a fresh salt every time", async () => {
    const password = "a-long-enough-passphrase";
    const first = await createPasswordRecord(password);
    const second = await createPasswordRecord(password);

    expect(first.hash).not.toBe(password);
    expect(first.hash).not.toBe(second.hash);
    expect(first.salt).not.toBe(second.salt);
    expect(first.algorithm).toBe(ALPHA_PASSWORD_ALGORITHM);
    expect(first.iterations).toBe(ALPHA_PASSWORD_ITERATIONS);
  }, 60_000);

  it("verifies the right password and rejects a wrong one", async () => {
    const password = "correct-horse-battery";
    const record = await createPasswordRecord(password);

    const good = await verifyPassword(password, record);
    expect(good.valid).toBe(true);
    expect(good.needsRehash).toBe(false);

    const bad = await verifyPassword("correct-horse-batterz", record);
    expect(bad.valid).toBe(false);
  }, 60_000);

  it("verifies an old record at its own cost and asks for a rehash", async () => {
    // An account created when the work factor was lower must keep working, and
    // must be upgraded on the next successful sign-in.
    const password = "legacy-record-password";
    const salt = toBase64Url(new Uint8Array(16).fill(7));
    const hash = await derivePasswordKey(password, salt, 1_000);
    const legacy = { hash, salt, iterations: 1_000, algorithm: ALPHA_PASSWORD_ALGORITHM };

    const result = await verifyPassword(password, legacy);
    expect(result.valid).toBe(true);
    expect(result.needsRehash).toBe(true);

    const upgraded = await createPasswordRecord(password);
    const after = await verifyPassword(password, upgraded);
    expect(after.valid).toBe(true);
    expect(after.needsRehash).toBe(false);
  }, 60_000);

  it("refuses an unknown password algorithm instead of guessing", async () => {
    await expect(
      verifyPassword("whatever-password", {
        hash: "zzz",
        salt: toBase64Url(new Uint8Array(16)),
        iterations: 1000,
        algorithm: "rot13",
      }),
    ).rejects.toThrow(/unsupported password algorithm/);
  });

  it("compares in constant time", () => {
    expect(constantTimeEqual("abc", "abc")).toBe(true);
    expect(constantTimeEqual("abc", "abd")).toBe(false);
    expect(constantTimeEqual("abc", "abcd")).toBe(false);
    expect(constantTimeEqual("", "")).toBe(true);
  });
});

describe("alpha auth validation", () => {
  it("accepts ordinary addresses and rejects malformed ones", () => {
    for (const ok of ["a@b.co", "first.last+tag@sub.example.org"]) {
      expect(isValidEmail(ok), ok).toBe(true);
    }
    for (const bad of [
      "",
      "no-at-sign",
      "two@@example.com",
      "trailing@example.",
      "spaces in@example.com",
      "user@localhost",
      `${"x".repeat(250)}@example.com`,
    ]) {
      expect(isValidEmail(bad), bad).toBe(false);
      expectRejection(() => validateEmail(bad), "invalid-email");
    }
    expect(normalizeEmail("  Person@Example.COM ")).toBe("person@example.com");
  });

  it("enforces the password policy with a code the client can map", () => {
    const email = "operator@example.com";

    expect(validatePassword("a-sufficiently-long-phrase", email)).toBe("a-sufficiently-long-phrase");
    expectRejection(() => validatePassword("short", email), "weak-password");
    expectRejection(() => validatePassword("password123", email), "weak-password");
    expectRejection(() => validatePassword("contains-operatorextra", email), "weak-password");
    expectRejection(() => validatePassword("           ", email), "weak-password");
    expectRejection(() => validatePassword(undefined, email), "invalid-password");
    expectRejection(() => validatePassword("x".repeat(201), email), "weak-password");
  });

  it("rejects session tokens that cannot be Alpha tokens", () => {
    expect(validateSessionTokenShape("AbC-123_xyz")).toBe("AbC-123_xyz");
    expectRejection(() => validateSessionTokenShape(""), "session-invalid");
    expectRejection(() => validateSessionTokenShape("has spaces"), "session-invalid");
    expectRejection(() => validateSessionTokenShape("has/slash+plus="), "session-invalid");
    expectRejection(() => validateSessionTokenShape("x".repeat(513)), "session-invalid");
    expectRejection(() => validateSessionTokenShape(42), "session-invalid");
  });

  it("trims a display name, and refuses control characters", () => {
    expect(validateDisplayName("  Operator ", "operator@example.com")).toBe("Operator");
    expect(validateDisplayName(undefined, "operator@example.com")).toBe("operator");
    expectRejection(() => validateDisplayName("bad\u0000name", "operator@example.com"), "username-invalid");
    expectRejection(() => validateDisplayName("z".repeat(81), "operator@example.com"), "username-invalid");
  });

  it("reports lockout state from the stored deadline", () => {
    expect(isLockedOut(undefined, 1_000)).toBe(false);
    expect(isLockedOut(500, 1_000)).toBe(false);
    expect(isLockedOut(1_500, 1_000)).toBe(true);
  });
});
