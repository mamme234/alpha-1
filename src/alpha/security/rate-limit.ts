/**
 * Alpha Security — rate limiting.
 *
 * Token buckets, one per actor, refilled continuously. Limits exist because
 * an agent loop that can call tools without bound is a denial-of-service
 * against its own host, not a feature.
 */

import { AlphaRateLimitError } from "../core/errors";

export type RateLimitPolicy = {
  /** Bucket size — the maximum burst. */
  capacity: number;
  /** Tokens restored per second. */
  refillPerSecond: number;
};

export type RateLimitDecision = {
  allowed: boolean;
  remaining: number;
  capacity: number;
  /** Milliseconds until at least one token is available (0 when allowed). */
  retryAfterMs: number;
};

type Bucket = {
  tokens: number;
  lastRefill: number;
  policy: RateLimitPolicy;
};

export const DEFAULT_RATE_LIMITS: Record<string, RateLimitPolicy> = {
  inference: { capacity: 60, refillPerSecond: 1 },
  embedding: { capacity: 200, refillPerSecond: 10 },
  tool: { capacity: 40, refillPerSecond: 0.5 },
  agent: { capacity: 10, refillPerSecond: 0.1 },
  workflow: { capacity: 20, refillPerSecond: 0.25 },
  "tool.execute.dangerous": { capacity: 4, refillPerSecond: 0.05 },
};

export class AlphaRateLimiter {
  private buckets = new Map<string, Bucket>();
  private policies: Record<string, RateLimitPolicy>;

  constructor(policies: Record<string, RateLimitPolicy> = DEFAULT_RATE_LIMITS) {
    this.policies = { ...policies };
  }

  setPolicy(kind: string, policy: RateLimitPolicy): void {
    this.policies[kind] = policy;
  }

  private bucketFor(actorId: string, kind: string): Bucket {
    const key = `${kind}:${actorId}`;
    let bucket = this.buckets.get(key);
    if (!bucket) {
      const policy = this.policies[kind] ?? { capacity: 30, refillPerSecond: 1 };
      bucket = { tokens: policy.capacity, lastRefill: Date.now(), policy };
      this.buckets.set(key, bucket);
    }
    return bucket;
  }

  /** Attempt to consume tokens; returns a decision instead of throwing. */
  consume(actorId: string, kind: string, cost = 1): RateLimitDecision {
    const bucket = this.bucketFor(actorId, kind);
    const now = Date.now();
    const elapsedSeconds = (now - bucket.lastRefill) / 1000;
    bucket.tokens = Math.min(
      bucket.policy.capacity,
      bucket.tokens + elapsedSeconds * bucket.policy.refillPerSecond,
    );
    bucket.lastRefill = now;
    if (bucket.tokens >= cost) {
      bucket.tokens -= cost;
      return {
        allowed: true,
        remaining: Number(bucket.tokens.toFixed(2)),
        capacity: bucket.policy.capacity,
        retryAfterMs: 0,
      };
    }
    const deficit = cost - bucket.tokens;
    return {
      allowed: false,
      remaining: Number(bucket.tokens.toFixed(2)),
      capacity: bucket.policy.capacity,
      retryAfterMs:
        bucket.policy.refillPerSecond > 0
          ? Math.ceil((deficit / bucket.policy.refillPerSecond) * 1000)
          : Number.POSITIVE_INFINITY,
    };
  }

  /** Throwing form used at module boundaries. */
  assert(actorId: string, kind: string, cost = 1): RateLimitDecision {
    const decision = this.consume(actorId, kind, cost);
    if (!decision.allowed) {
      throw new AlphaRateLimitError("security", `rate limit reached for "${kind}"`, {
        actorId,
        kind,
        retryAfterMs: decision.retryAfterMs,
        capacity: decision.capacity,
      });
    }
    return decision;
  }

  reset(actorId?: string): void {
    if (!actorId) {
      this.buckets.clear();
      return;
    }
    for (const key of [...this.buckets.keys()]) {
      if (key.endsWith(`:${actorId}`)) this.buckets.delete(key);
    }
  }
}
