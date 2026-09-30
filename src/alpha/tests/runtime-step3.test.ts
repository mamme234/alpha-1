/**
 * Alpha — Step 3 focused tests.
 *
 * These cover the capabilities added for the intelligence runtime: the KV
 * cache and its equivalence to the uncached path, generation cancellation,
 * tool output schemas and timeouts, and the agent sandbox's new limits.
 *
 * Everything runs on Alpha's own model and tokenizer. There are no mocks, and
 * nothing here calls a network service.
 */

import { describe, expect, it } from "vitest";
import {
  AlphaInferenceEngine,
  AlphaPolicyEngine,
  AlphaToolRegistry,
  createGenerationCancellation,
  createKvCache,
  kvCacheDropOldest,
  kvCacheOverflow,
  kvCacheStats,
  resetKvCache,
  SAMPLING_PRESETS,
  withToolTimeout,
  type GenerationResult,
} from "@/alpha";
import { AlphaTransformer } from "../model/transformer";
import { buildModel, buildTokenizer } from "./helpers";

function engines(): { cached: AlphaInferenceEngine; uncached: AlphaInferenceEngine; tokenizer: ReturnType<typeof buildTokenizer> } {
  const tokenizer = buildTokenizer();
  const model = buildModel(tokenizer);
  model.disableGrad();
  return {
    cached: new AlphaInferenceEngine({ model, tokenizer, stage: "trained", useCache: true }),
    uncached: new AlphaInferenceEngine({ model, tokenizer, stage: "trained", useCache: false }),
    tokenizer,
  };
}

const PROMPT = "Alpha is a self owned system";

describe("alpha kv cache", () => {
  it("allocates, resets and reports its own capacity", () => {
    const { tokenizer } = engines();
    const config = buildModel(tokenizer).config;
    const cache = createKvCache(config);
    expect(cache.length).toBe(0);
    expect(cache.capacity).toBe(config.contextLength);
    expect(cache.layers).toHaveLength(config.nLayers);
    expect(cache.bytes).toBe(config.nLayers * config.contextLength * config.dModel * 4 * 2);
    expect(kvCacheOverflow(cache, 5)).toBe(0);
    expect(kvCacheOverflow({ ...cache, length: cache.capacity }, 1)).toBe(1);
    resetKvCache(cache);
    expect(kvCacheStats(cache, true).used).toBe(true);
    expect(kvCacheStats(null, false).used).toBe(false);
  });

  it("drops the oldest positions when asked", () => {
    const { tokenizer } = engines();
    const model = buildModel(tokenizer);
    const config = model.config;
    const cache = createKvCache(config);
    // Write a recognisable value into row 0 of layer 0, then shift.
    cache.layers[0].keys[0] = 42;
    cache.length = 4;
    kvCacheDropOldest(cache, 1);
    expect(cache.length).toBe(3);
    expect(cache.layers[0].keys[0]).not.toBe(42);
  });

  it("produces the same token sequence as the uncached path under greedy", () => {
    const { cached, uncached } = engines();
    for (const maxNewTokens of [1, 4, 12, 24]) {
      const sampling = { ...SAMPLING_PRESETS.greedy, maxNewTokens };
      const a = cached.generate(PROMPT, sampling);
      const b = uncached.generate(PROMPT, sampling);
      expect(a.tokenIds).toEqual(b.tokenIds);
      expect(a.text).toBe(b.text);
      expect(a.stopReason).toBe(b.stopReason);
    }
  });

  it("produces the same token sequence under sampling for a fixed seed", () => {
    const { cached, uncached } = engines();
    const sampling = { ...SAMPLING_PRESETS.balanced, maxNewTokens: 16, seed: 21 };
    expect(cached.generate(PROMPT, sampling).tokenIds).toEqual(
      uncached.generate(PROMPT, sampling).tokenIds,
    );
  });

  it("reports cache usage in the generation result", () => {
    const { cached } = engines();
    const result = cached.generate(PROMPT, { ...SAMPLING_PRESETS.greedy, maxNewTokens: 8 });
    expect(result.cache.used).toBe(true);
    expect(result.cache.positions).toBeGreaterThan(0);
    expect(result.cache.positions).toBeLessThanOrEqual(result.cache.capacity);
    expect(result.cache.writes).toBeGreaterThan(0);
    expect(result.cache.hits).toBeGreaterThanOrEqual(0);
    expect(result.cache.bytes).toBeGreaterThan(0);
    expect(result.timing.prefillMs).toBeGreaterThanOrEqual(0);
    expect(result.timing.decodeMs).toBeGreaterThanOrEqual(0);
  });

  it("records the full metric set on every generation", () => {
    const { cached } = engines();
    const result = cached.generate(PROMPT, { ...SAMPLING_PRESETS.greedy, maxNewTokens: 6 });
    expect(result.generatedTokens).toBe(result.tokenIds.length);
    expect(result.promptTokens).toBeGreaterThan(0);
    expect(result.latencyMs).toBeGreaterThanOrEqual(0);
    expect(result.tokensPerSecond).toBeGreaterThanOrEqual(0);
    expect(typeof result.stopReason).toBe("string");
    expect(result.decoding).toBe("greedy");
  });

  it("is not slower than the uncached path once warmed", () => {
    const { cached, uncached } = engines();
    const sampling = { ...SAMPLING_PRESETS.greedy, maxNewTokens: 24 };
    for (let i = 0; i < 3; i++) {
      cached.generate(PROMPT, sampling);
      uncached.generate(PROMPT, sampling);
    }
    const timeOf = (engine: AlphaInferenceEngine): number => {
      const runs: number[] = [];
      for (let i = 0; i < 5; i++) {
        const started = performance.now();
        engine.generate(PROMPT, sampling);
        runs.push(performance.now() - started);
      }
      runs.sort((a, b) => a - b);
      return runs[2];
    };
    // A speedup is a measured claim, so it is measured here rather than assumed.
    expect(timeOf(cached)).toBeLessThan(timeOf(uncached));
  });
});

describe("alpha generation cancellation", () => {
  it("stops at the next token boundary and says why", () => {
    const { cached } = engines();
    const cancellation = createGenerationCancellation();
    cancellation.cancel("user navigated away");
    const result = cached.generate(PROMPT, { ...SAMPLING_PRESETS.greedy, maxNewTokens: 20 }, { cancellation });
    expect(result.stopReason).toBe("cancelled");
    expect(result.generatedTokens).toBe(0);
    expect(cancellation.cancelled).toBe(true);
    expect(cancellation.reason).toBe("user navigated away");
  });

  it("is a no-op when the request was never cancelled", () => {
    const { cached } = engines();
    const cancellation = createGenerationCancellation();
    const result = cached.generate(PROMPT, { ...SAMPLING_PRESETS.greedy, maxNewTokens: 4 }, { cancellation });
    expect(result.stopReason).not.toBe("cancelled");
  });

  it("halts a stream mid-flight without discarding what was produced", async () => {
    const { cached } = engines();
    const cancellation = createGenerationCancellation();
    const stream = cached.generateStream(
      PROMPT,
      { ...SAMPLING_PRESETS.balanced, maxNewTokens: 20, seed: 3 },
      { cancellation },
    );
    const chunks: string[] = [];
    let next = await stream.next();
    while (!next.done) {
      chunks.push(next.value.token);
      if (chunks.length === 2) cancellation.cancel("stopped by the operator");
      next = await stream.next();
    }
    const result: GenerationResult = next.value;
    expect(result.stopReason).toBe("cancelled");
    // Whatever was produced before the stop is kept, not thrown away.
    expect(result.generatedTokens).toBeGreaterThanOrEqual(chunks.length);
  });
});

describe("alpha inference streaming", () => {
  it("streams tokens and returns the same result as generate()", async () => {
    const { cached } = engines();
    const sampling = { ...SAMPLING_PRESETS.greedy, maxNewTokens: 8 };
    const direct = cached.generate(PROMPT, sampling);
    const stream = cached.generateStream(PROMPT, sampling);
    const tokens: string[] = [];
    let next = await stream.next();
    while (!next.done) {
      tokens.push(next.value.token);
      next = await stream.next();
    }
    expect(tokens.length).toBe(direct.generatedTokens);
    expect(next.value.tokenIds).toEqual(direct.tokenIds);
  });

  it("is deterministic across two identical streams", async () => {
    const { cached } = engines();
    const sampling = { ...SAMPLING_PRESETS.balanced, maxNewTokens: 8, seed: 5 };
    const collect = async (): Promise<number[]> => {
      const stream = cached.generateStream(PROMPT, sampling);
      const ids: number[] = [];
      let next = await stream.next();
      while (!next.done) {
        ids.push(next.value.tokenId);
        next = await stream.next();
      }
      return ids;
    };
    expect(await collect()).toEqual(await collect());
  });
});

describe("alpha inference stop conditions", () => {
  it("stops on a stop token id", () => {
    const { cached } = engines();
    const probe = cached.generate(PROMPT, { ...SAMPLING_PRESETS.greedy, maxNewTokens: 4 });
    const stopAt = probe.tokenIds[1];
    const result = cached.generate(PROMPT, {
      ...SAMPLING_PRESETS.greedy,
      maxNewTokens: 12,
      stopTokenIds: [stopAt],
    });
    expect(result.stopReason).toBe("stop-token");
  });

  it("stops on a stop sequence and trims it from the text", () => {
    const { cached } = engines();
    const result = cached.generate(PROMPT, { ...SAMPLING_PRESETS.greedy, maxNewTokens: 12, stopSequences: ["\n"] });
    if (result.stopReason === "stop-sequence") {
      expect(result.text.endsWith("\n")).toBe(false);
    } else {
      expect(result.stopReason).toBe("max-tokens");
    }
  });

  it("stops at the context window rather than wrapping", () => {
    const { tokenizer, cached } = engines();
    // Ask for more tokens than the window can hold. The engine clamps the
    // request and the decode loop stops at the real limit, so the window is
    // never exceeded — the stop reason says which bound actually applied.
    const result = cached.generate(PROMPT, { ...SAMPLING_PRESETS.greedy, maxNewTokens: 2_000 });
    expect(result.promptTokens + result.generatedTokens).toBeLessThanOrEqual(
      cached.maxContextTokens,
    );
    expect(result.stopReason).toBe("context-limit");
    expect(result.generatedTokens).toBe(cached.maxContextTokens - result.promptTokens);
    expect(tokenizer.vocabSize).toBeGreaterThan(0);
  });

  it("refuses a token budget beyond Alpha's resource ceiling", () => {
    const { cached } = engines();
    expect(() =>
      cached.generate(PROMPT, { ...SAMPLING_PRESETS.greedy, maxNewTokens: 100_000 }),
    ).toThrow(/exceeds Alpha's configured limit/);
  });
});

describe("alpha tool execution boundary", () => {
  function registryWith(definition: Record<string, unknown>): AlphaToolRegistry {
    const policy = new AlphaPolicyEngine();
    policy.assignRole("owner", "owner");
    const registry = new AlphaToolRegistry({ policy });
    registry.setServices({});
    registry.register({
      name: "alpha.test.tool",
      description: "Test tool",
      module: "test",
      inputSchema: { type: "object", properties: {}, required: [] },
      permission: "tool.execute",
      ...definition,
    } as never);
    return registry;
  }

  it("rejects a handler result that does not match its declared output schema", async () => {
    const registry = registryWith({
      outputSchema: {
        type: "object",
        properties: { count: { type: "number" } },
        required: ["count"],
      },
      handler: () => ({ count: "not a number" }),
    });
    const result = await registry.execute("alpha.test.tool", {}, { actorId: "owner" });
    expect(result.ok).toBe(false);
    expect(result.error?.message).toMatch(/validation|number/i);
  });

  it("accepts a handler result that matches its output schema", async () => {
    const registry = registryWith({
      outputSchema: {
        type: "object",
        properties: { count: { type: "number" } },
        required: ["count"],
      },
      handler: () => ({ count: 3 }),
    });
    const result = await registry.execute("alpha.test.tool", {}, { actorId: "owner" });
    expect(result.ok).toBe(true);
    expect(result.output).toEqual({ count: 3 });
  });

  it("stops a handler that overruns its timeout and records it as a timeout", async () => {
    const registry = registryWith({
      timeoutMs: 30,
      handler: () => new Promise((resolve) => setTimeout(() => resolve("late"), 2_000)),
    });
    const started = performance.now();
    const result = await registry.execute("alpha.test.tool", {}, { actorId: "owner" });
    const elapsed = performance.now() - started;
    expect(result.ok).toBe(false);
    expect(result.error?.message).toMatch(/timeout/i);
    expect(result.record.timedOut).toBe(true);
    // It really did stop early rather than waiting for the handler.
    expect(elapsed).toBeLessThan(1_000);
  });

  it("gives a tool a default timeout and reports it on the descriptor", () => {
    const registry = registryWith({ handler: () => "fine" });
    const descriptor = registry.describe().find((tool) => tool.name === "alpha.test.tool");
    expect(descriptor?.timeoutMs).toBeGreaterThan(0);
    expect(descriptor?.outputSchema).toBeNull();
  });

  it("passes work straight through when the timeout is disabled", async () => {
    await expect(withToolTimeout(Promise.resolve("value"), "t", 0)).resolves.toBe("value");
    await expect(withToolTimeout(Promise.resolve("value"), "t", 5_000)).resolves.toBe("value");
  });

  it("never runs a handler for an unpermitted actor", async () => {
    let ran = false;
    const registry = registryWith({
      handler: () => {
        ran = true;
        return "should not happen";
      },
    });
    const result = await registry.execute("alpha.test.tool", {}, { actorId: "stranger" });
    expect(result.ok).toBe(false);
    expect(ran).toBe(false);
  });
});
