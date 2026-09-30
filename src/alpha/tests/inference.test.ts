import { describe, expect, it } from "vitest";
import { AlphaInferenceEngine, SAMPLING_PRESETS, untrainedWarning } from "../inference/engine";
import { AlphaInferenceService } from "../inference/service";
import { ALPHA_RESOURCE_LIMITS } from "../core/limits";
import { AlphaTransformer } from "../model/transformer";
import { buildModel, buildTokenizer } from "./helpers";

function engine(options: { stage?: "untrained" | "trained"; maxContextTokens?: number } = {}) {
  const tokenizer = buildTokenizer();
  const model = buildModel(tokenizer);
  return {
    tokenizer,
    model,
    engine: new AlphaInferenceEngine({
      model,
      tokenizer,
      stage: options.stage ?? "untrained",
      maxContextTokens: options.maxContextTokens,
    }),
  };
}

describe("alpha inference engine", () => {
  it("generates tokens from Alpha's own weights and reports its stage", () => {
    const { engine: alpha, tokenizer } = engine();
    const result = alpha.generate("Alpha is a self owned", { maxNewTokens: 10, temperature: 0.8, seed: 11 });
    expect(result.generatedTokens).toBeGreaterThan(0);
    expect(result.tokenIds.length).toBe(result.generatedTokens);
    expect(result.text).toBe(tokenizer.decode(result.tokenIds));
    expect(result.modelStage).toBe("untrained");
    expect(result.warning).toMatch(/random initialisation/);
    expect(result.sampling.maxNewTokens).toBe(10);
    expect(result.tokenLogProbs.length).toBe(result.generatedTokens);
    expect(Number.isFinite(result.meanNll)).toBe(true);
    // A pad token must never be produced by free generation.
    expect(result.tokenIds).not.toContain(tokenizer.padId);
  });

  it("is reproducible with a fixed seed and greedy at temperature 0", () => {
    const { engine: alpha } = engine();
    const prompt = "the tokenizer learns merges";
    const greedyA = alpha.generate(prompt, { temperature: 0, maxNewTokens: 12 });
    const greedyB = alpha.generate(prompt, { temperature: 0, maxNewTokens: 12 });
    expect(greedyA.decoding).toBe("greedy");
    expect(greedyA.tokenIds).toEqual(greedyB.tokenIds);
    expect(greedyA.text).toBe(greedyB.text);

    const sampledA = alpha.generate(prompt, { temperature: 0.9, maxNewTokens: 12, seed: 42 });
    const sampledB = alpha.generate(prompt, { temperature: 0.9, maxNewTokens: 12, seed: 42 });
    expect(sampledA.decoding).toBe("sampled");
    expect(sampledA.tokenIds).toEqual(sampledB.tokenIds);

    // Deterministic mode wins over a non-zero temperature.
    const forced = alpha.generate(prompt, { temperature: 0.9, maxNewTokens: 12, deterministic: true });
    expect(forced.decoding).toBe("greedy");
    expect(forced.tokenIds).toEqual(greedyA.tokenIds);
  });

  it("honours the maximum generation length", () => {
    const { engine: alpha } = engine();
    const result = alpha.generate("alpha", { maxNewTokens: 4, temperature: 0 });
    expect(result.generatedTokens).toBeLessThanOrEqual(4);
    if (result.generatedTokens === 4) expect(result.stopReason).toBe("max-tokens");
  });

  it("stops on a stop sequence and trims it from the text", () => {
    const { engine: alpha } = engine();
    const first = alpha.generate("alpha writes", { temperature: 0, maxNewTokens: 3 });
    expect(first.generatedTokens).toBeGreaterThan(1);
    const stopSequence = first.text.slice(0, 2);
    const stopped = alpha.generate("alpha writes", {
      temperature: 0,
      maxNewTokens: 12,
      stopSequences: [stopSequence],
    });
    expect(stopped.stopReason).toBe("stop-sequence");
    expect(stopped.text.endsWith(stopSequence)).toBe(false);
    expect(stopped.text).toBe(first.text.slice(0, stopped.text.length));
  });

  it("stops on a configured stop token", () => {
    const { engine: alpha } = engine();
    const greedy = alpha.generate("alpha writes", { temperature: 0, maxNewTokens: 5 });
    expect(greedy.tokenIds.length).toBeGreaterThan(0);
    const stopToken = greedy.tokenIds[0];
    const stopped = alpha.generate("alpha writes", {
      temperature: 0,
      maxNewTokens: 5,
      stopTokenIds: [stopToken],
    });
    expect(stopped.stopReason).toBe("stop-token");
    expect(stopped.generatedTokens).toBe(0);
  });

  it("stops when the context window is full rather than overflowing it", () => {
    const { engine: alpha } = engine({ maxContextTokens: 3 });
    expect(alpha.maxContextTokens).toBe(3);
    const result = alpha.generate("alpha is a self owned model", { temperature: 0, maxNewTokens: 20 });
    expect(result.promptTokens).toBeLessThanOrEqual(2);
    expect(result.generatedTokens).toBeLessThanOrEqual(2);
    // The window is three tokens, so at most one token can follow a two-token
    // prompt. A stop token ending it first is equally valid and reported.
    expect(["context-limit", "eos"]).toContain(result.stopReason);
    if (result.stopReason === "context-limit") expect(result.generatedTokens).toBe(1);
  });

  it("clamps the completion to the window instead of overflowing it", () => {
    const { engine: alpha } = engine({ maxContextTokens: 8 });
    const sampling = alpha.resolveSampling({ maxNewTokens: 500 });
    expect(sampling.maxNewTokens).toBe(7);
    // The value reported on the result is the configuration actually used.
    const result = alpha.generate("alpha", { maxNewTokens: 500, temperature: 0 });
    expect(result.sampling.maxNewTokens).toBe(7);
    expect(result.generatedTokens).toBeLessThanOrEqual(7);
  });

  it("rejects sampling configurations that cannot work", () => {
    const { engine: alpha } = engine();
    expect(() => alpha.generate("x", { temperature: -1 })).toThrow(/temperature/);
    expect(() => alpha.generate("x", { temperature: 9 })).toThrow(/noise/);
    expect(() => alpha.generate("x", { topP: 0 })).toThrow(/topP/);
    expect(() => alpha.generate("x", { topP: 1.5 })).toThrow(/topP/);
    expect(() => alpha.generate("x", { topK: -1 })).toThrow(/topK/);
    expect(() => alpha.generate("x", { maxNewTokens: 0 })).toThrow(/maxNewTokens/);
    expect(() => alpha.generate("x", { maxNewTokens: ALPHA_RESOURCE_LIMITS.maxNewTokens + 1 })).toThrow(
      /exceeds Alpha's configured limit/,
    );
    expect(() => alpha.generate("x", { seed: Number.NaN })).toThrow(/seed/);
    expect(() => alpha.generate("x", { stopSequences: [1 as unknown as string] })).toThrow(/stopSequences/);
    expect(() => alpha.generate(1 as unknown as string)).toThrow(/prompt must be a string/);
  });

  it("provides named decoding presets", () => {
    expect(SAMPLING_PRESETS.greedy.temperature).toBe(0);
    expect(SAMPLING_PRESETS.greedy.deterministic).toBe(true);
    expect(SAMPLING_PRESETS.creative.temperature).toBeGreaterThan(SAMPLING_PRESETS.balanced.temperature);
    const { engine: alpha } = engine();
    const result = alpha.generate("alpha", { ...SAMPLING_PRESETS.greedy, maxNewTokens: 5 });
    expect(result.decoding).toBe("greedy");
  });

  it("streams the same tokens it returns at the end", () => {
    const { engine: alpha, tokenizer } = engine();
    const config = { temperature: 0, maxNewTokens: 8 };
    const whole = alpha.generate("alpha is a self owned", config);
    return (async () => {
      const chunks: number[] = [];
      const stream = alpha.generateStream("alpha is a self owned", config);
      let next = await stream.next();
      while (!next.done) {
        chunks.push(next.value.tokenId);
        next = await stream.next();
      }
      const final = next.value;
      expect(chunks).toEqual(whole.tokenIds);
      expect(final.text).toBe(whole.text);
      expect(final.generatedTokens).toBe(chunks.length);
      expect(tokenizer.decode(chunks)).toBe(whole.text);
    })();
  });

  it("scores next tokens so diagnostics use the same forward pass", () => {
    const { engine: alpha, tokenizer } = engine();
    const ids = tokenizer.encodeDetailed("alpha", { addBos: true }).ids;
    const logits = alpha.scoreNextTokens(ids);
    expect(logits.length).toBe(tokenizer.vocabSize);
    expect(Array.from(logits).some((value) => value !== 0)).toBe(true);
  });

  it("states the model stage honestly for every stage", () => {
    expect(untrainedWarning("architecture")).toMatch(/random initialisation/);
    expect(untrainedWarning("untrained")).toMatch(/random initialisation/);
    expect(untrainedWarning("trained")).toMatch(/trained from scratch/);
    expect(untrainedWarning("fine-tuned")).toMatch(/fine-tuned/);
    expect(untrainedWarning("production")).toBeNull();
  });
});

describe("alpha inference service", () => {
  it("generates by model id and lists what it holds", () => {
    const tokenizer = buildTokenizer();
    const model = buildModel(tokenizer);
    const service = new AlphaInferenceService();
    const id = AlphaInferenceService.modelId(model.config.name, model.config.version, tokenizer.fingerprint());
    const handle = service.registerModel({
      id,
      name: model.config.name,
      version: model.config.version,
      stage: "untrained",
      model,
      tokenizer,
      contextLength: model.config.contextLength,
      checkpointId: null,
    });
    expect(handle.tokenizerFingerprint).toBe(tokenizer.fingerprint());
    expect(service.defaultModel).toBe(id);
    expect(service.size).toBe(1);

    const result = service.generate({
      modelId: id,
      prompt: "alpha",
      generationConfig: { temperature: 0, maxNewTokens: 5 },
    });
    expect(result.generatedTokens).toBeGreaterThan(0);
    expect(result.modelStage).toBe("untrained");

    // The default model is used when no id is given.
    const byDefault = service.generate({ prompt: "alpha", generationConfig: { temperature: 0, maxNewTokens: 5 } });
    expect(byDefault.tokenIds).toEqual(result.tokenIds);

    const described = service.describe();
    expect(described.registered).toBe(1);
    expect(described.stages.untrained).toBe(1);
    expect(described.models[0].vocabularySize).toBe(tokenizer.vocabSize);
    expect(described.models[0].parameterCount).toBe(model.parameterCount);
  });

  it("fails loudly for an unknown model instead of falling back to another", () => {
    const tokenizer = buildTokenizer();
    const service = new AlphaInferenceService();
    // Nothing registered at all, and a named id that does not exist.
    expect(() => service.generate({ prompt: "alpha" })).toThrow(/no model is registered/);
    expect(() => service.generate({ modelId: "missing", prompt: "alpha" })).toThrow(/unknown modelId/);
    service.registerModel({
      id: "alpha@test",
      name: "alpha",
      version: "test",
      stage: "untrained",
      model: buildModel(tokenizer),
      tokenizer,
      contextLength: 32,
      checkpointId: null,
    });
    expect(() => service.generate({ modelId: "nope", prompt: "alpha" })).toThrow(/unknown modelId/);
    expect(service.has("alpha@test")).toBe(true);
    expect(service.unregisterModel("alpha@test")).toBe(true);
    expect(service.unregisterModel("alpha@test")).toBe(false);
    expect(service.size).toBe(0);
    service.clear();
  });

  it("refuses to register a model whose vocabulary cannot hold the tokenizer", () => {
    const tokenizer = buildTokenizer();
    const service = new AlphaInferenceService();
    const tooSmall = new AlphaTransformer({ ...buildModel(tokenizer).config, vocabSize: 8 });
    expect(tooSmall.config.vocabSize).toBeLessThan(tokenizer.vocabSize);
    expect(() =>
      service.registerModel({
        id: "too-small",
        name: "too-small",
        version: "test",
        stage: "untrained",
        model: tooSmall,
        tokenizer,
        contextLength: tooSmall.config.contextLength,
        checkpointId: null,
      }),
    ).toThrow(/larger than the model vocabulary/);
    expect(service.size).toBe(0);
  });
});
