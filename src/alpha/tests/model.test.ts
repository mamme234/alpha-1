import { describe, expect, it } from "vitest";
import { AlphaTransformer } from "../model/transformer";
import { ALPHA_MODEL_PRESETS, countParameters, describeArchitecture, deriveStage } from "../model/config";
import { backward, setGradEnabled } from "../core/tensor";

const config = { ...ALPHA_MODEL_PRESETS.nano, vocabSize: 64, contextLength: 16, dModel: 32, nHeads: 4, nLayers: 2, dFeedForward: 64 };

describe("alpha transformer", () => {
  it("produces logits with the configured vocabulary size", () => {
    const model = new AlphaTransformer(config);
    const ids = Int32Array.from([1, 2, 3, 4]);
    const out = model.forward(ids, 1, 4);
    expect(out.logits.shape).toEqual([4, config.vocabSize]);
    expect(out.batch).toBe(1);
    expect(out.seq).toBe(4);
  });

  it("counts parameters exactly as the architecture table says", () => {
    const model = new AlphaTransformer(config);
    const fromTable = describeArchitecture(config).reduce((sum, row) => sum + row.parameters, 0);
    expect(model.parameterCount).toBe(fromTable);
    expect(model.parameterCount).toBe(countParameters(config));
    const namedSum = model.parameters().reduce((sum, param) => sum + param.tensor.size, 0);
    // Tied embeddings mean the head shares the token embedding matrix.
    expect(namedSum).toBe(model.parameterCount);
  });

  it("never lets a token attend to a later token", () => {
    setGradEnabled(false);
    const model = new AlphaTransformer(config);
    const base = Int32Array.from([5, 6, 7, 8, 9, 10]);
    const changed = Int32Array.from([5, 6, 7, 8, 9, 33]); // only the last token differs
    const first = model.forward(base, 1, 6);
    const second = model.forward(changed, 1, 6);
    const vocab = config.vocabSize;
    // Logits for positions 0..4 must be identical; position 5 may differ.
    for (let t = 0; t < 5; t++) {
      for (let v = 0; v < vocab; v++) {
        expect(second.logits.data[t * vocab + v]).toBeCloseTo(first.logits.data[t * vocab + v], 6);
      }
    }
    const lastDiffers = Array.from({ length: vocab }, (_, v) => Math.abs(second.logits.data[5 * vocab + v] - first.logits.data[5 * vocab + v])).some((d) => d > 1e-6);
    expect(lastDiffers).toBe(true);
    setGradEnabled(true);
  });

  it("rejects sequences longer than the configured context", () => {
    const model = new AlphaTransformer(config);
    expect(() => model.forward(new Int32Array(config.contextLength + 1), 1, config.contextLength + 1)).toThrow(
      /exceeds context length/,
    );
  });

  it("rejects configurations where the model width does not divide by heads", () => {
    expect(() => new AlphaTransformer({ ...config, dModel: 30, nHeads: 4 })).toThrow(/divisible by nHeads/);
  });

  it("backpropagates a real gradient into the embedding table", () => {
    setGradEnabled(true);
    const model = new AlphaTransformer(config);
    const ids = Int32Array.from([1, 2, 3, 4, 5, 6]);
    const forward = model.forward(ids, 1, 6, { training: false });
    const embeddings = model.parameterMap().get("token_embedding")!;
    embeddings.zeroGrad();
    // d(loss)/d(logit) = 1 for every logit, routed through the real graph.
    backward(forward.logits);
    let nonZero = 0;
    for (let i = 0; i < embeddings.size; i++) {
      if (Math.abs(embeddings.grad![i]) > 1e-9) nonZero++;
    }
    expect(nonZero).toBeGreaterThan(0);
  });

  it("serialises weights and reloads them without changing outputs", () => {
    setGradEnabled(false);
    const model = new AlphaTransformer(config);
    const payload = model.serializeWeights();
    const ids = Int32Array.from([2, 9, 4]);
    const before = model.forward(ids, 1, 3).logits.toArray();

    const reloaded = new AlphaTransformer(config);
    reloaded.loadWeights(payload);
    const after = reloaded.forward(ids, 1, 3).logits.toArray();

    expect(after.length).toBe(before.length);
    for (let i = 0; i < before.length; i++) expect(after[i]).toBeCloseTo(before[i], 6);
    setGradEnabled(true);
  });

  it("derives an honest model stage from what actually exists", () => {
    expect(deriveStage({ hasWeights: false, hasCheckpoint: false, trainedTokens: 0, isFineTune: false, promoted: false })).toBe("architecture");
    expect(deriveStage({ hasWeights: true, hasCheckpoint: false, trainedTokens: 0, isFineTune: false, promoted: false })).toBe("untrained");
    expect(deriveStage({ hasWeights: true, hasCheckpoint: true, trainedTokens: 1024, isFineTune: false, promoted: false })).toBe("trained");
    expect(deriveStage({ hasWeights: true, hasCheckpoint: true, trainedTokens: 1024, isFineTune: true, promoted: false })).toBe("fine-tuned");
    expect(deriveStage({ hasWeights: true, hasCheckpoint: true, trainedTokens: 1024, isFineTune: false, promoted: true })).toBe("production");
  });
});
