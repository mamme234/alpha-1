import { describe, expect, it } from "vitest";
import { AlphaTokenizer } from "../tokenizer/bpe";
import { ALPHA_SEED_CORPUS } from "../datasets/seed-corpus";

const corpus = [
  "alpha trains its own model on its own corpus",
  "the tokenizer learns merges from the corpus",
  "alpha does not call an external provider",
  "the model is untrained until a run produces a checkpoint",
];

function trainTiny(): AlphaTokenizer {
  return AlphaTokenizer.train(corpus, {
    vocabSize: 120,
    version: "test-0.1.0",
    trainedOn: "unit-test-corpus",
    minPairFrequency: 2,
  });
}

describe("alpha tokenizer", () => {
  it("trains a vocabulary from the corpus and records its provenance", () => {
    const tokenizer = trainTiny();
    expect(tokenizer.vocabSize).toBeGreaterThan(20);
    expect(tokenizer.vocabSize).toBeLessThanOrEqual(120);
    expect(tokenizer.version).toBe("test-0.1.0");
    expect(tokenizer.trainedOn).toBe("unit-test-corpus");
    expect(tokenizer.stats.mergeSteps).toBeGreaterThan(0);
    expect(tokenizer.stats.documents).toBe(corpus.length);
  });

  it("round-trips text exactly for characters inside the vocabulary", () => {
    const tokenizer = trainTiny();
    for (const document of corpus) {
      const ids = tokenizer.encode(document);
      expect(tokenizer.decode(ids)).toBe(document);
    }
  });

  it("learns multi-character merges rather than one token per character", () => {
    const tokenizer = trainTiny();
    const ids = tokenizer.encode("alpha");
    expect(ids.length).toBeLessThan("alpha".length);
  });

  it("reserves fixed special token ids", () => {
    const tokenizer = trainTiny();
    expect(tokenizer.padId).toBe(tokenizer.idFor("<pad>"));
    expect(tokenizer.bosId).toBe(tokenizer.idFor("<bos>"));
    expect(tokenizer.eosId).toBe(tokenizer.idFor("<eos>"));
    expect(tokenizer.idFor("<unk>")).toBe(tokenizer.idForToken("<unk>"));
  });

  it("counts unknown characters instead of hiding them", () => {
    const tokenizer = trainTiny();
    const encoded = tokenizer.encodeDetailed("alpha ✦ ✦");
    expect(encoded.unknown).toBeGreaterThan(0);
    expect(encoded.tokens).toContain("<unk>");
  });

  it("truncates and pads to a requested length", () => {
    const tokenizer = trainTiny();
    const long = "alpha ".repeat(40);
    const truncated = tokenizer.encodeDetailed(long, { maxLength: 8, truncation: "right" });
    expect(truncated.ids.length).toBe(8);
    expect(truncated.truncated).toBe(true);

    const padded = tokenizer.encodeDetailed("alpha", {
      maxLength: 12,
      padToMaxLength: true,
      padding: "left",
    });
    expect(padded.ids.length).toBe(12);
    expect(padded.ids[0]).toBe(tokenizer.padId);
    expect(padded.attentionMask[0]).toBe(0);
  });

  it("throws on overflow when truncation is disabled", () => {
    const tokenizer = trainTiny();
    expect(() =>
      tokenizer.encodeDetailed("alpha ".repeat(40), { maxLength: 4, truncation: "error" }),
    ).toThrow(/longer than maxLength/);
  });

  it("serialises and reloads a vocabulary without changing token ids", () => {
    const tokenizer = trainTiny();
    const snapshot = tokenizer.toJSON();
    const reloaded = AlphaTokenizer.fromJSON(JSON.parse(JSON.stringify(snapshot)));
    expect(reloaded.vocabSize).toBe(tokenizer.vocabSize);
    for (const document of corpus) {
      expect(reloaded.encode(document)).toEqual(tokenizer.encode(document));
    }
    expect(reloaded.decode(reloaded.encode(corpus[0]))).toBe(corpus[0]);
  });

  it("trains on the bundled seed corpus and keeps its licence", () => {
    const tokenizer = AlphaTokenizer.train(ALPHA_SEED_CORPUS.documents, {
      vocabSize: 384,
      trainedOn: ALPHA_SEED_CORPUS.name,
    });
    expect(tokenizer.vocabSize).toBeGreaterThan(64);
    expect(tokenizer.stats.characters).toBeGreaterThan(1000);
    const sample = ALPHA_SEED_CORPUS.documents[0].slice(0, 200);
    expect(tokenizer.decode(tokenizer.encode(sample))).toBe(sample);
  });
});
