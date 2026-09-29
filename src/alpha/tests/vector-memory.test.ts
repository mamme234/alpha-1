import { describe, expect, it } from "vitest";
import { AlphaVectorStore } from "../vector/store";
import { AlphaMemoryStore, ConversationMemory } from "../memory/store";
import { AlphaEmbedder, cosineSimilarity, normalize } from "../embeddings/embedder";
import { AlphaTokenizer } from "../tokenizer/bpe";
import { AlphaTransformer } from "../model/transformer";
import { ALPHA_MODEL_PRESETS } from "../model/config";
import { setGradEnabled } from "../core/tensor";

const corpus = [
  "alpha trains its own model from scratch",
  "the vector store keeps its own vectors",
  "memory is approved before it becomes long term",
  "retrieval assembles context with source references",
];

function buildEmbedder() {
  const tokenizer = AlphaTokenizer.train(corpus, { vocabSize: 160, minPairFrequency: 1 });
  const config = {
    ...ALPHA_MODEL_PRESETS.nano,
    vocabSize: tokenizer.vocabSize,
    contextLength: 32,
    dModel: 32,
    nHeads: 4,
    nLayers: 2,
    dFeedForward: 64,
  };
  const model = new AlphaTransformer(config);
  setGradEnabled(false);
  return { tokenizer, embedder: new AlphaEmbedder({ model, tokenizer, maxTokens: 24 }) };
}

describe("alpha vector store", () => {
  it("creates collections, inserts and rejects dimension mismatches", () => {
    const store = new AlphaVectorStore();
    store.createCollection({ name: "docs", dimension: 3, metric: "cosine" });
    store.insert({ id: "a", collection: "docs", vector: [1, 0, 0], text: "first" });
    expect(store.count("docs")).toBe(1);
    expect(() => store.insert({ id: "b", collection: "docs", vector: [1, 0], text: "bad" })).toThrow(
      /expects 3/,
    );
    expect(() => store.insert({ id: "c", collection: "missing", vector: [1, 0, 0], text: "x" })).toThrow(
      /does not exist/,
    );
  });

  it("searches by cosine similarity and breaks ties deterministically", () => {
    const store = new AlphaVectorStore();
    store.createCollection({ name: "docs", dimension: 2, metric: "cosine" });
    store.insert({ id: "near", collection: "docs", vector: [1, 0], text: "east" });
    store.insert({ id: "mid", collection: "docs", vector: [0.7, 0.7], text: "north east" });
    store.insert({ id: "far", collection: "docs", vector: [0, 1], text: "north" });
    const hits = store.search({ collection: "docs", vector: [1, 0], topK: 2 });
    expect(hits.map((hit) => hit.record.id)).toEqual(["near", "mid"]);
    expect(hits[0].score).toBeCloseTo(1, 6);
    expect(hits[0].rank).toBe(1);
  });

  it("filters by metadata and supports update, delete and source deletion", () => {
    const store = new AlphaVectorStore();
    store.ensureCollection("docs", 2);
    store.insert({ id: "a", collection: "docs", vector: [1, 0], text: "a", sourceId: "doc-1", metadata: { lang: "en" } });
    store.insert({ id: "b", collection: "docs", vector: [0.9, 0.1], text: "b", sourceId: "doc-2", metadata: { lang: "fr" } });
    const filtered = store.search({
      collection: "docs",
      vector: [1, 0],
      topK: 5,
      filter: (record) => record.metadata.lang === "fr",
    });
    expect(filtered.map((hit) => hit.record.id)).toEqual(["b"]);

    store.update("b", { text: "updated", vector: [0, 1] });
    expect(store.get("b")!.text).toBe("updated");
    expect(store.deleteBySource("docs", "doc-1")).toBe(1);
    expect(store.count()).toBe(1);
    expect(store.delete("b")).toBe(true);
    expect(store.count()).toBe(0);
  });

  it("exports and imports a snapshot without losing records", () => {
    const store = new AlphaVectorStore();
    store.ensureCollection("docs", 2);
    store.insert({ id: "a", collection: "docs", vector: [1, 0], text: "a", metadata: { tag: "x" } });
    const snapshot = JSON.parse(JSON.stringify(store.exportSnapshot()));
    const restored = new AlphaVectorStore();
    const imported = restored.importSnapshot(snapshot);
    expect(imported.records).toBe(1);
    expect(restored.get("a")!.metadata.tag).toBe("x");
    expect(restored.getCollection("docs")!.metric).toBe("cosine");
  });
});

describe("alpha embeddings", () => {
  it("produces normalised vectors with the model width as dimension", () => {
    const { embedder } = buildEmbedder();
    const record = embedder.embed("alpha trains its own model");
    expect(record.dimension).toBe(32);
    expect(record.vector).toHaveLength(32);
    const magnitude = Math.sqrt(record.vector.reduce((sum, value) => sum + value * value, 0));
    expect(magnitude).toBeCloseTo(1, 5);
    expect(record.modelStage).toBe("untrained");
    expect(record.tokens).toBeGreaterThan(0);
  });

  it("reports truncation when the text exceeds the embedding window", () => {
    const { embedder } = buildEmbedder();
    const record = embedder.embed("alpha ".repeat(60));
    expect(record.truncated).toBe(true);
    expect(record.tokens).toBeLessThanOrEqual(24);
  });

  it("computes similarity consistently", () => {
    expect(cosineSimilarity([1, 0], [1, 0])).toBeCloseTo(1, 6);
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 6);
    expect(cosineSimilarity([1, 1], [2, 2])).toBeCloseTo(1, 6);
    expect(cosineSimilarity([0, 0], [1, 0])).toBe(0);
    expect(normalize([3, 4])[0]).toBeCloseTo(0.6, 6);
  });

  it("embeds documents and queries with the same encoder", () => {
    const { embedder } = buildEmbedder();
    const docs = embedder.embedDocuments(["alpha trains", "memory is approved"]);
    expect(docs).toHaveLength(2);
    expect(docs[0].vector).toHaveLength(embedder.dimension);
    const query = embedder.embedQuery("alpha trains");
    expect(query.vector).toHaveLength(embedder.dimension);
  });
});

describe("alpha memory", () => {
  it("requires approval for long-term memory", () => {
    const { embedder } = buildEmbedder();
    const memory = new AlphaMemoryStore({ embedder });
    expect(() =>
      memory.write({ scope: "long-term", key: "user.name", content: "Ada" }),
    ).toThrow(/explicit user approval/);
    const record = memory.write({
      scope: "long-term",
      key: "user.name",
      content: "Ada",
      approved: true,
      source: "user",
    });
    expect(record.approved).toBe(true);
    expect(record.scope).toBe("long-term");
  });

  it("requires a session id for conversation and session scopes", () => {
    const { embedder } = buildEmbedder();
    const memory = new AlphaMemoryStore({ embedder });
    expect(() => memory.write({ scope: "session", key: "topic", content: "x" })).toThrow(/requires a sessionId/);
  });

  it("updates an existing memory with the same key instead of duplicating it", () => {
    const { embedder } = buildEmbedder();
    const memory = new AlphaMemoryStore({ embedder });
    memory.write({ scope: "session", sessionId: "s1", key: "topic", content: "first" });
    memory.write({ scope: "session", sessionId: "s1", key: "topic", content: "second" });
    expect(memory.list({ scope: "session" })).toHaveLength(1);
    expect(memory.list({ scope: "session" })[0].content).toBe("second");
  });

  it("scores relevance from similarity, recency, importance and usage", () => {
    const { embedder } = buildEmbedder();
    const memory = new AlphaMemoryStore({ embedder });
    const record = memory.write({
      scope: "session",
      sessionId: "s1",
      key: "k",
      content: "alpha trains its own model",
      importance: 0.9,
    });
    const vector = embedder.embedQuery("alpha trains its own model").vector;
    const scored = memory.score(record, vector);
    expect(scored.similarity).toBeCloseTo(1, 4);
    expect(scored.relevance).toBeGreaterThan(0);
    expect(scored.relevance).toBeLessThanOrEqual(1);
    expect(scored.ageHours).toBeLessThan(0.1);
    record.importance = 0.1;
    expect(memory.score(record, vector).relevance).toBeLessThan(scored.relevance);
  });

  it("retrieves, updates and deletes memory", () => {
    const { embedder } = buildEmbedder();
    const memory = new AlphaMemoryStore({ embedder });
    const a = memory.write({ scope: "session", sessionId: "s1", key: "a", content: "alpha trains its own model" });
    memory.write({ scope: "session", sessionId: "s1", key: "b", content: "memory is approved" });
    const results = memory.retrieve("alpha model", { sessionId: "s1", topK: 2 });
    expect(results.length).toBe(2);
    expect(results[0].relevance).toBeGreaterThanOrEqual(results[1].relevance);
    expect(results[0].record.accessCount).toBe(1);

    memory.update(a.id, { content: "alpha trains its own tokenizer" });
    expect(memory.get(a.id)!.content).toContain("tokenizer");
    expect(memory.forget(a.id)).toBe(true);
    expect(memory.get(a.id)).toBeNull();
    expect(memory.forgetSession("s1")).toBe(1);
    expect(memory.stats().session).toBe(0);
  });

  it("excludes unapproved long-term memories from retrieval by default", () => {
    const { embedder } = buildEmbedder();
    const memory = new AlphaMemoryStore({ embedder });
    const record = memory.write({
      scope: "long-term",
      key: "note",
      content: "alpha stores long term memory",
      approved: true,
    });
    memory.update(record.id, { approved: false });
    expect(memory.retrieve("long term memory", { scopes: ["long-term"] })).toHaveLength(0);
    expect(
      memory.retrieve("long term memory", { scopes: ["long-term"], includeUnapproved: true }),
    ).toHaveLength(1);
  });

  it("keeps a rolling conversation window", () => {
    const conversation = new ConversationMemory("s1", 2);
    conversation.append("user", "first");
    conversation.append("alpha", "second");
    conversation.append("user", "third");
    expect(conversation.length).toBe(2);
    expect(conversation.transcript()).toContain("third");
    expect(conversation.transcript()).not.toContain("first");
    conversation.clear();
    expect(conversation.length).toBe(0);
  });
});
