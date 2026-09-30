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
    store.insert({ id: "a", collection: "docs", vector: [1, 0, 0], text: "first", ownerId: "user_1" });
    expect(store.count("docs")).toBe(1);
    expect(() => store.insert({ id: "b", collection: "docs", vector: [1, 0], text: "bad", ownerId: "user_1" })).toThrow(
      /expects 3/,
    );
    expect(() => store.insert({ id: "c", collection: "missing", vector: [1, 0, 0], text: "x", ownerId: "user_1" })).toThrow(
      /does not exist/,
    );
  });

  it("searches by cosine similarity and breaks ties deterministically", () => {
    const store = new AlphaVectorStore();
    store.createCollection({ name: "docs", dimension: 2, metric: "cosine" });
    store.insert({ id: "near", collection: "docs", vector: [1, 0], text: "east", ownerId: "user_1" });
    store.insert({ id: "mid", collection: "docs", vector: [0.7, 0.7], text: "north east", ownerId: "user_1" });
    store.insert({ id: "far", collection: "docs", vector: [0, 1], text: "north", ownerId: "user_1" });
    const hits = store.search({ collection: "docs", vector: [1, 0], topK: 2, ownerId: "user_1" });
    expect(hits.map((hit) => hit.record.id)).toEqual(["near", "mid"]);
    expect(hits[0].score).toBeCloseTo(1, 6);
    expect(hits[0].rank).toBe(1);
  });

  it("filters by metadata and supports update, delete and source deletion", () => {
    const store = new AlphaVectorStore();
    store.ensureCollection("docs", 2);
    store.insert({ id: "a", collection: "docs", vector: [1, 0], text: "a", ownerId: "user_1", sourceId: "doc-1", metadata: { lang: "en" } });
    store.insert({ id: "b", collection: "docs", vector: [0.9, 0.1], text: "b", ownerId: "user_1", sourceId: "doc-2", metadata: { lang: "fr" } });
    const filtered = store.search({
      collection: "docs",
      vector: [1, 0],
      topK: 5,
      ownerId: "user_1",
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
    store.insert({ id: "a", collection: "docs", vector: [1, 0], text: "a", ownerId: "user_1", metadata: { tag: "x" } });
    const snapshot = JSON.parse(JSON.stringify(store.exportSnapshot()));
    const restored = new AlphaVectorStore();
    const imported = restored.importSnapshot(snapshot);
    expect(imported.records).toBe(1);
    expect(restored.get("a")!.metadata.tag).toBe("x");
    expect(restored.getCollection("docs")!.metric).toBe("cosine");
    expect(restored.get("a")!.ownerId).toBe("user_1");
  });

  it("keeps one account's vectors out of another account's retrieval", () => {
    const store = new AlphaVectorStore();
    store.ensureCollection("docs", 2);
    store.insert({ id: "mine", collection: "docs", vector: [1, 0], text: "mine", ownerId: "user_1" });
    store.insert({ id: "theirs", collection: "docs", vector: [1, 0], text: "theirs", ownerId: "user_2" });

    const asOne = store.search({ collection: "docs", vector: [1, 0], topK: 10, ownerId: "user_1" });
    expect(asOne.map((hit) => hit.record.id)).toEqual(["mine"]);

    const asTwo = store.search({ collection: "docs", vector: [1, 0], topK: 10, ownerId: "user_2" });
    expect(asTwo.map((hit) => hit.record.id)).toEqual(["theirs"]);

    // A third account sees nothing at all rather than a shared collection.
    const asThree = store.search({ collection: "docs", vector: [1, 0], topK: 10, ownerId: "user_3" });
    expect(asThree).toEqual([]);

    // Ownership is enforced on read, update and delete, not by convention.
    expect(store.getOwned("mine", "user_2")).toBeNull();
    expect(store.deleteOwned("mine", "user_2")).toBe(false);
    expect(() => store.updateOwned("mine", "user_2", { text: "hijacked" })).toThrow(/not owned/);
    expect(store.get("mine")!.text).toBe("mine");
    expect(store.deleteOwned("mine", "user_1")).toBe(true);
  });

  it("refuses a vector with no owner and records the embedding model", () => {
    const store = new AlphaVectorStore();
    store.ensureCollection("docs", 2);
    expect(() =>
      store.insert({ id: "x", collection: "docs", vector: [1, 0], text: "x", ownerId: "" }),
    ).toThrow(/ownerId/);
    const record = store.insert({
      id: "ok",
      collection: "docs",
      vector: [1, 0],
      text: "ok",
      ownerId: "user_1",
      embedding: { model: "alpha-nano", version: "0.1.0" },
    });
    expect(record.embedding).toEqual({ model: "alpha-nano", version: "0.1.0", dimension: 2 });
    expect(store.collectionsFor("user_1").map((c) => c.name)).toEqual(["docs"]);
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
      memory.write({ scope: "long-term", key: "user.name", content: "Ada", ownerId: "user_1" }),
    ).toThrow(/explicit user approval/);
    const record = memory.write({
      scope: "long-term",
      key: "user.name",
      content: "Ada",
      ownerId: "user_1",
      approved: true,
      source: "user",
      provenance: { origin: "onboarding", referenceId: "conv-1", recordedBy: "user_1" },
    });
    expect(record.approved).toBe(true);
    expect(record.scope).toBe("long-term");
    expect(record.ownerId).toBe("user_1");
    expect(record.provenance).toEqual({
      origin: "onboarding",
      referenceId: "conv-1",
      recordedBy: "user_1",
    });
  });

  it("requires a session id for conversation and session scopes", () => {
    const { embedder } = buildEmbedder();
    const memory = new AlphaMemoryStore({ embedder });
    expect(() => memory.write({ scope: "session", key: "topic", content: "x", ownerId: "user_1" })).toThrow(/requires a sessionId/);
  });

  it("refuses a memory with no owner", () => {
    const { embedder } = buildEmbedder();
    const memory = new AlphaMemoryStore({ embedder });
    expect(() => memory.write({ scope: "session", sessionId: "s1", key: "k", content: "c", ownerId: "" })).toThrow(
      /ownerId/,
    );
  });

  it("updates an existing memory with the same key instead of duplicating it", () => {
    const { embedder } = buildEmbedder();
    const memory = new AlphaMemoryStore({ embedder });
    memory.write({ scope: "session", sessionId: "s1", key: "topic", content: "first", ownerId: "user_1" });
    memory.write({ scope: "session", sessionId: "s1", key: "topic", content: "second", ownerId: "user_1" });
    expect(memory.list({ scope: "session", ownerId: "user_1" })).toHaveLength(1);
    expect(memory.list({ scope: "session", ownerId: "user_1" })[0].content).toBe("second");
  });

  it("keeps one account's memory out of another account's recall", () => {
    const { embedder } = buildEmbedder();
    const memory = new AlphaMemoryStore({ embedder });
    memory.write({
      scope: "long-term",
      key: "secret",
      content: "user one keeps a private preference",
      ownerId: "user_1",
      approved: true,
    });
    memory.write({
      scope: "long-term",
      key: "other",
      content: "user two remembers something else",
      ownerId: "user_2",
      approved: true,
    });

    const asOne = memory.retrieve("private preference", { ownerId: "user_1", topK: 5 });
    expect(asOne.map((entry) => entry.record.ownerId)).toEqual(["user_1"]);

    const asTwo = memory.retrieve("private preference", { ownerId: "user_2", topK: 5 });
    expect(asTwo.every((entry) => entry.record.ownerId === "user_2")).toBe(true);
    expect(asTwo.map((entry) => entry.record.key)).not.toContain("secret");

    expect(memory.retrieve("private preference", { ownerId: "user_3", topK: 5 })).toEqual([]);

    // Approval and deletion are owner-scoped too.
    const theirs = asTwo[0].record;
    expect(() => memory.approve(theirs.id, "user_1")).toThrow(/not owned/);
    expect(memory.forgetOwned(theirs.id, "user_1")).toBe(false);
    expect(memory.forgetOwned(theirs.id, "user_2")).toBe(true);
  });

  it("records provenance for every stored memory", () => {
    const { embedder } = buildEmbedder();
    const memory = new AlphaMemoryStore({ embedder });
    const record = memory.write({
      scope: "session",
      sessionId: "s1",
      key: "k",
      content: "c",
      ownerId: "user_1",
      provenance: { origin: "tool:alpha.memory.write", referenceId: "call-9" },
    });
    expect(record.provenance.origin).toBe("tool:alpha.memory.write");
    expect(record.provenance.referenceId).toBe("call-9");
    expect(record.provenance.recordedBy).toBe("user_1");
    // The default records at least the scope as an origin.
    const other = memory.write({ scope: "session", sessionId: "s1", key: "k2", content: "c2", ownerId: "user_1" });
    expect(other.provenance.origin).toBe("session");
  });

  it("scores relevance from similarity, recency, importance and usage", () => {
    const { embedder } = buildEmbedder();
    const memory = new AlphaMemoryStore({ embedder });
    const record = memory.write({
      scope: "session",
      sessionId: "s1",
      key: "k",
      content: "alpha trains its own model",
      ownerId: "user_1",
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
    const a = memory.write({ scope: "session", sessionId: "s1", key: "a", content: "alpha trains its own model", ownerId: "user_1" });
    memory.write({ scope: "session", sessionId: "s1", key: "b", content: "memory is approved", ownerId: "user_1" });
    const results = memory.retrieve("alpha model", { sessionId: "s1", topK: 2, ownerId: "user_1" });
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
      ownerId: "user_1",
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
