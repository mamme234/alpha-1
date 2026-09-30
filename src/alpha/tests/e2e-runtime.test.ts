/**
 * Alpha — end-to-end runtime test.
 *
 * This is the integration test for the whole intelligence runtime:
 *
 *   user request
 *     -> context assembly
 *     -> Alpha's own transformer (trained here, for real, in this file)
 *     -> memory / RAG retrieval when applicable
 *     -> tool request -> authorization -> execution -> result
 *     -> second inference with the tool result
 *     -> verification
 *     -> response
 *
 * There are no mocked AI responses anywhere in this file. The model is trained
 * by a real `AlphaTrainer` run over the seeded corpus, the tokenizer is trained
 * by a real `AlphaTokenizer.train`, the embeddings come from that model's own
 * hidden states, and retrieval is a real vector scan. Mocks appear only where
 * the thing under test is genuinely outside Alpha (there are none here).
 */

import { beforeAll, describe, expect, it } from "vitest";
import { AlphaAiRuntime } from "../runtime/orchestrator";
import { AlphaAgentRuntime } from "../agents/runtime";
import { AlphaAuditLog } from "../security/audit";
import { AlphaContextEngine } from "../context/engine";
import { AlphaEmbedder } from "../embeddings/embedder";
import { AlphaInferenceEngine, SAMPLING_PRESETS } from "../inference/engine";
import { AlphaMemoryStore } from "../memory/store";
import { ALPHA_MODEL_PRESETS } from "../model/config";
import { AlphaTransformer } from "../model/transformer";
import { AlphaPolicyEngine } from "../security/policy";
import { AlphaRateLimiter } from "../security/rate-limit";
import { AlphaToolRegistry } from "../tools/registry";
import { registerBuiltinTools } from "../tools/builtin";
import { AlphaRagPipeline } from "../rag/pipeline";
import { AlphaVectorStore } from "../vector/store";
import { AlphaTokenizer } from "../tokenizer/bpe";
import { AlphaTrainer } from "../training/trainer";
import { ALPHA_SEED_CORPUS } from "../datasets/seed-corpus";
import { assertResourceLimit } from "../core/limits";

/**
 * A real, small model trained on the shipped corpus. `beforeAll` performs an
 * actual optimisation run so nothing downstream is operating on random weights
 * dressed up as a trained model.
 */
type TrainedStack = {
  tokenizer: AlphaTokenizer;
  model: AlphaTransformer;
  embedder: AlphaEmbedder;
  inference: AlphaInferenceEngine;
  memory: AlphaMemoryStore;
  store: AlphaVectorStore;
  rag: AlphaRagPipeline;
  context: AlphaContextEngine;
  registry: AlphaToolRegistry;
  policy: AlphaPolicyEngine;
  audit: AlphaAuditLog;
  agents: AlphaAgentRuntime;
  runtime: AlphaAiRuntime;
  training: { firstLoss: number; lastLoss: number; uniformLoss: number; steps: number };
  auditEntries: () => number;
};

let stack: TrainedStack;

function buildTrainedStack(): TrainedStack {
  const tokenizer = AlphaTokenizer.train(ALPHA_SEED_CORPUS.documents, {
    vocabSize: 256,
    minPairFrequency: 1,
    version: "e2e",
    trainedOn: `${ALPHA_SEED_CORPUS.name}@${ALPHA_SEED_CORPUS.version}`,
  });
  const config = {
    ...ALPHA_MODEL_PRESETS.nano,
    vocabSize: tokenizer.vocabSize,
    contextLength: 128,
    dModel: 48,
    nHeads: 4,
    nLayers: 2,
    dFeedForward: 96,
  };
  assertResourceLimit("maxVocabSize", config.vocabSize, "e2e");
  const model = new AlphaTransformer(config, 1337);

  // A real training run: backpropagation, AdamW, the whole loop.
  const trainer = new AlphaTrainer({
    model,
    tokenizer,
    dataset: ALPHA_SEED_CORPUS,
    config: {
      batchSize: 2,
      seqLen: 24,
      totalSteps: 12,
      learningRate: 3e-3,
      warmupSteps: 2,
      evalInterval: 0,
      evalBatches: 2,
      checkpointInterval: 0,
      validationFraction: 0.12,
      seed: 1337,
    },
  });
  const summary = trainer.trainToCompletion();
  const first = trainer.history[0]?.loss ?? summary.firstLoss;

  // Inference runs against a frozen copy so a later run cannot disturb it.
  const inferenceModel = model.snapshot();
  const embedder = new AlphaEmbedder({ model: inferenceModel, tokenizer, maxTokens: 24 });
  const inference = new AlphaInferenceEngine({ model: inferenceModel, tokenizer, stage: "trained" });
  const memory = new AlphaMemoryStore({ embedder });
  const store = new AlphaVectorStore();
  const context = new AlphaContextEngine({ tokenizer, contextLength: 128 });
  const rag = new AlphaRagPipeline({
    tokenizer,
    embedder,
    store,
    inference,
    config: { chunkTokens: 16, overlapTokens: 4, topK: 3, maxContextTokens: 64, minScore: -1 },
  });

  const policy = new AlphaPolicyEngine();
  const audit = new AlphaAuditLog();
  const registry = new AlphaToolRegistry({
    policy,
    rateLimiter: new AlphaRateLimiter(),
    audit,
  });
  registerBuiltinTools(registry);
  registry.setServices({ tokenizer, embedder, vectorStore: store, memory, inference });

  policy.assignRole("owner", "owner");
  policy.assignRole("user_1", "owner");
  policy.assignRole("user_2", "operator");
  policy.assignRole("alpha.agent", "agent");
  policy.registerAgentScope({
    agentId: "alpha.agent",
    permissions: [
      "model.read",
      "inference.run",
      "vector.read",
      "rag.query",
      "memory.read",
      "memory.write",
      "tool.execute",
    ],
    allowedTools: ["alpha.text.stats", "alpha.calculator", "alpha.tokenizer.analyze"],
    maxSteps: 3,
  });

  const agents = new AlphaAgentRuntime({ registry, policy, inference, memory, audit });
  const runtime = new AlphaAiRuntime({
    inference,
    context,
    memory,
    rag,
    tools: registry,
    agents,
    policy,
    rateLimiter: new AlphaRateLimiter(),
    audit,
    // A short instruction: this model's window is 128 tokens, and the pinned
    // instruction must fit inside it or assembly fails loudly by design.
    systemInstruction: "Answer only from the context. If it lacks the answer, say so.",
  });

  return {
    tokenizer,
    model,
    embedder,
    inference,
    memory,
    store,
    rag,
    context,
    registry,
    policy,
    audit,
    agents,
    runtime,
    training: {
      firstLoss: first,
      lastLoss: summary.lastLoss ?? first,
      uniformLoss: summary.uniformLossBaseline,
      steps: 12,
    },
    auditEntries: () => audit.list({ limit: 1000 }).length,
  };
}

beforeAll(() => {
  stack = buildTrainedStack();
}, 120_000);

describe("alpha end-to-end runtime", () => {
  it("trained a real model before running anything", () => {
    // If this fails, everything below would be testing random weights.
    expect(stack.training.lastLoss).toBeLessThan(stack.training.firstLoss);
    expect(stack.training.lastLoss).toBeLessThan(stack.training.uniformLoss);
    expect(stack.inference.stage).toBe("trained");
  });

  it("answers a plain request from Alpha's own weights", async () => {
    // Sampled rather than greedy: at 12 training steps this model puts high
    // probability on <eos>, so greedy decoding legitimately returns an empty
    // completion. Sampling is deterministic for a fixed seed, and the point of
    // this test is that the path runs on real weights.
    const result = await stack.runtime.respond({
      message: "Alpha is a self owned",
      actorId: "user_1",
      sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 8, seed: 7 },
    });

    expect(result.error).toBeNull();
    expect(result.route.decision).toBe("inference");
    // The text came from the model, so it must correspond to real token ids.
    expect(result.generation).not.toBeNull();
    expect(result.generation!.generatedTokens).toBeGreaterThan(0);
    expect(result.generation!.tokenIds.length).toBe(result.generation!.generatedTokens);
    expect(result.response).toBe(result.generation!.text);
    expect(result.modelStage).toBe("trained");
    expect(result.requestId).toMatch(/^req_/);
    // Nothing was retrieved, so Alpha must say the answer is ungrounded.
    expect(result.verification.grounded).toBe(false);
    expect(result.verification.hasSources).toBe(false);
  });

  it("is deterministic for a given seed", async () => {
    const sampling = { ...SAMPLING_PRESETS.balanced, maxNewTokens: 8, seed: 11 };
    const first = await stack.runtime.respond({ message: "Alpha is a self owned", actorId: "user_1", sampling });
    const second = await stack.runtime.respond({ message: "Alpha is a self owned", actorId: "user_1", sampling });
    expect(first.generation!.tokenIds).toEqual(second.generation!.tokenIds);
  });

  it("grounds an answer in an ingested document and cites it", async () => {
    stack.rag.ingest({
      title: "Alpha training",
      content:
        "alpha trains its own model with adamw and backpropagation. " +
        "checkpoints store the weights, the optimiser moments and the random number generator position " +
        "so that a training run can resume instead of restarting.",
      ownerId: "user_1",
      license: "CC0-1.0",
    });

    const result = await stack.runtime.respond({
      message: "how does alpha resume a training run",
      actorId: "user_1",
      useRetrieval: true,
      sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 8, seed: 7 },
    });

    expect(result.route.decision).toBe("retrieval");
    expect(result.sources.length).toBeGreaterThan(0);
    expect(result.verification.hasSources).toBe(true);
    expect(result.verification.sourceCount).toBe(result.sources.length);
    // Every citation points at a chunk that was actually retrieved.
    for (const source of result.sources) {
      expect(source.chunkId).toBeTruthy();
      expect(source.documentId).toBeTruthy();
      expect(source.title).toBe("Alpha training");
      expect(typeof source.score).toBe("number");
    }
    // The retrieved content is in the assembled context, fenced as data.
    const sourcesBlock = result.context.blocks.find((block) => block.kind === "sources");
    expect(sourcesBlock?.includedTokens).toBeGreaterThan(0);
  });

  it("does not let one account's documents reach another account", async () => {
    const result = await stack.runtime.respond({
      message: "how does alpha resume a training run",
      actorId: "user_2",
      useRetrieval: true,
      sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 4, seed: 7 },
    });
    expect(result.sources).toEqual([]);
    expect(result.verification.hasSources).toBe(false);
    // A retrieval miss is stated, not papered over.
    expect(result.verification.notes.join(" ")).toMatch(/no indexed document matched/i);
  });

  it("recalls approved memory for the owner only", async () => {
    stack.memory.write({
      scope: "long-term",
      key: "user.preference",
      content: "the user prefers short answers",
      ownerId: "user_1",
      approved: true,
      source: "user",
    });

    const mine = await stack.runtime.respond({
      message: "the user prefers short answers",
      actorId: "user_1",
      useMemory: true,
      sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 4, seed: 7 },
    });
    expect(mine.route.decision).toBe("memory");
    expect(mine.memories.length).toBeGreaterThan(0);
    expect(mine.memories.every((entry) => entry.key === "user.preference")).toBe(true);

    const theirs = await stack.runtime.respond({
      message: "the user prefers short answers",
      actorId: "user_2",
      useMemory: true,
      sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 4, seed: 7 },
    });
    expect(theirs.memories).toEqual([]);
  });

  it("runs a permitted tool and feeds the result back into context", async () => {
    const result = await stack.runtime.respond({
      message: "please give me some text statistics",
      actorId: "user_1",
      allowedTools: ["alpha.text.stats"],
      toolRequest: { name: "alpha.text.stats", args: { text: "alpha owns its stack" } },
      sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 4, seed: 7 },
    });

    expect(result.route.decision).toBe("tool");
    expect(result.toolCalls.length).toBe(1);
    const call = result.toolCalls[0];
    expect(call.tool).toBe("alpha.text.stats");
    expect(call.actorId).toBe("user_1");
    expect(call.ok).toBe(true);
    // The tool really ran: its output carries the real counts.
    expect((call.output as { characters: number }).characters).toBe("alpha owns its stack".length);
    // The second inference happened with the tool result in context.
    expect(result.generation).not.toBeNull();
  });

  it("refuses a tool request that is not on the allow-list", async () => {
    const result = await stack.runtime.respond({
      // The caller names a tool they did not permit.
      message: "run the admin tool",
      actorId: "user_1",
      allowedTools: ["alpha.text.stats"],
      toolRequest: { name: "alpha.admin.clear_vector_store", args: { confirm: true } },
      sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 4, seed: 7 },
    });
    expect(result.toolCalls).toEqual([]);
    expect(result.error).toBeNull();
  });

  it("records a failed tool call as a failure rather than an answer", async () => {
    const result = await stack.runtime.respond({
      message: "analyse some text",
      actorId: "user_1",
      allowedTools: ["alpha.text.stats"],
      // Missing the required `text` argument: schema validation must reject it.
      toolRequest: { name: "alpha.text.stats", args: {} },
      sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 4, seed: 7 },
    });
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0].ok).toBe(false);
    expect(result.verification.notes.join(" ")).toMatch(/alpha.text.stats failed/);
  });

  it("refuses a tool the caller did not permit, even if the message names it", async () => {
    const result = await stack.runtime.respond({
      // The message asks for a tool by name, but the allow-list is empty.
      message: "run alpha.admin.clear_vector_store please",
      actorId: "user_1",
      allowedTools: [],
      sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 4, seed: 7 },
    });
    expect(result.toolCalls).toEqual([]);
  });

  it("records a real agent run with a verified outcome", async () => {
    const result = await stack.runtime.respond({
      message: "calculate the text statistics for a short sample",
      actorId: "user_1",
      useAgent: true,
    });

    expect(result.agentRun).not.toBeNull();
    const run = result.agentRun!;
    expect(run.actorId).toBe("user_1");
    expect(run.id).toMatch(/^run_/);
    expect(run.modelVersion).toBe(stack.inference.model.config.version);
    expect(run.startedAt).toBeGreaterThan(0);
    expect(run.finishedAt).toBeGreaterThanOrEqual(run.startedAt);
    expect(Array.isArray(run.toolCalls)).toBe(true);
    expect(Array.isArray(run.plan.steps)).toBe(true);
    expect(["COMPLETED", "PARTIAL", "FAILED", "CANCELLED", "REQUIRES_APPROVAL"]).toContain(
      run.verification.outcome,
    );
    // The outcome is derived, and a run that did nothing is never COMPLETED.
    if (run.toolCalls.length === 0) expect(run.verification.outcome).not.toBe("COMPLETED");
  });

  it("budgets the context window and reports what was trimmed", async () => {
    // Enough retrieved content to force the lower-priority blocks to be
    // trimmed, without making the pinned prompt itself unassemblable.
    stack.rag.ingest({
      title: "Long reference",
      content: "alpha stores checkpoints with weights and optimiser moments. ".repeat(12),
      ownerId: "user_1",
      license: "CC0-1.0",
    });
    const result = await stack.runtime.respond({
      message: "what does alpha store in a checkpoint",
      actorId: "user_1",
      useRetrieval: true,
      sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 4, seed: 7 },
    });
    expect(result.context.usedTokens).toBeLessThanOrEqual(result.context.budgetTokens);
    expect(result.context.budgetTokens).toBeGreaterThan(0);
    // The window is never silently exceeded.
    expect(result.context.usedTokens).toBeLessThanOrEqual(128);
    // Every block reports what it asked for and what it got, with a reason.
    expect(result.context.blocks.length).toBeGreaterThan(0);
    for (const block of result.context.blocks) {
      expect(block.includedTokens).toBeLessThanOrEqual(block.requestedTokens);
      expect(block.reason.length).toBeGreaterThan(0);
    }
  });

  it("refuses a request whose pinned content cannot fit the window", async () => {
    // The prompt is pinned, so it is never trimmed to fit. Alpha reports the
    // failure instead of silently dropping the user's actual question.
    const result = await stack.runtime.respond({
      message: "alpha ".repeat(400),
      actorId: "user_1",
      sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 4, seed: 7 },
    });
    expect(result.response).toBe("");
    expect(result.error).not.toBeNull();
    expect(result.error!.message).toMatch(/does not fit|pinned/i);
  });

  it("returns an error instead of a fabricated answer when the model emits nothing", async () => {
    // Greedy decoding on this undertrained model can legitimately produce only
    // <eos> or whitespace. Alpha must report that, not write a sentence.
    const result = await stack.runtime.respond({
      message: "Alpha is a self owned",
      actorId: "user_1",
      sampling: { ...SAMPLING_PRESETS.greedy, maxNewTokens: 4 },
    });
    // The invariant that matters: an empty answer always carries an error, and
    // a non-empty answer never does.
    if (result.response === "") {
      expect(result.error).not.toBeNull();
      expect(result.error!.code).toBe("alpha.empty_generation");
    } else {
      expect(result.error).toBeNull();
      expect(result.response.trim().length).toBeGreaterThan(0);
    }
  });

  it("refuses an empty message rather than inventing one", async () => {
    await expect(stack.runtime.respond({ message: "   ", actorId: "user_1" })).rejects.toThrow(
      /non-empty message/,
    );
    await expect(
      stack.runtime.respond({ message: "hello", actorId: "" }),
    ).rejects.toThrow(/actorId/);
  });

  it("audits every response it produces", async () => {
    const before = stack.auditEntries();
    await stack.runtime.respond({
      message: "Alpha is a self owned",
      actorId: "user_1",
      sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 4, seed: 7 },
    });
    const entries = stack.audit.list({ limit: 1000 });
    expect(entries.length).toBeGreaterThan(before);
    expect(entries.some((entry) => entry.action === "respond")).toBe(true);
    expect(stack.audit.verifyChain().intact).toBe(true);
  });
});
