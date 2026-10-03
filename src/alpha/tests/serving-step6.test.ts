/**
 * Step 6 — serving and chat.
 *
 * These tests run the exact modules the Convex chat API bundles: the checked-in
 * Step 5 artefact, the serving runtime and the chat-turn layer. They verify the
 * real generation path and the failures that must fail — a tampered artefact, a
 * request that cannot fit the window, a stop request, a stop channel that
 * breaks, and embeddings from a different model.
 */

import { AlphaValidationError } from "../core/errors";
import {
  loadServingArtifact,
  parseServingArtifact,
  type AlphaServingArtifact,
} from "../serving/artifact";
import {
  ALPHA_CHAT_TOOL_ALLOWLIST,
  createServingRuntime,
  type AlphaServingRuntime,
} from "../serving/runtime";
import { buildChatRequest, runChatTurn, runChatTurnStream } from "../serving/chat";
import STEP5_ARTIFACT from "../serving/step5-artifact.json";
import { beforeAll, describe, expect, it } from "vitest";

const artifact = STEP5_ARTIFACT as unknown as AlphaServingArtifact;

/** Deterministic, short, and without memory/retrieval side effects. */
const stableTurn = {
  actorId: "test.actor",
  conversationId: "test.conversation",
  message: "Hello",
  settings: { deterministic: true, maxNewTokens: 16, useMemory: false, useRetrieval: false },
} as const;

let runtime: AlphaServingRuntime;

beforeAll(() => {
  runtime = createServingRuntime({ artifact });
});

describe("step 6 · serving the verified Step 5 artefact", () => {
  it("loads the artefact and answers to its published fingerprints", () => {
    const parsed = parseServingArtifact(artifact);
    expect(parsed.model.configFingerprint).toBe("cfg_49e2ffd4");
    expect(parsed.tokenizer.fingerprint).toBe("tok_e8edb134");
    expect(parsed.model.parameterCount).toBe(418_656);
    expect(parsed.stage).toBe("trained");
    expect(parsed.externalModels).toBe("none");

    const loaded = loadServingArtifact(artifact);
    expect(loaded.tokenizer.fingerprint()).toBe("tok_e8edb134");
    expect(loaded.model.config.contextLength).toBe(96);
    expect(loaded.model.config.vocabSize).toBe(768);
  });

  it("refuses a tampered artefact, a zeroed model and a claimed external model", () => {
    const tampered = structuredClone(artifact);
    tampered.tokenizer.fingerprint = "tok_deadbeef";
    expect(() => loadServingArtifact(tampered)).toThrow(AlphaValidationError);

    const zeroed = structuredClone(artifact);
    const firstTensor = Object.keys(zeroed.weights.tensors)[0];
    zeroed.weights.tensors[firstTensor] = zeroed.weights.tensors[firstTensor].replace(/[^=]/g, "A");
    expect(() => loadServingArtifact(zeroed)).toThrow(/all zero/i);

    const external = structuredClone(artifact);
    (external as { externalModels: string }).externalModels = "openai";
    expect(() => parseServingArtifact(external)).toThrow(/external model/i);

    const biggerVocab = structuredClone(artifact);
    biggerVocab.tokenizer.vocabSize = 769;
    expect(() => parseServingArtifact(biggerVocab)).toThrow(/vocabulary/i);
  });

  it("reports a window that can hold its instruction and a real request", () => {
    expect(runtime.limits.contextLength).toBe(96);
    expect(runtime.limits.instructionTokens).toBeGreaterThan(0);
    expect(runtime.limits.instructionTokens).toBe(runtime.context.countTokens(runtime.ai.systemInstruction));
    expect(runtime.limits.maxRequestTokens).toBeLessThan(runtime.limits.contextLength);
    expect(runtime.info.externalModels).toBe("none");

    const request = buildChatRequest(runtime, { ...stableTurn, message: "What is 2+2?" });
    expect(runtime.context.countTokens(request.message)).toBeLessThanOrEqual(runtime.limits.maxRequestTokens);
  });

  it("serves a real chat turn from its own trained weights", async () => {
    const result = await runChatTurn(runtime, stableTurn);
    expect(result.error).toBeNull();
    expect(result.response.length).toBeGreaterThan(0);
    expect(result.generation).not.toBeNull();
    expect(result.generation?.modelStage).toBe("trained");
    expect(result.generation?.generatedTokens).toBeGreaterThan(0);
    expect(["eos", "max-tokens", "context-limit", "stop-token", "stop-sequence", "cancelled"]).toContain(
      result.generation?.stopReason,
    );
    // The context report is honest: measured usage never exceeds the budget.
    expect(result.context.usedTokens).toBeLessThanOrEqual(result.context.budgetTokens);
  });

  it("streams deltas that rebuild the final answer exactly", async () => {
    const deltas: string[] = [];
    const iterator = runChatTurnStream(runtime, { ...stableTurn, shouldStopEvery: 2 });
    let step = await iterator.next();
    while (!step.done) {
      if (step.value.type === "delta") deltas.push(step.value.text);
      step = await iterator.next();
    }
    expect(deltas.length).toBeGreaterThan(1);
    expect(deltas.join("")).toBe(step.value.response);
    expect(step.value.error).toBeNull();
  });

  it("stops generation on request and keeps the partial answer", async () => {
    let checks = 0;
    const iterator = runChatTurnStream(runtime, {
      ...stableTurn,
      settings: { ...stableTurn.settings, maxNewTokens: 60 },
      shouldStopEvery: 1,
      shouldStop: async () => {
        checks += 1;
        return checks >= 8;
      },
    });
    let step = await iterator.next();
    while (!step.done) step = await iterator.next();
    expect(step.value.generation?.stopReason).toBe("cancelled");
    expect(step.value.error).toBeNull();
    expect(step.value.response.length).toBeGreaterThan(0);
    expect(step.value.generation?.generatedTokens).toBeLessThan(60);
  });

  it("does not treat a broken stop channel as a stop request", async () => {
    const iterator = runChatTurnStream(runtime, {
      ...stableTurn,
      settings: { ...stableTurn.settings, maxNewTokens: 12 },
      shouldStopEvery: 2,
      shouldStop: async () => {
        throw new Error("simulated stop-channel failure");
      },
    });
    let step = await iterator.next();
    while (!step.done) step = await iterator.next();
    expect(step.value.error).toBeNull();
    expect(step.value.generation?.stopReason).not.toBe("cancelled");
  });

  it("refuses an oversized request with the measured token count", () => {
    let message = "";
    try {
      buildChatRequest(runtime, { ...stableTurn, message: "x".repeat(400) });
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    }
    expect(message).toContain("this message is 400 tokens");
    expect(message).toContain(String(runtime.limits.maxRequestTokens));
  });

  it("clips a caller-supplied tool list to the chat allow-list", () => {
    const widened = buildChatRequest(runtime, {
      actorId: "test.actor",
      conversationId: "test.conversation",
      message: "Hello",
      settings: { allowedTools: ["alpha.admin.clear_vector_store", "alpha.calculator"] },
    });
    expect(widened.allowedTools).toEqual(["alpha.calculator"]);
    expect(widened.allowedTools?.every((name) => ALPHA_CHAT_TOOL_ALLOWLIST.includes(name))).toBe(true);
  });

  it("executes a permitted tool and cites a stored document", async () => {
    const toolTurn = await runChatTurn(runtime, {
      ...stableTurn,
      message: 'calculate "12 * 4"',
    });
    const calculator = toolTurn.toolCalls.find((call) => call.tool === "alpha.calculator");
    expect(calculator?.ok).toBe(true);
    expect((calculator?.output as { value?: number } | undefined)?.value).toBe(48);

    const document = "Alpha is a self-owned AI system.";
    runtime.hydrateFor("test.rag", {
      vectors: [
        {
          recordId: "vec_test_doc",
          collection: "alpha_documents",
          text: document,
          embedding: runtime.embedder.embed(document).vector,
          metadata: {},
        },
      ],
    });
    const ragTurn = await runChatTurn(runtime, {
      actorId: "test.rag",
      conversationId: "test.rag",
      message: document,
      settings: { deterministic: true, maxNewTokens: 12, useMemory: false, useRetrieval: true },
    });
    expect(ragTurn.sources.length).toBeGreaterThanOrEqual(1);
    expect(ragTurn.context.blocks.some((block) => block.kind === "sources")).toBe(true);
  });

  it("skips embeddings from another model with a reason and keeps the matching ones", () => {
    const dimension = runtime.embedder.dimension;
    const vector = new Array<number>(dimension).fill(0);
    vector[0] = 1;
    const report = runtime.hydrateFor("test.hydration", {
      memories: [
        {
          memoryId: "mem_ok",
          scope: "long-term",
          key: "fact",
          content: "Alpha serves its own weights.",
          embedding: vector,
          approved: true,
          source: "system",
        },
        {
          memoryId: "mem_foreign",
          scope: "long-term",
          key: "foreign",
          content: "Embedded by another model.",
          embedding: [1, 2, 3],
          approved: true,
          source: "system",
        },
      ],
      vectors: [
        {
          recordId: "vec_ok",
          collection: "alpha_documents",
          text: "Alpha owns its stack.",
          embedding: vector,
          metadata: {},
        },
        {
          recordId: "vec_foreign",
          collection: "alpha_documents",
          text: "From a different embedding space.",
          embedding: [0.1, 0.2],
          metadata: {},
        },
      ],
    });
    expect(report.memories.imported).toBe(1);
    expect(report.memories.skipped).toBe(1);
    expect(report.vectors.imported).toBe(1);
    expect(report.vectors.skipped).toBe(1);
    expect(report.vectors.reasons[0]).toContain("does not match this model");
  });

  it("recovers after refused requests on the same runtime", async () => {
    expect(() => buildChatRequest(runtime, { ...stableTurn, message: "" })).toThrow(AlphaValidationError);
    expect(() => buildChatRequest(runtime, { ...stableTurn, message: "x".repeat(400) })).toThrow(AlphaValidationError);
    const result = await runChatTurn(runtime, {
      ...stableTurn,
      settings: { ...stableTurn.settings, maxNewTokens: 12 },
    });
    expect(result.error).toBeNull();
    expect(result.response.length).toBeGreaterThan(0);
  });
});
