/**
 * Test fixtures.
 *
 * These build a *real* stack — a trained tokenizer, a real transformer, a real
 * vector store — rather than mocks, because the point of the tests is to check
 * that the modules actually work together. Everything is sized down so the
 * suite stays fast; no network access and no filesystem access is involved.
 */

import { AlphaTokenizer } from "../tokenizer/bpe";
import { AlphaTransformer } from "../model/transformer";
import { ALPHA_MODEL_PRESETS } from "../model/config";
import { AlphaEmbedder } from "../embeddings/embedder";
import { AlphaInferenceEngine } from "../inference/engine";
import { AlphaVectorStore } from "../vector/store";
import { AlphaMemoryStore } from "../memory/store";
import { AlphaRagPipeline } from "../rag/pipeline";
import { AlphaPolicyEngine } from "../security/policy";
import { AlphaRateLimiter } from "../security/rate-limit";
import { AlphaAuditLog } from "../security/audit";
import { AlphaToolRegistry } from "../tools/registry";
import { registerBuiltinTools } from "../tools/builtin";
import { AlphaObservability } from "../observability/index";
import { AlphaWorkspace } from "../workspace";
import { seedCorpusSlice } from "../datasets/seed-corpus";

const SMALL_MODEL = {
  ...ALPHA_MODEL_PRESETS.nano,
  contextLength: 32,
  dModel: 48,
  nHeads: 4,
  nLayers: 2,
  dFeedForward: 96,
};

export function buildTokenizer(): AlphaTokenizer {
  return AlphaTokenizer.train(seedCorpusSlice(8).documents, {
    vocabSize: 220,
    minPairFrequency: 1,
    version: "test",
    trainedOn: "seed-slice",
  });
}

export function buildModel(tokenizer: AlphaTokenizer): AlphaTransformer {
  return new AlphaTransformer({ ...SMALL_MODEL, vocabSize: tokenizer.vocabSize });
}

/**
 * The smallest complete model + tokenizer pair, for tests that train for real
 * (checkpoints, gradients, lifecycle) without slowing the suite down.
 */
export function gradientCheckFixture(): { tokenizer: AlphaTokenizer; model: AlphaTransformer } {
  const tokenizer = buildTokenizer();
  return { tokenizer, model: buildModel(tokenizer) };
}

export type TestStack = {
  tokenizer: AlphaTokenizer;
  model: AlphaTransformer;
  embedder: AlphaEmbedder;
  inference: AlphaInferenceEngine;
  store: AlphaVectorStore;
  memory: AlphaMemoryStore;
  policy: AlphaPolicyEngine;
  registry: AlphaToolRegistry;
  audit: AlphaAuditLog;
  observability: AlphaObservability;
  workspace: AlphaWorkspace;
};

/** Wire a full stack, including the agent actor scope. */
export async function buildStack(): Promise<TestStack> {
  const tokenizer = buildTokenizer();
  const model = buildModel(tokenizer);
  const store = new AlphaVectorStore();
  const embedder = new AlphaEmbedder({ model, tokenizer, maxTokens: 24 });
  const inference = new AlphaInferenceEngine({ model, tokenizer, stage: "untrained" });
  const memory = new AlphaMemoryStore({ embedder });
  const policy = new AlphaPolicyEngine();
  const audit = new AlphaAuditLog();
  const observability = new AlphaObservability({ logLevel: "warn" });
  const registry = new AlphaToolRegistry({
    policy,
    rateLimiter: new AlphaRateLimiter(),
    audit,
    onExecution: (record) => observability.recordToolExecution(record),
  });
  registerBuiltinTools(registry);
  registry.setServices({ tokenizer, embedder, vectorStore: store, memory, inference });

  policy.assignRole("owner", "owner");
  policy.assignRole("alpha.agent", "agent");
  policy.assignRole("alpha.system", "system");
  policy.registerAgentScope({
    agentId: "alpha.agent",
    permissions: ["model.read", "inference.run", "vector.read", "rag.query", "memory.read", "memory.write", "tool.execute"],
    allowedTools: [
      "alpha.text.stats",
      "alpha.calculator",
      "alpha.tokenizer.analyze",
      "alpha.corpus.search",
      "alpha.memory.search",
      "alpha.memory.write",
    ],
    maxSteps: 4,
  });

  const workspace = new AlphaWorkspace({
    config: {
      preset: "nano",
      model: { ...SMALL_MODEL, vocabSize: 256 },
      tokenizer: { targetVocabSize: 220, version: "test" },
      training: {
        batchSize: 2,
        seqLen: 24,
        totalSteps: 4,
        evalInterval: 0,
        checkpointInterval: 0,
        evalBatches: 2,
      },
      rag: { chunkTokens: 16, overlapTokens: 4, topK: 3, maxContextTokens: 48 },
    },
    logLevel: "warn",
  });
  await workspace.initialise();

  return { tokenizer, model, embedder, inference, store, memory, policy, registry, audit, observability, workspace };
}

/** RAG pipeline plus the store it writes into, for assertions. */
export async function buildRagFixture(): Promise<{ pipeline: AlphaRagPipeline; store: AlphaVectorStore }> {
  const tokenizer = buildTokenizer();
  const model = buildModel(tokenizer);
  const store = new AlphaVectorStore();
  const embedder = new AlphaEmbedder({ model, tokenizer, maxTokens: 24 });
  const inference = new AlphaInferenceEngine({ model, tokenizer, stage: "untrained" });
  const pipeline = new AlphaRagPipeline({
    tokenizer,
    embedder,
    store,
    inference,
    config: { chunkTokens: 16, overlapTokens: 4, topK: 3, maxContextTokens: 64, minScore: -1 },
  });
  return { pipeline, store };
}
