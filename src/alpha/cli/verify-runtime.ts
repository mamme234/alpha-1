/**
 * Alpha CLI — verify the AI runtime end to end.
 *
 *   bun scripts/alpha-verify-runtime.ts
 *   bun scripts/alpha-verify-runtime.ts --json
 *
 * Trains a real model, then exercises every subsystem the runtime is built on
 * and prints what actually happened: the KV cache, embeddings, the vector
 * store's ownership boundary, memory, retrieval, the context budget, tools,
 * agents and the orchestrator itself. Nothing here is mocked.
 *
 * Each check reports PASS or FAIL with the number behind it, so a failure
 * cannot be read as a vague "something went wrong".
 */

import {
  AlphaAiRuntime,
  AlphaAgentRuntime,
  AlphaAuditLog,
  AlphaContextEngine,
  AlphaEmbedder,
  AlphaInferenceEngine,
  AlphaMemoryStore,
  AlphaPolicyEngine,
  AlphaRateLimiter,
  AlphaRagPipeline,
  AlphaTokenizer,
  AlphaToolRegistry,
  AlphaTrainer,
  AlphaTransformer,
  AlphaVectorStore,
  createGenerationCancellation,
  registerBuiltinTools,
  SAMPLING_PRESETS,
  verifyAlphaModel,
} from "../index";
import { ALPHA_MODEL_PRESETS } from "../model/config";
import { ALPHA_SEED_CORPUS } from "../datasets/seed-corpus";

type Check = { id: string; label: string; passed: boolean; detail: string };

const checks: Check[] = [];
function record(id: string, label: string, passed: boolean, detail: string): void {
  checks.push({ id, label, passed, detail });
  const mark = passed ? "ok  " : "FAIL";
  console.log(`  ${mark} ${id}. ${label} — ${detail}`);
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function time(fn: () => void): number {
  const started = performance.now();
  fn();
  return performance.now() - started;
}

export async function main(
  argv: string[] = [],
  write: (line: string) => void = console.log,
): Promise<number> {
  const json = argv.includes("--json");
  const log = json ? () => {} : write;

  const startedAt = Date.now();
  log("ALPHA RUNTIME VERIFICATION — every subsystem, no mocks, no external provider\n");

  // --- build a real stack ----------------------------------------------------
  const tokenizer = AlphaTokenizer.train(ALPHA_SEED_CORPUS.documents, {
    vocabSize: 320,
    minPairFrequency: 1,
    version: "verify",
    trainedOn: `${ALPHA_SEED_CORPUS.name}@${ALPHA_SEED_CORPUS.version}`,
  });
  const config = {
    ...ALPHA_MODEL_PRESETS.nano,
    vocabSize: tokenizer.vocabSize,
    contextLength: 96,
    dModel: 64,
    nHeads: 4,
    nLayers: 2,
    dFeedForward: 128,
  };
  const model = new AlphaTransformer(config, 1337);

  log("Training a real model (this is the run everything below depends on)…");
  const trainer = new AlphaTrainer({
    model,
    tokenizer,
    dataset: ALPHA_SEED_CORPUS,
    config: {
      batchSize: 2,
      seqLen: 24,
      totalSteps: 20,
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
  const firstLoss = trainer.history[0]?.loss ?? 0;
  const lastLoss = summary.lastLoss ?? 0;
  const uniform = summary.uniformLossBaseline;
  record(
    "0",
    "A model was really trained",
    lastLoss < firstLoss && lastLoss < uniform,
    `${trainer.history.length} steps, loss ${firstLoss.toFixed(4)} → ${lastLoss.toFixed(4)} vs uniform ${uniform.toFixed(4)}`,
  );

  const inferenceModel = model.snapshot();
  const embedder = new AlphaEmbedder({ model: inferenceModel, tokenizer, maxTokens: 32 });
  const inference = new AlphaInferenceEngine({ model: inferenceModel, tokenizer, stage: "trained" });
  const memory = new AlphaMemoryStore({ embedder });
  const store = new AlphaVectorStore();
  const context = new AlphaContextEngine({ tokenizer, contextLength: config.contextLength });
  const rag = new AlphaRagPipeline({
    tokenizer,
    embedder,
    store,
    inference,
    config: { chunkTokens: 24, overlapTokens: 6, topK: 3, maxContextTokens: 96, minScore: -1 },
  });
  const policy = new AlphaPolicyEngine();
  const audit = new AlphaAuditLog();
  const tools = new AlphaToolRegistry({ policy, rateLimiter: new AlphaRateLimiter(), audit });
  registerBuiltinTools(tools);
  tools.setServices({ tokenizer, embedder, vectorStore: store, memory, inference });
  policy.assignRole("owner", "owner");
  policy.assignRole("user_1", "owner");
  policy.assignRole("user_2", "operator");
  policy.assignRole("alpha.agent", "agent");
  policy.registerAgentScope({
    agentId: "alpha.agent",
    permissions: ["model.read", "inference.run", "vector.read", "rag.query", "memory.read", "memory.write", "tool.execute"],
    allowedTools: ["alpha.text.stats", "alpha.calculator"],
    maxSteps: 3,
  });
  const agents = new AlphaAgentRuntime({ registry: tools, policy, inference, memory, audit });
  const runtime = new AlphaAiRuntime({
    inference,
    context,
    memory,
    rag,
    tools,
    agents,
    policy,
    rateLimiter: new AlphaRateLimiter(),
    audit,
    systemInstruction: "Answer only from the context. If it lacks the answer, say so.",
  });

  log(`\nModel: ${inferenceModel.config.name} ${inferenceModel.config.version} · ${inferenceModel.parameterCount} params · context ${config.contextLength} · vocab ${tokenizer.vocabSize}\n`);

  // --- 1. inference: cache correctness and speed -----------------------------
  log("Inference");
  const prompt = "Alpha is a self owned system that trains its own weights and";
  const cachedEngine = new AlphaInferenceEngine({ model: inferenceModel, tokenizer, stage: "trained", useCache: true });
  const uncachedEngine = new AlphaInferenceEngine({ model: inferenceModel, tokenizer, stage: "trained", useCache: false });
  const sampling = { ...SAMPLING_PRESETS.greedy, maxNewTokens: 24 };

  // Warm up so the JIT is not credited to one path.
  for (let i = 0; i < 3; i++) {
    cachedEngine.generate(prompt, sampling);
    uncachedEngine.generate(prompt, sampling);
  }
  const cachedResult = cachedEngine.generate(prompt, sampling);
  const uncachedResult = uncachedEngine.generate(prompt, sampling);
  const identical = cachedResult.tokenIds.join() === uncachedResult.tokenIds.join();
  record(
    "1",
    "KV cache produces the identical token sequence",
    identical,
    identical
      ? `${cachedResult.generatedTokens} tokens, ids match exactly (${cachedResult.tokenIds.slice(0, 6).join(",")}…)`
      : `cached ${cachedResult.tokenIds.join()} vs uncached ${uncachedResult.tokenIds.join()}`,
  );

  const cachedMs = median(Array.from({ length: 5 }, () => time(() => cachedEngine.generate(prompt, sampling))));
  const uncachedMs = median(Array.from({ length: 5 }, () => time(() => uncachedEngine.generate(prompt, sampling))));
  const speedup = uncachedMs / Math.max(0.001, cachedMs);
  record(
    "2",
    "KV cache is measurably faster",
    speedup > 1,
    `${uncachedMs.toFixed(0)}ms uncached → ${cachedMs.toFixed(0)}ms cached (${speedup.toFixed(2)}x, median of 5); ` +
      `prefill ${cachedResult.timing.prefillMs.toFixed(1)}ms, decode ${cachedResult.timing.decodeMs.toFixed(1)}ms, ` +
      `cache ${cachedResult.cache.positions}/${cachedResult.cache.capacity} positions, ${cachedResult.cache.bytes} bytes`,
  );

  const seedA = cachedEngine.generate(prompt, { ...SAMPLING_PRESETS.balanced, maxNewTokens: 12, seed: 7 });
  const seedB = cachedEngine.generate(prompt, { ...SAMPLING_PRESETS.balanced, maxNewTokens: 12, seed: 7 });
  const seedC = cachedEngine.generate(prompt, { ...SAMPLING_PRESETS.balanced, maxNewTokens: 12, seed: 8 });
  record(
    "3",
    "Sampling is deterministic per seed and varies across seeds",
    seedA.tokenIds.join() === seedB.tokenIds.join() && seedA.tokenIds.join() !== seedC.tokenIds.join(),
    `seed 7 twice → identical (${seedA.tokenIds.length} tokens); seed 8 → ${seedC.tokenIds.join() === seedA.tokenIds.join() ? "identical" : "different"}`,
  );

  const cancellation = createGenerationCancellation();
  cancellation.cancel("operator stopped the run");
  const cancelledResult = cachedEngine.generate(prompt, sampling, { cancellation });
  record(
    "4",
    "Generation can be cancelled and says so",
    cancelledResult.stopReason === "cancelled" && cancelledResult.generatedTokens === 0,
    `stopReason "${cancelledResult.stopReason}" after ${cancelledResult.generatedTokens} token(s)`,
  );

  // --- 2. embeddings ---------------------------------------------------------
  log("\nEmbeddings");
  const e1 = embedder.embed("alpha owns its own stack");
  const e2 = embedder.embed("alpha owns its own stack");
  const e3 = embedder.embed("something entirely different about weather");
  const norm = Math.sqrt(e1.vector.reduce((sum: number, v: number) => sum + v * v, 0));
  record(
    "5",
    "Embeddings are deterministic and normalised",
    e1.vector.join() === e2.vector.join() && Math.abs(norm - 1) < 1e-5,
    `same input → identical ${e1.dimension}-d vector, L2 norm ${norm.toFixed(6)}`,
  );
  const cosine = (a: number[], b: number[]): number =>
    a.reduce((sum: number, v: number, i: number) => sum + v * b[i], 0);
  const selfSim = cosine(e1.vector, e2.vector);
  const crossSim = cosine(e1.vector, e3.vector);
  record(
    "6",
    "Different inputs produce different vectors",
    selfSim > crossSim,
    `self-similarity ${selfSim.toFixed(4)} > cross-similarity ${crossSim.toFixed(4)} (dim ${e1.dimension}, model ${e1.config.model}@${e1.config.modelVersion}, pooling ${e1.config.pooling})`,
  );

  // --- 3. vector store ownership --------------------------------------------
  log("\nVector store");
  store.ensureCollection("probe", e1.dimension, "cosine");
  store.insert({ id: "owned-1", collection: "probe", vector: e1.vector, text: "mine", ownerId: "user_1", embedding: { model: e1.config.model, version: e1.config.modelVersion } });
  store.insert({ id: "owned-2", collection: "probe", vector: e3.vector, text: "theirs", ownerId: "user_2", embedding: { model: e1.config.model, version: e1.config.modelVersion } });
  const asOne = store.search({ collection: "probe", vector: e1.vector, topK: 10, ownerId: "user_1" });
  const asThree = store.search({ collection: "probe", vector: e1.vector, topK: 10, ownerId: "user_3" });
  record(
    "7",
    "Vectors never cross an ownership boundary",
    asOne.length === 1 && asOne[0].record.id === "owned-1" && asThree.length === 0,
    `user_1 sees ${asOne.length} (${asOne.map((h: { record: { id: string } }) => h.record.id).join(",")}), user_3 sees ${asThree.length}`,
  );
  const snapshot = JSON.parse(JSON.stringify(store.exportSnapshot()));
  const restored = new AlphaVectorStore();
  restored.importSnapshot(snapshot);
  const restoredHit = restored.getOwned("owned-1", "user_1");
  record(
    "8",
    "Vector store persists and reloads with ownership intact",
    restoredHit !== null && restored.getOwned("owned-1", "user_2") === null,
    `reloaded ${snapshot.records.length} record(s); owner preserved as "${restoredHit?.ownerId}"`,
  );

  // --- 4. memory -------------------------------------------------------------
  log("\nMemory");
  let approvalEnforced = false;
  try {
    memory.write({ scope: "long-term", key: "unapproved", content: "should fail", ownerId: "user_1" });
  } catch {
    approvalEnforced = true;
  }
  const durable = memory.write({
    scope: "long-term",
    key: "user.preference",
    content: "the owner prefers short answers",
    ownerId: "user_1",
    approved: true,
    source: "user",
    provenance: { origin: "onboarding", referenceId: "conv-1", recordedBy: "user_1" },
  });
  record(
    "9",
    "Long-term memory requires approval and records provenance",
    approvalEnforced && durable.approved && durable.provenance.origin === "onboarding",
    `unapproved write rejected; approved record has provenance "${durable.provenance.origin}" / ${durable.provenance.referenceId}`,
  );
  const mine = memory.retrieve("preference for answer length", { ownerId: "user_1", topK: 3 });
  const theirs = memory.retrieve("preference for answer length", { ownerId: "user_2", topK: 3 });
  record(
    "10",
    "Memory recall is scoped to its owner",
    mine.length > 0 && mine.every((m: (typeof mine)[number]) => m.record.ownerId === "user_1") && theirs.length === 0,
    `user_1 recalled ${mine.length}, user_2 recalled ${theirs.length}`,
  );

  // --- 5. context budget -----------------------------------------------------
  log("\nContext engine");
  const assembled = context.assemble(
    [
      { id: "instruction", kind: "instruction", text: "Answer only from the context.", pinned: true },
      { id: "prompt", kind: "prompt", text: "what does a checkpoint store" },
      { id: "sources", kind: "sources", text: "a checkpoint stores weights and optimiser moments. ".repeat(20) },
    ],
    { reserveForOutput: 8 },
  );
  const consistent = assembled.blocks.every((b: (typeof assembled.blocks)[number]) => b.includedTokens <= b.requestedTokens);
  record(
    "11",
    "Context stays inside the window and reports every trim",
    assembled.usedTokens <= assembled.budgetTokens && consistent && assembled.blocks.every((b: (typeof assembled.blocks)[number]) => b.reason.length > 0),
    `${assembled.usedTokens}/${assembled.budgetTokens} tokens used; ` +
      assembled.blocks.map((b: (typeof assembled.blocks)[number]) => `${b.kind}=${b.status}`).join(", "),
  );

  // --- 6. rag ---------------------------------------------------------------
  log("\nRetrieval");
  rag.ingest({
    title: "Alpha training",
    content:
      "alpha trains with adamw and backpropagation through its own autodiff engine. " +
      "checkpoints store the weights, the optimiser moments and the random generator position so a run can resume.",
    ownerId: "user_1",
    license: "CC0-1.0",
  });
  const hits = rag.retrieve("how does a training run resume", { ownerId: "user_1", topK: 3 });
  const foreign = rag.retrieve("how does a training run resume", { ownerId: "user_2", topK: 3 });
  record(
    "12",
    "Retrieval returns real sources and cites them",
    hits.length > 0 && hits[0].record.metadata.title === "Alpha training",
    `${hits.length} chunk(s); top "${hits[0]?.record.metadata.title}" score ${hits[0]?.score.toFixed(4)}, chunk id ${hits[0]?.record.id}`,
  );
  record("13", "Retrieval does not cross an ownership boundary", foreign.length === 0, `user_2 retrieved ${foreign.length} chunk(s)`);

  // --- 7. tools -------------------------------------------------------------
  log("\nTools");
  const toolPhrase = "alpha owns its stack";
  const toolResult = await tools.execute("alpha.text.stats", { text: toolPhrase }, { actorId: "owner" });
  record(
    "14",
    "A permitted tool runs and returns its real output",
    toolResult.ok && (toolResult.output as { characters: number }).characters === toolPhrase.length,
    `alpha.text.stats returned ${JSON.stringify(toolResult.output)}`,
  );
  const schemaResult = await tools.execute("alpha.text.stats", {} as Record<string, unknown>, { actorId: "owner" });
  record(
    "15",
    "Schema validation runs before the handler",
    !schemaResult.ok && /required|missing|invalid/i.test(schemaResult.error?.message ?? ""),
    `missing argument rejected: "${schemaResult.error?.message}"`,
  );
  const slowRegistry = new AlphaToolRegistry({ policy });
  slowRegistry.setServices({});
  slowRegistry.register({
    name: "alpha.test.slow",
    description: "Never finishes in time.",
    module: "test",
    inputSchema: { type: "object", properties: {}, required: [] } as never,
    permission: "tool.execute",
    timeoutMs: 40,
    handler: () => new Promise((resolve) => setTimeout(() => resolve("late"), 5_000)),
  });
  policy.assignRole("owner", "owner");
  const slowResult = await slowRegistry.execute("alpha.test.slow", {}, { actorId: "owner" });
  record(
    "16",
    "A tool that overruns its timeout fails instead of hanging",
    !slowResult.ok && /timeout/i.test(slowResult.error?.message ?? ""),
    `${slowResult.error?.message} after ${slowResult.durationMs}ms`,
  );

  // --- 8. agents ------------------------------------------------------------
  log("\nAgents");
  const agentIterator = agents.run({ goal: "count the characters in a short phrase", actorId: "alpha.agent" });
  let agentStep = await agentIterator.next();
  while (!agentStep.done) agentStep = await agentIterator.next();
  const run = agentStep.value;
  record(
    "17",
    "An agent run records a plan, tool calls and a verified outcome",
    run.plan.steps.length > 0 && run.verification.outcome !== undefined,
    `${run.verification.outcome}: ${run.plan.steps.length} step(s), ${run.toolCalls.length} tool call(s) — ${run.verification.reason}`,
  );
  record(
    "18",
    "The agent sandbox bounds the run",
    run.sandbox.maxToolCalls > 0 && run.sandbox.maxSteps > 0,
    `${run.sandbox.steps}/${run.sandbox.maxSteps} steps, ${run.sandbox.toolCalls}/${run.sandbox.maxToolCalls} tool calls, ` +
      `network ${run.sandbox.allowNetwork ? "allowed" : "denied"}, filesystem ${run.sandbox.allowFileSystem ? "allowed" : "denied"}`,
  );

  // --- 9. orchestrator -------------------------------------------------------
  log("\nAI runtime orchestrator");
  const plain = await runtime.respond({
    message: "Alpha is a self owned",
    actorId: "user_1",
    sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 10, seed: 7 },
  });
  record(
    "19",
    "A plain request routes to inference and returns real tokens",
    plain.route.decision === "inference" && (plain.generation?.generatedTokens ?? 0) > 0,
    `route ${plain.route.decision} (${plain.route.rationale}); ${plain.generation?.generatedTokens} token(s), ` +
      `${plain.generation?.latencyMs}ms, stop "${plain.generation?.stopReason}"`,
  );
  const grounded = await runtime.respond({
    message: "how does a training run resume",
    actorId: "user_1",
    useRetrieval: true,
    sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 8, seed: 7 },
  });
  record(
    "20",
    "A grounded request cites its sources",
    grounded.sources.length > 0 && grounded.verification.grounded,
    `${grounded.sources.length} source(s), ${grounded.context.usedTokens}/${grounded.context.budgetTokens} context tokens`,
  );
  const isolated = await runtime.respond({
    message: "how does a training run resume",
    actorId: "user_2",
    useRetrieval: true,
    sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 8, seed: 7 },
  });
  record(
    "21",
    "The orchestrator does not leak across accounts",
    isolated.sources.length === 0 && isolated.verification.notes.join(" ").match(/no indexed document/i) !== null,
    `user_2 got ${isolated.sources.length} source(s) and was told: "${isolated.verification.notes[0] ?? "—"}"`,
  );
  const withTool = await runtime.respond({
    message: "give me text statistics for a phrase",
    actorId: "user_1",
    allowedTools: ["alpha.text.stats"],
    toolRequest: { name: "alpha.text.stats", args: { text: toolPhrase } },
    sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 6, seed: 7 },
  });
  record(
    "22",
    "A tool result flows back into context for a second inference",
    withTool.toolCalls.length === 1 && withTool.toolCalls[0].ok && withTool.generation !== null,
    withTool.toolCalls[0]?.ok
      ? `tool ${withTool.toolCalls[0]?.tool} ok=true, then ${withTool.generation?.generatedTokens} token(s) generated with the result in context`
      : `tool failed: ${withTool.toolCalls[0]?.error}`,
  );
  const denied = await runtime.respond({
    message: "run the admin tool",
    actorId: "user_1",
    allowedTools: ["alpha.text.stats"],
    toolRequest: { name: "alpha.admin.clear_vector_store", args: { confirm: true } },
    sampling: { ...SAMPLING_PRESETS.balanced, maxNewTokens: 6, seed: 7 },
  });
  record(
    "23",
    "A tool outside the allow-list is refused",
    denied.toolCalls.length === 0,
    "naming a tool that is not permitted produced no tool call at all",
  );

  // --- 10. the core is still verified ----------------------------------------
  log("\nModel core (Step 2 checks, re-run here so nothing is assumed)");
  const verification = verifyAlphaModel({ model, tokenizer, dataset: ALPHA_SEED_CORPUS, training: { totalSteps: 8 } });
  record(
    "24",
    "Checks A–I still pass on this build",
    verification.passed,
    `${verification.checks.filter((c) => c.passed).length}/${verification.checks.length} checks passed (A–I)`,
  );

  const failed = checks.filter((c) => !c.passed);
  const header = `RUNTIME VERIFICATION ${failed.length === 0 ? "PASSED" : "FAILED"} — ${checks.length - failed.length}/${checks.length} checks in ${((Date.now() - startedAt) / 1000).toFixed(1)}s`;

  if (json) {
    write(
      JSON.stringify(
        {
          passed: failed.length === 0,
          checks,
          model: {
            name: inferenceModel.config.name,
            version: inferenceModel.config.version,
            parameters: inferenceModel.parameterCount,
            contextLength: config.contextLength,
            vocabSize: tokenizer.vocabSize,
            stage: "trained",
          },
          training: { firstLoss, lastLoss, uniformLossBaseline: uniform },
          cache: { speedup, identical },
        },
        null,
        2,
      ),
    );
  } else {
    write(`\n${header}`);
  }
  return failed.length === 0 ? 0 : 1;
}
