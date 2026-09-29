import { describe, expect, it } from "vitest";
import { AlphaRagPipeline, buildRagPrompt } from "../rag/pipeline";
import { AlphaAgentRuntime } from "../agents/runtime";
import { extractCapabilityPhrases, parsePlannerJson, planFromCapabilities, planWithModel } from "../agents/planner";
import { buildToolArguments } from "../agents/runtime";
import { AlphaAutomationEngine, evaluateCondition } from "../automation/engine";
import { buildRagFixture, buildStack } from "./helpers";
import type { AgentRunResult } from "../agents/types";

describe("alpha rag pipeline", () => {
  it("parses, chunks with overlap and keeps source offsets", async () => {
    const { pipeline } = await buildRagFixture();
    const document = pipeline.parse({
      title: "Alpha architecture",
      content: "alpha is a self owned stack. ".repeat(30),
      license: "CC0-1.0",
    });
    const chunks = pipeline.chunk(document);
    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks[0].index).toBe(0);
    expect(chunks[0].documentId).toBe(document.id);
    for (const chunk of chunks) {
      expect(chunk.tokens).toBeGreaterThan(0);
      // The decoded window must be recoverable from the source text.
      expect(document.text.slice(chunk.start, chunk.end)).toBe(chunk.text);
    }
    // Overlap: the second chunk starts before the first one ends.
    expect(chunks[1].start).toBeLessThan(chunks[0].end);
  });

  it("rejects unsupported document kinds instead of guessing", async () => {
    const { pipeline } = await buildRagFixture();
    expect(() => pipeline.parse({ title: "x", content: "y", kind: "pdf" })).toThrow(/not implemented/);
    expect(() => pipeline.parse({ title: "", content: "y" })).toThrow(/needs a title/);
    expect(() => pipeline.parse({ title: "x", content: "   " })).toThrow(/needs content/);
  });

  it("ingests documents and retrieves chunks by similarity", async () => {
    const { pipeline } = await buildRagFixture();
    const ingested = pipeline.ingest({
      title: "Training notes",
      content:
        "alpha trains with adamw and backpropagation. the optimiser keeps moments so a run can resume. checkpoints store weights and the random generator position.",
      license: "CC0-1.0",
    });
    expect(ingested.document.chunks).toBeGreaterThan(0);
    expect(pipeline.listDocuments()).toHaveLength(1);
    const hits = pipeline.retrieve("how does alpha train", { topK: 3 });
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].rank).toBe(1);
    expect(hits[0].record.metadata.title).toBe("Training notes");
  });

  it("assembles context with citations and answers through Alpha", async () => {
    const { pipeline } = await buildRagFixture();
    pipeline.ingest({
      title: "Memory policy",
      content: "long term memory requires explicit approval before it can be recalled.",
      license: "CC0-1.0",
    });
    const answer = pipeline.answer("what does long term memory require", { maxNewTokens: 8 });
    expect(answer.sources.length).toBeGreaterThan(0);
    expect(answer.sources[0].title).toBe("Memory policy");
    expect(answer.sources[0].excerpt.length).toBeGreaterThan(0);
    expect(answer.contextTokens).toBeGreaterThan(0);
    expect(answer.modelStage).toBe("untrained");
    expect(answer.generation.warning).toMatch(/random initialisation/);
    expect(answer.answeredWithoutContext).toBe(false);
  });

  it("says so when there is nothing to retrieve", async () => {
    const { pipeline } = await buildRagFixture();
    const answer = pipeline.answer("anything at all", { maxNewTokens: 4 });
    expect(answer.retrieved).toBe(0);
    expect(answer.answeredWithoutContext).toBe(true);
    expect(buildRagPrompt("q", "")).toContain("using only what you know");
  });

  it("removes a document and its vectors together", async () => {
    const { pipeline, store } = await buildRagFixture();
    const { document } = pipeline.ingest({ title: "Temp", content: "temporary content about vectors", license: "CC0-1.0" });
    expect(store.count()).toBeGreaterThan(0);
    const removed = pipeline.removeDocument(document.id);
    expect(removed).toBeGreaterThan(0);
    expect(store.count()).toBe(0);
    expect(pipeline.listDocuments()).toHaveLength(0);
  });
});

describe("alpha agents", () => {
  it("plans from capabilities without calling a model", async () => {
    const { registry, policy } = await buildStack();
    const plan = planFromCapabilities(
      { id: "t1", goal: "calculate 12 * 4 and count the words in this sentence", createdAt: Date.now() },
      registry,
      "alpha.agent",
      { maxSteps: 3 },
    );
    expect(plan.plannerSource).toBe("capability-match");
    expect(plan.steps.length).toBeGreaterThan(0);
    expect(plan.rationale).toMatch(/no model call/);
    expect(extractCapabilityPhrases("please calculate the total")).toContain("calculate");
    expect(policy.roleOf("alpha.agent")).toBe("agent");
  });

  it("extracts arithmetic arguments instead of passing the raw goal", async () => {
    const { registry } = await buildStack();
    const calculator = registry.describe().find((tool) => tool.name === "alpha.calculator")!;
    const args = buildToolArguments(calculator.inputSchema, {
      id: "t3",
      goal: "calculate 12 * 4 and count the words",
      createdAt: Date.now(),
    });
    expect(args.expression).toBe("12 * 4");
  });

  it("parses a model plan and rejects malformed output", () => {
    expect(parsePlannerJson('here: {"steps":[{"tool":"a","description":"d"}],"rationale":"r"}')).toEqual({
      steps: [{ tool: "a", description: "d" }],
      rationale: "r",
    });
    expect(parsePlannerJson("no json here")).toBeNull();
    expect(parsePlannerJson('{"steps":[]}')).toBeNull();
  });

  it("refuses to plan with an untrained model", async () => {
    const { registry, inference } = await buildStack();
    const plan = planWithModel(
      { id: "t2", goal: "calculate 2 + 2", createdAt: Date.now() },
      { inference, registry, actorId: "alpha.agent" },
    );
    expect(plan).toBeNull();
    expect(inference.stage).toBe("untrained");
  });

  it("runs a real multi-step task with tool calls and a structured synthesis", async () => {
    const { registry, policy } = await buildStack();
    const runtime = new AlphaAgentRuntime({ registry, policy });
    const iterator = runtime.run({
      goal: "calculate 12 * 4 and count the characters in the alpha corpus",
      actorId: "alpha.agent",
      maxSteps: 3,
    });
    const events: string[] = [];
    let next = await iterator.next();
    let result: AgentRunResult | null = null;
    while (!next.done) {
      events.push(next.value.type);
      if (next.value.type === "done") result = next.value.result;
      next = await iterator.next();
    }
    expect(events).toContain("plan");
    expect(events).toContain("tool");
    expect(events).toContain("done");
    expect(result!.status).toBe("completed");
    expect(result!.toolCalls.length).toBeGreaterThan(0);
    expect(result!.toolCalls.every((call) => call.ok)).toBe(true);
    expect(result!.synthesis.method).toBe("structured");
    expect(result!.synthesis.warning).toMatch(/no model was invoked/);
    expect(result!.history.some((entry) => entry.phase === "plan")).toBe(true);
    expect(result!.sandbox.violations).toHaveLength(0);
    expect(runtime.stats().runs).toBe(1);
  }, 60_000);

  it("blocks a task that needs a tool the agent scope does not allow", async () => {
    const { registry, policy } = await buildStack();
    const runtime = new AlphaAgentRuntime({ registry, policy });
    const result = await (async (): Promise<AgentRunResult> => {
      const iterator = runtime.run({
        goal: "clear the vector store and delete every embedded document",
        actorId: "alpha.agent",
        maxSteps: 2,
      });
      let next = await iterator.next();
      let output: AgentRunResult | null = null;
      while (!next.done) {
        if (next.value.type === "done") output = next.value.result;
        next = await iterator.next();
      }
      if (!output) throw new Error("agent run produced no result");
      return output;
    })();
    const allowed = [
      "alpha.text.stats",
      "alpha.calculator",
      "alpha.tokenizer.analyze",
      "alpha.corpus.search",
      "alpha.memory.search",
      "alpha.memory.write",
    ];
    // The plan can only bind steps to tools this agent is allowed to call.
    expect(
      result.plan.steps.every((step) => step.toolName === null || allowed.includes(step.toolName)),
    ).toBe(true);
    expect(result.toolCalls.every((call) => call.tool !== "alpha.admin.clear_vector_store")).toBe(true);

    // And a direct attempt at the destructive tool is refused by the policy.
    const denied = await registry.execute(
      "alpha.admin.clear_vector_store",
      { confirm: true },
      { actorId: "alpha.agent" },
    );
    expect(denied.ok).toBe(false);
    expect(denied.error?.code).toBe("alpha.permission_denied");
  }, 60_000);

  it("records an agent run in the audit log", async () => {
    const { registry, policy, audit } = await buildStack();
    const runtime = new AlphaAgentRuntime({ registry, policy, audit });
    const iterator = runtime.run({ goal: "calculate 3 + 4", actorId: "alpha.agent", maxSteps: 2 });
    let next = await iterator.next();
    while (!next.done) next = await iterator.next();
    const entries = audit.list({ module: "agents" });
    expect(entries.length).toBeGreaterThan(0);
    expect(entries[0].action).toBe("agent.run");
  }, 60_000);
});

describe("alpha automation", () => {
  it("evaluates conditions", () => {
    expect(evaluateCondition({ path: "a.b", operator: "equals", value: 2 }, { a: { b: 2 } })).toBe(true);
    expect(evaluateCondition({ path: "a", operator: "exists" }, { a: 1 })).toBe(true);
    expect(evaluateCondition({ path: "missing", operator: "exists" }, {})).toBe(false);
    expect(evaluateCondition({ path: "n", operator: "greater-than", value: 3 }, { n: 5 })).toBe(true);
    expect(evaluateCondition({ path: "s", operator: "contains", value: "al" }, { s: "alpha" })).toBe(true);
    expect(evaluateCondition({ path: "s", operator: "matches", value: "^a.*a$" }, { s: "alpha" })).toBe(true);
  });

  it("runs a workflow with a tool action and records history", async () => {
    const { registry, policy, audit, memory } = await buildStack();
    const engine = new AlphaAutomationEngine({ registry, policy, audit, memory, maxAttempts: 2 });
    const workflow = engine.registerWorkflow({
      name: "nightly stats",
      description: "compute text statistics for the corpus",
      trigger: { kind: "manual" },
      conditions: [{ path: "enabled", operator: "equals", value: true }],
      actions: [
        { kind: "tool", toolName: "alpha.text.stats", args: { text: "alpha owns its own stack" } },
        { kind: "log", message: "nightly stats complete" },
      ],
      enabled: true,
      actorId: "owner",
    });
    const job = await engine.run(workflow.id, { enabled: true });
    expect(job.status).toBe("succeeded");
    expect(job.outcomes).toHaveLength(2);
    expect(job.attempts).toBe(1);
    expect(job.durationMs).toBeGreaterThanOrEqual(0);
    expect(engine.executionHistory()).toHaveLength(1);
    expect(engine.stats().succeeded).toBe(1);
  }, 60_000);

  it("skips a job whose conditions are not met and says why", async () => {
    const { registry, policy } = await buildStack();
    const engine = new AlphaAutomationEngine({ registry, policy });
    const workflow = engine.registerWorkflow({
      name: "conditional",
      description: "",
      trigger: { kind: "manual" },
      conditions: [{ path: "ready", operator: "equals", value: true }],
      actions: [{ kind: "log", message: "should not run" }],
      enabled: true,
      actorId: "owner",
    });
    const job = await engine.run(workflow.id, { ready: false });
    expect(job.status).toBe("skipped");
    expect(job.skippedReason).toMatch(/conditions not met/);
    expect(job.outcomes).toHaveLength(0);
  });

  it("retries a failing action with backoff and then reports failure", async () => {
    const { registry, policy } = await buildStack();
    const engine = new AlphaAutomationEngine({ registry, policy, maxAttempts: 2, baseRetryDelayMs: 1 });
    const workflow = engine.registerWorkflow({
      name: "failing",
      description: "",
      trigger: { kind: "manual" },
      conditions: [],
      actions: [{ kind: "tool", toolName: "alpha.calculator", args: { expression: "1 / 0" } }],
      enabled: true,
      actorId: "owner",
    });
    const job = await engine.run(workflow.id);
    expect(job.status).toBe("failed");
    expect(job.attempts).toBe(2);
    expect(job.errors.join(" ")).toMatch(/division by zero/);
    expect(job.outcomes[0].ok).toBe(false);
    // The timeline shows the retry, not just the final attempt.
    expect(engine.executionHistory().length).toBeGreaterThan(0);
    const recorded = engine.executionHistory()[0];
    expect(recorded.attempts).toBe(2);
  }, 60_000);

  it("registers interval triggers only while started", async () => {
    const { registry, policy } = await buildStack();
    const engine = new AlphaAutomationEngine({ registry, policy });
    engine.registerWorkflow({
      name: "interval",
      description: "",
      trigger: { kind: "interval", intervalMs: 60_000 },
      conditions: [],
      actions: [{ kind: "log", message: "tick" }],
      enabled: true,
      actorId: "owner",
    });
    expect(engine.running).toBe(0);
    expect(engine.start()).toBe(1);
    expect(engine.running).toBe(1);
    engine.stop();
    expect(engine.running).toBe(0);
  });
});

describe("alpha workspace integration", () => {
  it("initialises every subsystem and reports honest status", async () => {
    const { workspace } = await buildStack();
    expect(workspace.isInitialised).toBe(true);
    const snapshot = workspace.snapshot();
    expect(snapshot.model.stage).toBe("untrained");
    expect(snapshot.model.parameterCount).toBeGreaterThan(1000);
    expect(snapshot.tokenizer.ready).toBe(true);
    expect(snapshot.tokenizer.vocabSize).toBeGreaterThan(20);
    expect(snapshot.tools.registered.length).toBeGreaterThan(5);
    expect(snapshot.statuses.model).toBe("untrained");
    expect(snapshot.statuses.mcp).toBe("not-configured");
    expect(snapshot.modules.length).toBeGreaterThan(10);
    expect(snapshot.security.chainIntact).toBe(true);
  });

  it("generates with an explicit untrained warning", async () => {
    const { workspace } = await buildStack();
    const result = workspace.generate("alpha is", { maxNewTokens: 5, seed: 3 });
    expect(result.generatedTokens).toBeGreaterThan(0);
    expect(result.modelStage).toBe("untrained");
    expect(result.warning).toMatch(/random initialisation/);
    const snapshot = workspace.snapshot();
    expect(snapshot.inference.requests).toBe(1);
    expect(snapshot.inference.tokensGenerated).toBe(result.generatedTokens);
  });

  it("ingests a document, answers with citations and records the run", async () => {
    const { workspace } = await buildStack();
    await workspace.ingestDocument({
      title: "Policy",
      content: "alpha requires approval before long term memory is written.",
      license: "CC0-1.0",
    });
    const answer = workspace.ask("when is approval required", { maxNewTokens: 6 });
    expect(answer.sources.length).toBeGreaterThan(0);
    const snapshot = workspace.snapshot();
    expect(snapshot.rag.vectors).toBeGreaterThan(0);
    expect(snapshot.observability.metrics.some((metric) => metric.name === "alpha.rag.retrievals")).toBe(true);
  });

  it("refuses to write long-term memory without approval", async () => {
    const { workspace } = await buildStack();
    expect(() =>
      workspace.writeMemory({ scope: "long-term", key: "user.name", content: "Ada" }),
    ).toThrow(/explicit approval/);
    const record = workspace.writeMemory({
      scope: "long-term",
      key: "user.name",
      content: "Ada",
      approved: true,
    });
    expect(record.approved).toBe(true);
    expect(workspace.snapshot().memory.counts["long-term"]).toBe(1);
    expect(workspace.forgetMemory(record.id)).toBe(true);
    expect(workspace.snapshot().memory.counts["long-term"]).toBe(0);
  });

  it("runs an agent task through the workspace and records observability", async () => {
    const { workspace } = await buildStack();
    const result = await workspace.runAgent("calculate 6 * 7", { maxSteps: 2 });
    expect(result.status).toBe("completed");
    const snapshot = workspace.snapshot();
    expect(snapshot.agents.runs).toBeGreaterThan(0);
    expect(snapshot.observability.metrics.some((metric) => metric.name === "alpha.agent.runs")).toBe(true);
  }, 60_000);
});

