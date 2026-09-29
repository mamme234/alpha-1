import { describe, expect, it } from "vitest";
import { AlphaContextEngine } from "../context/engine";
import { buildTokenizer } from "./helpers";
import { AlphaWorkspace } from "../workspace";
import { seedCorpusSlice } from "../datasets/seed-corpus";

async function buildEngine(contextLength = 48) {
  const tokenizer = buildTokenizer();
  return { tokenizer, engine: new AlphaContextEngine({ tokenizer, contextLength }) };
}

describe("alpha context engine", () => {
  it("counts tokens the same way the tokenizer does", async () => {
    const { engine, tokenizer } = await buildEngine();
    expect(engine.countTokens("alpha trains its own model")).toBe(tokenizer.countTokens("alpha trains its own model"));
  });

  it("includes everything when the budget is generous", async () => {
    const { engine } = await buildEngine(200);
    const assembled = engine.assembleConversation({
      prompt: "what does alpha store?",
      instruction: "Answer briefly.",
      sources: "Alpha stores checkpoints.",
      conversation: "user: hello",
    });
    expect(assembled.blocks.every((block) => block.status === "included")).toBe(true);
    expect(assembled.truncated).toBe(false);
    expect(assembled.usedTokens).toBeLessThanOrEqual(assembled.budgetTokens);
    expect(assembled.text).toContain("Instruction:");
    expect(assembled.text).toContain("Prompt:");
  });

  it("never drops the prompt when the window is tight", async () => {
    const { engine } = await buildEngine(24);
    const assembled = engine.assembleConversation({
      prompt: "summarise the training engine",
      instruction: "You are Alpha. Answer from what you know and never invent a fact.",
      sources: "the training engine stores weights, optimiser moments and the random generator position in a checkpoint",
      reserveForOutput: 4,
    });
    const promptReport = assembled.blocks.find((block) => block.kind === "prompt")!;
    expect(promptReport.status).not.toBe("dropped");
    expect(promptReport.includedTokens).toBeGreaterThan(0);
    expect(assembled.usedTokens).toBeLessThanOrEqual(assembled.budgetTokens);
    expect(assembled.text).toContain("summarise the training engine");
  });

  it("truncates sources before it truncates the prompt", async () => {
    const { engine } = await buildEngine(40);
    const assembled = engine.assembleConversation({
      prompt: "when does approval happen?",
      instruction: "Answer briefly.",
      sources: "long term memory requires explicit approval before it can be recalled and every approval is recorded in the audit log",
      reserveForOutput: 4,
    });
    const sources = assembled.blocks.find((block) => block.kind === "sources")!;
    const prompt = assembled.blocks.find((block) => block.kind === "prompt")!;
    expect(sources.status === "truncated" || sources.status === "dropped").toBe(true);
    expect(prompt.status).toBe("included");
    expect(assembled.notes.length).toBeGreaterThan(0);
  });

  it("reports dropped blocks with a reason instead of losing them silently", async () => {
    const { engine } = await buildEngine(12);
    const assembled = engine.assemble([
      { id: "prompt", kind: "prompt", text: "what is alpha?" },
      { id: "sources", kind: "sources", text: "a long block of retrieved text that cannot possibly fit" },
      { id: "memory", kind: "memory", text: "another long block of recalled memory" },
    ], { reserveForOutput: 2 });
    const dropped = assembled.blocks.filter((block) => block.status !== "included");
    expect(dropped.length).toBeGreaterThan(0);
    for (const block of dropped) {
      expect(block.reason.length).toBeGreaterThan(0);
    }
    expect(assembled.blocks.find((block) => block.id === "prompt")!.status).toBe("included");
  });

  it("honours per-kind caps, keeping the newest blocks", async () => {
    const { engine } = await buildEngine(200);
    const assembled = engine.assemble(
      [
        { id: "turn1", kind: "conversation", text: "first turn" },
        { id: "turn2", kind: "conversation", text: "second turn" },
        { id: "turn3", kind: "conversation", text: "third turn" },
        { id: "prompt", kind: "prompt", text: "continue" },
      ],
      { maxBlocksPerKind: { conversation: 1 } },
    );
    const kept = assembled.blocks.filter((block) => block.kind === "conversation" && block.status !== "dropped");
    expect(kept).toHaveLength(1);
    expect(kept[0].id).toBe("turn3");
    // Capped blocks are still reported, so nothing disappears without a trace.
    const capped = assembled.blocks.filter((block) => block.status === "dropped");
    expect(capped.map((block) => block.id).sort()).toEqual(["turn1", "turn2"]);
    expect(assembled.notes.join(" ")).toMatch(/capped at 1/);
  });

  it("refuses to silently drop a pinned block", async () => {
    const { engine } = await buildEngine(12);
    expect(() =>
      engine.assemble(
        [
          { id: "pinned", kind: "instruction", text: "an instruction far longer than the entire window allows", pinned: true },
          { id: "prompt", kind: "prompt", text: "hi" },
        ],
        { reserveForOutput: 0 },
      ),
    ).toThrow(/does not fit/);
  });

  it("fits a single string with each strategy", async () => {
    const { engine } = await buildEngine(64);
    const text = "alpha ".repeat(40);
    const right = engine.fit(text, { maxTokens: 8 });
    expect(right.truncated).toBe(true);
    expect(right.tokens).toBe(8);
    const left = engine.fit(text, { maxTokens: 8, strategy: "left" });
    expect(left.tokens).toBe(8);
    const middle = engine.fit(text, { maxTokens: 8, strategy: "middle" });
    expect(middle.text).toContain("…");
    expect(engine.fit("short", { maxTokens: 8 }).truncated).toBe(false);
  });

  it("keeps the measured prompt inside the budget at every window size", async () => {
    const tokenizer = buildTokenizer();
    const blocks = [
      { id: "instruction", kind: "instruction" as const, text: "You are Alpha, a self-owned model. Answer only from what you know." },
      { id: "memory", kind: "memory" as const, text: "the owner asked about checkpoints on tuesday", label: "recall" },
      { id: "sources", kind: "sources" as const, text: "a checkpoint stores weights, optimiser moments and the random generator position", label: "retrieval" },
      { id: "conversation", kind: "conversation" as const, text: "user: hello\nassistant: hello" },
      { id: "prompt", kind: "prompt" as const, text: "what does a checkpoint contain?", label: "user" },
    ];
    for (let contextLength = 12; contextLength <= 160; contextLength += 4) {
      for (const reserveForOutput of [0, 4, Math.floor(contextLength / 4)]) {
        const engine = new AlphaContextEngine({ tokenizer, contextLength });
        const assembled = engine.assemble(blocks, { reserveForOutput });
        expect(assembled.usedTokens).toBeLessThanOrEqual(assembled.budgetTokens);
        expect(assembled.budgetTokens + assembled.reserveForOutput).toBe(contextLength);
        // The question the user asked is the one thing that is never dropped.
        expect(assembled.blocks.find((block) => block.id === "prompt")!.status).not.toBe("dropped");
      }
    }
  });

  it("drops a header rather than the content it belongs to", async () => {
    // Two tokens short of fitting the whole prompt with its header, so the
    // decoration goes and the words stay.
    const tokenizer = buildTokenizer();
    const prompt = "what is alpha?";
    const withHeader = tokenizer.countTokens(`Prompt:\n(user)\n${prompt}`);
    const engine = new AlphaContextEngine({ tokenizer, contextLength: withHeader - 2 });
    const assembled = engine.assemble([{ id: "prompt", kind: "prompt", text: prompt, label: "user" }], {
      reserveForOutput: 0,
    });
    expect(assembled.text).toBe(prompt);
    expect(assembled.blocks[0].status).toBe("included");
    expect(assembled.blocks[0].reason).toMatch(/without its header/);
  });

  it("reports its own budget", async () => {
    const { engine } = await buildEngine(64);
    expect(engine.describe()).toEqual({ contextLength: 64, defaultReserveForOutput: 16 });
  });

  it("is on the workspace's generation path and reports the assembly", async () => {
    const tokenizer = buildTokenizer();
    const workspace = new AlphaWorkspace({
      config: {
        preset: "nano",
        model: { vocabSize: 256, contextLength: 64, dModel: 48, nHeads: 4, nLayers: 2, dFeedForward: 96 },
        tokenizer: { targetVocabSize: tokenizer.vocabSize },
        training: { batchSize: 2, seqLen: 24, totalSteps: 2, evalInterval: 0, checkpointInterval: 0, evalBatches: 1 },
        rag: { chunkTokens: 16, overlapTokens: 4, maxContextTokens: 32 },
      },
      dataset: seedCorpusSlice(4),
      logLevel: "warn",
    });
    await workspace.initialise();
    const result = workspace.generate("what does alpha store?", { maxNewTokens: 4, seed: 5 });
    expect(result.generatedTokens).toBeGreaterThan(0);
    const snapshot = workspace.snapshot();
    expect(snapshot.context.contextLength).toBe(64);
    expect(snapshot.context.last).not.toBeNull();
    expect(snapshot.context.last!.blocks.some((block) => block.kind === "instruction")).toBe(true);
    expect(snapshot.context.last!.usedTokens).toBeLessThanOrEqual(snapshot.context.last!.budgetTokens);
  }, 120_000);
});
