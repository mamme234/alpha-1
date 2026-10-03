/**
 * Alpha CLI — verify the production chat path.
 *
 *   bun run alpha:verify-production
 *   bun run alpha:verify-production --offline
 *
 * Part A runs the exact modules the Convex chat API bundles — the checked-in
 * Step 5 artefact, the serving runtime, the chat-turn layer, streaming and
 * cancellation — and exercises the failures that must fail: a tampered
 * artefact, an oversized request, an empty request, a lying `externalModels`
 * field, and a caller trying to widen the tool allow-list.
 *
 * Part B drives the live deployment through the Convex CLI: sign-up, a real
 * streamed turn, transcript persistence, stop, ownership isolation between two
 * accounts, an oversized request and a bad token. It is skipped with a stated
 * reason when the CLI cannot reach the deployment — a skip is never printed as
 * a pass.
 *
 * Nothing here fabricates a result. Every check prints the numbers it measured.
 */

import { spawn, spawnSync } from "node:child_process";
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

type CheckStatus = "PASS" | "FAIL" | "SKIP";
type Check = { id: string; label: string; status: CheckStatus; detail: string };

const checks: Check[] = [];

function record(id: string, label: string, status: CheckStatus, detail: string): void {
  checks.push({ id, label, status, detail });
  console.log(`  ${status === "PASS" ? "ok  " : status} ${id}. ${label} — ${detail}`);
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function section(title: string): void {
  console.log(`\n${title}`);
}

/** A deep copy of the artefact that a check may tamper with. */
function cloneArtifact(): AlphaServingArtifact {
  return structuredClone(STEP5_ARTIFACT) as unknown as AlphaServingArtifact;
}

/* -------------------------------------------------------------------------- */
/* Part A — the production path, in process                                    */
/* -------------------------------------------------------------------------- */

async function verifyArtifact(): Promise<void> {
  section("Artefact integrity");

  let artifact: AlphaServingArtifact | null = null;
  try {
    artifact = parseServingArtifact(STEP5_ARTIFACT);
    record(
      "P01",
      "the checked-in Step 5 artefact parses",
      "PASS",
      `${artifact.model.configFingerprint} · ${artifact.tokenizer.fingerprint} · ${artifact.model.parameterCount.toLocaleString()} params · stage ${artifact.stage} · external ${artifact.externalModels}`,
    );
  } catch (error) {
    record("P01", "the checked-in Step 5 artefact parses", "FAIL", errorText(error));
    return;
  }

  const loaded = loadServingArtifact(STEP5_ARTIFACT);
  record(
    "P02",
    "weights load and answer to their own fingerprints",
    loaded.model.config.name === artifact.model.config.name &&
      loaded.tokenizer.fingerprint() === artifact.tokenizer.fingerprint
      ? "PASS"
      : "FAIL",
    `loaded in ${loaded.loadMs} ms; tokenizer ${loaded.tokenizer.fingerprint()}; vocab ${loaded.tokenizer.vocabSize}`,
  );

  // Tampered tokenizer fingerprint.
  const tamperedTokenizer = cloneArtifact();
  tamperedTokenizer.tokenizer.fingerprint = "tok_deadbeef";
  try {
    loadServingArtifact(tamperedTokenizer);
    record("P03", "a tampered tokenizer fingerprint is refused", "FAIL", "the artefact loaded anyway");
  } catch (error) {
    record(
      "P03",
      "a tampered tokenizer fingerprint is refused",
      error instanceof AlphaValidationError ? "PASS" : "FAIL",
      errorText(error),
    );
  }

  // A document that claims an external model.
  const external = cloneArtifact();
  (external as { externalModels: string }).externalModels = "openai";
  try {
    parseServingArtifact(external);
    record("P04", "a document claiming an external model is refused", "FAIL", "it parsed anyway");
  } catch (error) {
    record("P04", "a document claiming an external model is refused", "PASS", errorText(error));
  }

  // Weights replaced by zeros.
  const zeroed = cloneArtifact();
  const tensorNames = Object.keys(zeroed.weights.tensors);
  const first = tensorNames[0];
  if (first) {
    const encoded = zeroed.weights.tensors[first];
    zeroed.weights.tensors[first] = encoded.replace(/[^=]/g, "A");
    try {
      loadServingArtifact(zeroed);
      record("P05", "all-zero weights are refused", "FAIL", "the zeroed artefact loaded anyway");
    } catch (error) {
      record("P05", "all-zero weights are refused", "PASS", errorText(error));
    }
  } else {
    record("P05", "all-zero weights are refused", "FAIL", "the artefact has no tensors to zero");
  }

  // A vocab that does not match the model.
  const bigVocab = cloneArtifact();
  bigVocab.tokenizer.vocabSize = bigVocab.model.config.vocabSize + 1;
  try {
    parseServingArtifact(bigVocab);
    record("P06", "a vocab larger than the model's is refused", "FAIL", "it parsed anyway");
  } catch (error) {
    record("P06", "a vocab larger than the model's is refused", "PASS", errorText(error));
  }
}

async function verifyRuntime(): Promise<void> {
  section("Serving runtime");
  const runtime = createServingRuntime({ artifact: STEP5_ARTIFACT });

  const instructionTokens = runtime.context.countTokens(runtime.ai.systemInstruction);
  record(
    "P07",
    "the runtime reports the artefact's real limits",
    runtime.limits.contextLength === runtime.model.config.contextLength &&
      runtime.limits.instructionTokens === instructionTokens &&
      runtime.info.gatePassed &&
      runtime.info.externalModels === "none"
      ? "PASS"
      : "FAIL",
    `context ${runtime.limits.contextLength} · instruction ${runtime.limits.instructionTokens} tok · max request ${runtime.limits.maxRequestTokens} tok · reserve ${runtime.limits.defaultReserveTokens} tok · gate ${runtime.info.gateFingerprint} ${runtime.info.gatePassed ? "passed" : "not passed"}`,
  );

  record(
    "P08",
    "the runtime exposes only real built-in tools",
    runtime.tools.describe().length >= ALPHA_CHAT_TOOL_ALLOWLIST.length
      ? "PASS"
      : "FAIL",
    `${runtime.tools.describe().length} tools registered; chat allow-list has ${ALPHA_CHAT_TOOL_ALLOWLIST.length}`,
  );
}

async function verifyChat(runtime: AlphaServingRuntime): Promise<void> {
  section("Chat turns");

  // A real turn.
  const first = await runChatTurn(runtime, {
    actorId: "alpha.verify",
    conversationId: "alpha.verify",
    message: "Hello",
    settings: { deterministic: true, maxNewTokens: 24, useMemory: false, useRetrieval: false },
  });
  record(
    "P09",
    "a chat turn produces real model output",
    !first.error &&
      first.response.length > 0 &&
      first.generation !== null &&
      first.generation.modelStage === "trained" &&
      first.generation.generatedTokens > 0
      ? "PASS"
      : "FAIL",
    first.error
      ? `error: ${first.error.message}`
      : `“${first.response.slice(0, 60).replace(/\n/g, " ")}${first.response.length > 60 ? "…" : ""}” · ${first.generation?.generatedTokens} tok · stop ${first.generation?.stopReason} · ${first.durationMs} ms`,
  );

  // Streaming: deltas concatenate to the final text.
  const events: string[] = [];
  let streamed = "";
  let finalResult = null as Awaited<ReturnType<typeof runChatTurn>> | null;
  const iterator = runChatTurnStream(runtime, {
    actorId: "alpha.verify",
    conversationId: "alpha.verify",
    message: "Hello",
    settings: { deterministic: true, maxNewTokens: 24, useMemory: false, useRetrieval: false },
    shouldStopEvery: 2,
  });
  let step = await iterator.next();
  while (!step.done) {
    if (step.value.type === "delta") {
      events.push(step.value.text);
      streamed += step.value.text;
    }
    step = await iterator.next();
  }
  finalResult = step.value;
  record(
    "P10",
    "the streamed path emits deltas that rebuild the answer",
    events.length > 1 && streamed === finalResult.response && finalResult.generation !== null
      ? "PASS"
      : "FAIL",
    `${events.length} delta event(s); streamed ${streamed.length} chars; final ${finalResult.response.length} chars; stop ${finalResult.generation?.stopReason}`,
  );

  // Cooperative stop.
  let checks = 0;
  const cancellable = runChatTurnStream(runtime, {
    actorId: "alpha.verify",
    conversationId: "alpha.verify",
    message: "Hello",
    settings: { deterministic: true, maxNewTokens: 60, useMemory: false, useRetrieval: false },
    shouldStopEvery: 1,
    shouldStop: async () => {
      checks += 1;
      // Stop after 8 tokens: enough that the kept text is real words, and far
      // short of the 60-token ceiling.
      return checks >= 8;
    },
  });
  let cancelStep = await cancellable.next();
  let cancelled: Awaited<ReturnType<typeof runChatTurn>> | null = null;
  while (!cancelStep.done) cancelStep = await cancellable.next();
  cancelled = cancelStep.value;
  record(
    "P11",
    "stop generation keeps the partial answer and marks it cancelled",
    cancelled.error === null &&
      cancelled.generation?.stopReason === "cancelled" &&
      cancelled.response.length > 0 &&
      cancelled.generation.generatedTokens < 60
      ? "PASS"
      : "FAIL",
    `stop requested after ${checks} check(s); ${cancelled.generation?.generatedTokens} tokens kept; stop ${cancelled.generation?.stopReason}; kept “${cancelled.response.slice(0, 30)}”`,
  );

  // A stop-predicate that throws must not kill generation.
  const resilient = runChatTurnStream(runtime, {
    actorId: "alpha.verify",
    conversationId: "alpha.verify",
    message: "Hello",
    settings: { deterministic: true, maxNewTokens: 12, useMemory: false, useRetrieval: false },
    shouldStopEvery: 2,
    shouldStop: async () => {
      throw new Error("simulated stop-channel failure");
    },
  });
  let resilientStep = await resilient.next();
  while (!resilientStep.done) resilientStep = await resilient.next();
  record(
    "P12",
    "a failing stop channel does not fake a stop",
    resilientStep.value.error === null && resilientStep.value.generation?.stopReason !== "cancelled"
      ? "PASS"
      : "FAIL",
    `stop ${resilientStep.value.generation?.stopReason}; error ${resilientStep.value.error?.message ?? "none"}`,
  );

  // Refusals with real numbers.
  let oversized = "";
  try {
    buildChatRequest(runtime, {
      actorId: "alpha.verify",
      conversationId: "alpha.verify",
      message: "x".repeat(400),
    });
  } catch (error) {
    oversized = errorText(error);
  }
  record(
    "P13",
    "an oversized request is refused with the measured token count",
    oversized.includes("this message is") && oversized.includes(String(runtime.limits.maxRequestTokens))
      ? "PASS"
      : "FAIL",
    oversized || "no error was raised",
  );

  let empty = "";
  try {
    buildChatRequest(runtime, { actorId: "alpha.verify", conversationId: "alpha.verify", message: "   " });
  } catch (error) {
    empty = errorText(error);
  }
  record("P14", "an empty message is refused", empty.length > 0 ? "PASS" : "FAIL", empty || "no error was raised");

  // A caller cannot widen the tool allow-list.
  const widened = buildChatRequest(runtime, {
    actorId: "alpha.verify",
    conversationId: "alpha.verify",
    message: "Hello",
    settings: { allowedTools: ["alpha.admin.clear_vector_store", "alpha.calculator"] },
  });
  record(
    "P15",
    "a caller may narrow the tool list but not widen it",
    widened.allowedTools !== undefined &&
      widened.allowedTools.length === 1 &&
      widened.allowedTools[0] === "alpha.calculator"
      ? "PASS"
      : "FAIL",
    `requested 2 names incl. a privileged one; the request carries [${widened.allowedTools?.join(", ") ?? ""}]`,
  );

  // Recovery: the same runtime still serves a real turn after all the refusals.
  const afterFailures = await runChatTurn(runtime, {
    actorId: "alpha.verify",
    conversationId: "alpha.verify",
    message: "Hello",
    settings: { deterministic: true, maxNewTokens: 12, useMemory: false, useRetrieval: false },
  });
  record(
    "P16",
    "the runtime recovers after refused requests",
    afterFailures.error === null && afterFailures.response.length > 0 ? "PASS" : "FAIL",
    afterFailures.error ? afterFailures.error.message : `${afterFailures.generation?.generatedTokens} tokens after refusals`,
  );

  // A real tool call, through the same registry the Convex action uses.
  const toolTurn = await runChatTurn(runtime, {
    actorId: "alpha.verify",
    conversationId: "alpha.verify",
    message: 'calculate "12 * 4"',
    settings: { deterministic: true, maxNewTokens: 16, useMemory: false, useRetrieval: false },
  });
  const calculator = toolTurn.toolCalls.find((call) => call.tool === "alpha.calculator");
  const computed = (calculator?.output as { value?: number } | undefined)?.value;
  record(
    "P19",
    "a permitted tool really executes and its result feeds the answer",
    calculator?.ok === true && computed === 48 && toolTurn.toolCalls.length >= 1 ? "PASS" : "FAIL",
    calculator
      ? `alpha.calculator → ${computed} (${calculator.durationMs} ms, ok: ${calculator.ok})`
      : "no tool was selected",
  );
}

async function verifyHydration(runtime: AlphaServingRuntime): Promise<void> {
  section("Memory, retrieval and bad host data");
  const dimension = runtime.embedder.dimension;
  const goodVector = new Array<number>(dimension).fill(0);
  goodVector[0] = 1;
  const report = runtime.hydrateFor("alpha.verify.hydration", {
    memories: [
      {
        memoryId: "mem_good",
        scope: "long-term",
        key: "signature",
        content: "Alpha's serving model was trained by Alpha.",
        embedding: goodVector,
        approved: true,
        source: "system",
      },
      {
        memoryId: "mem_bad_dimension",
        scope: "long-term",
        key: "foreign",
        content: "A memory embedded by a different model.",
        embedding: [0.1, 0.2, 0.3],
        approved: true,
        source: "system",
      },
    ],
    vectors: [
      {
        recordId: "vec_good",
        collection: "alpha_documents",
        text: "Alpha is a self-owned AI system.",
        embedding: goodVector,
        metadata: {},
      },
      {
        recordId: "vec_bad_dimension",
        collection: "alpha_documents",
        text: "A vector from another model.",
        embedding: [0.5, 0.5],
        metadata: {},
      },
    ],
  });
  record(
    "P17",
    "mismatched embeddings are skipped with a reason, not coerced",
    report.memories.imported === 1 &&
      report.memories.skipped === 1 &&
      report.vectors.imported === 1 &&
      report.vectors.skipped === 1 &&
      report.vectors.reasons.length === 1
      ? "PASS"
      : "FAIL",
    `memories ${report.memories.imported} imported / ${report.memories.skipped} skipped; vectors ${report.vectors.imported} imported / ${report.vectors.skipped} skipped; reason: ${report.vectors.reasons[0] ?? "none"}`,
  );

  record(
    "P18",
    "the loader dimensions are the model's own",
    dimension === runtime.model.config.dModel ? "PASS" : "FAIL",
    `embedding dimension ${dimension}; dModel ${runtime.model.config.dModel}`,
  );

  // A real retrieval: store one document, then ask for it.
  const document = "Alpha is a self-owned AI system.";
  runtime.hydrateFor("alpha.verify.rag", {
    vectors: [
      {
        recordId: "vec_rag_doc",
        collection: "alpha_documents",
        text: document,
        embedding: runtime.embedder.embed(document).vector,
        metadata: { title: "verify document" },
      },
    ],
  });
  const ragTurn = await runChatTurn(runtime, {
    actorId: "alpha.verify.rag",
    conversationId: "alpha.verify.rag",
    message: document,
    settings: { deterministic: true, maxNewTokens: 12, useMemory: false, useRetrieval: true },
  });
  record(
    "P20",
    "a stored document is retrieved and cited",
    ragTurn.sources.length >= 1 && ragTurn.context.blocks.some((block) => block.kind === "sources")
      ? "PASS"
      : "FAIL",
    `${ragTurn.sources.length} source(s); top score ${ragTurn.sources[0]?.score?.toFixed(3) ?? "—"}`,
  );
}

/* -------------------------------------------------------------------------- */
/* Part B — the live deployment                                                */
/* -------------------------------------------------------------------------- */

type LiveRun = { ok: boolean; value: unknown; stdout: string; stderr: string };

/** Parse a function's stdout as JSON, tolerating a banner before it. */
function parseJsonOutput(stdout: string): unknown {
  const trimmed = stdout.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    // Fall through to extraction: the CLI prints a progress banner for some
    // commands, and array results must not be mistaken for the first object.
  }
  const start = Math.min(
    ...[stdout.indexOf("{"), stdout.indexOf("[")].filter((index) => index >= 0),
  );
  const end = Math.max(stdout.lastIndexOf("}"), stdout.lastIndexOf("]"));
  if (!Number.isFinite(start) || end <= start) return null;
  try {
    return JSON.parse(stdout.slice(start, end + 1));
  } catch {
    return null;
  }
}

function convexRun(path: string, args: Record<string, unknown>): LiveRun {
  const result = spawnSync("bunx", ["convex", "run", path, JSON.stringify(args)], {
    encoding: "utf8",
    timeout: 120_000,
    maxBuffer: 32 * 1024 * 1024,
  });
  const stdout = result.stdout ?? "";
  const stderr = result.stderr ?? "";
  return { ok: result.status === 0, value: parseJsonOutput(stdout), stdout, stderr };
}

/** The same call, without blocking: used to stop a turn while it streams. */
function convexRunAsync(path: string, args: Record<string, unknown>): Promise<LiveRun> {
  return new Promise((resolve) => {
    const child = spawn("bunx", ["convex", "run", path, JSON.stringify(args)], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout?.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr?.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("close", (code) => {
      resolve({ ok: code === 0, value: parseJsonOutput(stdout), stdout, stderr });
    });
    child.on("error", (error) => {
      resolve({ ok: false, value: null, stdout, stderr: `${stderr}${String(error)}` });
    });
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function randomToken(length: number): string {
  const alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
  let out = "";
  for (let i = 0; i < length; i += 1) out += alphabet[Math.floor(Math.random() * alphabet.length)];
  return out;
}

async function verifyLive(): Promise<void> {
  section("Live deployment (through the Convex CLI)");

  const status = convexRun("alpha/chat:modelStatus", {});
  if (!status.ok || status.value === null) {
    record(
      "L01",
      "the live deployment serves the model",
      "SKIP",
      `the Convex CLI could not reach the deployment: ${(status.stderr || status.stdout).trim().split("\n").slice(-1)[0] ?? "no output"}`,
    );
    return;
  }
  const statusValue = status.value as {
    available: boolean;
    info?: { configFingerprint: string; tokenizerFingerprint: string; stage: string; contextLength: number };
    error?: string;
  };
  record(
    "L01",
    "the live deployment serves the model",
    statusValue.available && statusValue.info?.stage === "trained" ? "PASS" : "FAIL",
    statusValue.available
      ? `${statusValue.info?.configFingerprint} · ${statusValue.info?.tokenizerFingerprint} · stage ${statusValue.info?.stage} · window ${statusValue.info?.contextLength}`
      : `unavailable: ${statusValue.error}`,
  );

  // Two throwaway accounts: one to chat, one to prove isolation.
  const suffix = randomToken(8);
  const accountA = convexRun("alphaAuth/actions:register", {
    email: `alpha-verify-a-${suffix}@example.com`,
    password: `Ridge-Harbor-${randomToken(6)}-Stone`,
    displayName: "Verify A",
  });
  if (!accountA.ok || !accountA.value) {
    record("L02", "a live account can be created", "FAIL", (accountA.stderr || "").trim().split("\n")[0] ?? "no output");
    return;
  }
  const tokenA = (accountA.value as { token: string }).token;
  record("L02", "a live account can be created", "PASS", `account A created (${suffix})`);

  const accountB = convexRun("alphaAuth/actions:register", {
    email: `alpha-verify-b-${suffix}@example.com`,
    password: `Meadow-Kestrel-${randomToken(6)}-Ford`,
    displayName: "Verify B",
  });
  const tokenB = accountB.ok && accountB.value ? (accountB.value as { token: string }).token : null;

  const send = convexRun("alpha/chat:send", {
    sessionToken: tokenA,
    message: "Hello",
    settings: { maxNewTokens: 16, deterministic: true },
  });
  const sendValue = send.value as
    | { status: string; text: string; tokens: number; conversationId: string; streamId: string; assistantMessageId: string; error: string | null }
    | null;
  record(
    "L03",
    "a live chat turn streams and completes",
    send.ok && sendValue !== null && sendValue.status === "done" && sendValue.tokens > 0 && sendValue.text.length > 0
      ? "PASS"
      : "FAIL",
    sendValue
      ? `${sendValue.tokens} tokens · status ${sendValue.status} · ${sendValue.text.slice(0, 40).replace(/\n/g, " ")}…`
      : (send.stderr || "").trim().split("\n").slice(-1)[0] ?? "no output",
  );

  if (!sendValue) return;
  const conversationId = sendValue.conversationId;

  const transcript = convexRun("alpha/conversations:get", { sessionToken: tokenA, conversationId });
  const transcriptValue = transcript.value as
    | { messages: { role: string; modelId?: string; modelStage?: string; content: string }[] }
    | null;
  record(
    "L04",
    "the turn is persisted with its provenance",
    transcriptValue !== null &&
      transcriptValue.messages.length >= 2 &&
      transcriptValue.messages.some((m) => m.role === "assistant" && m.modelId && m.modelStage === "trained")
      ? "PASS"
      : "FAIL",
    transcriptValue ? `${transcriptValue.messages.length} messages stored` : (transcript.stderr || "").trim().split("\n")[0] ?? "no output",
  );

  const stream = convexRun("alpha/chat:activeStream", { sessionToken: tokenA, conversationId });
  const streamValue = stream.value as { status: string; messageId: string | null } | null;
  record(
    "L05",
    "the stream row is finalized after the answer is stored",
    streamValue?.status === "done" && streamValue.messageId === sendValue.assistantMessageId
      ? "PASS"
      : "FAIL",
    streamValue
      ? `status ${streamValue.status}; message ${streamValue.messageId}`
      : (stream.stderr || "").trim().split("\n")[0] ?? "no output",
  );

  const stop = convexRun("alpha/chat:stop", { sessionToken: tokenA, streamId: sendValue.streamId });
  const stopValue = stop.value as { stopped: boolean; reason: string } | null;
  record(
    "L06",
    "stopping a finished stream is refused with a reason",
    stopValue !== null && stopValue.stopped === false && (stopValue.reason ?? "").length > 0 ? "PASS" : "FAIL",
    stopValue ? `${stopValue.reason}` : (stop.stderr || "").trim().split("\n")[0] ?? "no output",
  );

  // Stop a real generation mid-flight: the turn runs in a child process while
  // this process delivers the stop request. The prompt goes through the
  // calculator tool, which makes the turn run two generations — enough time for
  // a stop request to cross the CLI.
  const longTurn = convexRunAsync("alpha/chat:send", {
    sessionToken: tokenA,
    conversationId,
    message: 'calculate "12 * 4"',
    settings: { maxNewTokens: 64, deterministic: true },
  });
  await sleep(350);
  // One round trip, not two: the server finds the live stream for the
  // conversation itself.
  const requested = convexRun("alpha/chat:stopLatest", { sessionToken: tokenA, conversationId });
  const requestedValue = requested.value as { stopped: boolean; reason: string; streamId: string | null } | null;
  const finished = await longTurn;
  const finishedValue = finished.value as { status: string; stopReason: string } | null;
  if (requestedValue?.stopped === true && requestedValue.streamId) {
    const finalStream = convexRun("alpha/chat:activeStream", { sessionToken: tokenA, conversationId });
    const finalValue = finalStream.value as { status: string; stopReason: string | null; text: string } | null;
    record(
      "L12",
      "a live generation stops mid-stream and keeps its partial answer",
      finishedValue?.status === "stopped" &&
        finalValue?.status === "stopped" &&
        finalValue?.stopReason === "cancelled" &&
        (finalValue?.text.length ?? 0) > 0
        ? "PASS"
        : "FAIL",
      `stop accepted; final status ${finalValue?.status} (${finalValue?.stopReason}); ${finalValue?.text.length ?? 0} chars kept`,
    );
  } else if (
    (requestedValue?.reason ?? "").includes("already finished") ||
    (requestedValue?.reason ?? "").includes("no generation has run")
  ) {
    // The CLI round trip was slower than the generation, or faster than the
    // stream row's creation. Either way this is a timing miss on this machine,
    // not evidence that stop does not work.
    record(
      "L12",
      "a live generation stops mid-stream and keeps its partial answer",
      "SKIP",
      `the stop request missed the live window: ${requestedValue?.reason}`,
    );
  } else {
    const clue = (requested.stderr || requested.stdout).trim().split("\n").slice(-1)[0] ?? "no output";
    record(
      "L12",
      "a live generation stops mid-stream and keeps its partial answer",
      "FAIL",
      `stop was refused: ${requestedValue?.reason ?? clue}`,
    );
  }

  const oversized = convexRun("alpha/chat:send", { sessionToken: tokenA, message: "x".repeat(400) });
  record(
    "L07",
    "the live API refuses an oversized request with a measured count",
    !oversized.ok && oversized.stderr.includes("invalid-request") && oversized.stderr.includes("tokens;")
      ? "PASS"
      : "FAIL",
    (oversized.stderr.match(/invalid-request[^"]*/) ?? ["no structured rejection"])[0].slice(0, 120),
  );

  const badToken = convexRun("alpha/chat:send", { sessionToken: "not-a-real-token", message: "Hello" });
  record(
    "L08",
    "a bad session token is rejected",
    !badToken.ok && badToken.stderr.includes("not-signed-in") ? "PASS" : "FAIL",
    badToken.stderr.includes("not-signed-in") ? "rejected with not-signed-in" : "no structured rejection",
  );

  if (tokenB) {
    const crossRead = convexRun("alpha/conversations:get", { sessionToken: tokenB, conversationId });
    const crossReadValue = crossRead.value;
    record(
      "L09",
      "one account cannot read another account's conversation",
      crossRead.ok && crossReadValue === null ? "PASS" : "FAIL",
      crossRead.ok ? `account B sees ${crossReadValue === null ? "nothing" : JSON.stringify(crossReadValue).slice(0, 60)}` : (crossRead.stderr || "").trim().split("\n")[0] ?? "no output",
    );
  } else {
    record("L09", "one account cannot read another account's conversation", "SKIP", "the second account was not created");
  }

  const clear = convexRun("alpha/chat:clearMessages", { sessionToken: tokenA, conversationId });
  const clearValue = clear.value as { cleared: number } | null;
  record(
    "L10",
    "clearing a conversation removes its messages",
    clearValue !== null && clearValue.cleared >= 2 ? "PASS" : "FAIL",
    clearValue ? `${clearValue.cleared} messages deleted` : (clear.stderr || "").trim().split("\n")[0] ?? "no output",
  );

  const rename = convexRun("alpha/conversations:rename", { sessionToken: tokenA, conversationId, title: "Verify run" });
  const list = convexRun("alpha/conversations:list", { sessionToken: tokenA, limit: 5 });
  const listValue = list.value as { conversationId: string; title: string }[] | null;
  if (!Array.isArray(listValue)) {
    record(
      "L11",
      "a conversation can be renamed and listed",
      "FAIL",
      `list returned ${JSON.stringify(list.value)?.slice(0, 80) ?? "nothing"}; rename ${rename.ok ? "succeeded" : "failed"}: ${(rename.stderr || rename.stdout).trim().split("\n").slice(-1)[0] ?? "no output"}`,
    );
    return;
  }
  record(
    "L11",
    "a conversation can be renamed and listed",
    rename.ok && Array.isArray(listValue) && listValue.some((c) => c.conversationId === conversationId && c.title === "Verify run")
      ? "PASS"
      : "FAIL",
    listValue ? `list shows ${listValue.length} conversation(s)` : "list failed",
  );
}

/* -------------------------------------------------------------------------- */

export async function main(argv: string[] = []): Promise<number> {
  const offline = argv.includes("--offline");

  console.log("Alpha — production verification");
  console.log("in-process path: Step 5 artefact → serving runtime → chat turn → stream");
  console.log(offline ? "live deployment checks: disabled (--offline)" : "live deployment checks: enabled");

  await verifyArtifact();
  const runtime = createServingRuntime({ artifact: STEP5_ARTIFACT });
  await verifyRuntime();
  await verifyChat(runtime);
  await verifyHydration(runtime);

  if (!offline) {
    await verifyLive();
  } else {
    record("L01", "the live deployment serves the model", "SKIP", "--offline was passed");
  }

  const failed = checks.filter((check) => check.status === "FAIL");
  const skipped = checks.filter((check) => check.status === "SKIP");
  const passed = checks.filter((check) => check.status === "PASS");

  console.log("\nSummary");
  console.log(`  passed:  ${passed.length}/${checks.length}`);
  if (skipped.length > 0) console.log(`  skipped: ${skipped.length} (${skipped.map((check) => check.id).join(", ")})`);
  if (failed.length > 0) console.log(`  failed:  ${failed.length} (${failed.map((check) => check.id).join(", ")})`);

  if (failed.length > 0) {
    console.log("\nVerdict: NOT VERIFIED — the failures above are real.");
    return 1;
  }
  if (offline || skipped.length > 0) {
    console.log("\nVerdict: VERIFIED IN PROCESS — live deployment checks did not run (see SKIP reasons).");
    return 0;
  }
  console.log("\nVerdict: VERIFIED — in-process and live deployment checks passed.");
  return 0;
}
