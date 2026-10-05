/**
 * Alpha — backend/CLI vs frontend byte-for-byte comparison (READ-ONLY).
 *
 * Two paths, one prompt, four prompts actually:
 *
 *   PATH A "frontend/backend" — the *exact* code `src/convex/alpha/chat.ts`
 *     runs: `createServingRuntime({ artifact: STEP5_ARTIFACT })` then
 *     `buildChatRequest` + `respondStream`, concatenating the `delta` events
 *     exactly as runTurn() batches them. No Convex round trip is needed: the
 *     runtime is the same object, and `text` is what the Chat page renders.
 *
 *   PATH B "CLI/checkpoint" — the final mv2 checkpoint driven directly through
 *     AlphaInferenceEngine, i.e. what a CLI user gets.
 *
 * If PATH A and PATH B disagree, the difference is the serving configuration,
 * not the model. This prints the byte-level diff.
 *
 *   bun scripts/alpha-serving-vs-cli.ts
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

import STEP5_ARTIFACT from "../src/alpha/serving/step5-artifact.json";
import { createServingRuntime } from "../src/alpha/serving/runtime";
import { buildChatRequest } from "../src/alpha/serving/chat";
import { parseCheckpoint } from "../src/alpha/training/checkpoint";
import { AlphaTransformer } from "../src/alpha/model/transformer";
import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import { AlphaInferenceEngine, SAMPLING_PRESETS } from "../src/alpha/inference/engine";

const ROOT = join(__dirname, "..");
const CKPT = join(ROOT, "src/alpha/experiments/language/language-mv2-final.alpha-ckpt.json");

const PROMPTS = ["Hello", "Hi, I want to ask you something.", "What is Alpha?", "Tell me about Ethiopia."];

function hex(s: string): string {
  return Buffer.from(s, "utf8").toString("hex");
}

async function backendText(prompt: string): Promise<{ text: string; error: string | null }> {
  try {
    const runtime = createServingRuntime({ artifact: STEP5_ARTIFACT, logLevel: "warn" });
    runtime.hydrateFor("diag-actor", { memories: [], vectors: [] });
    const request = buildChatRequest(runtime, {
      actorId: "diag-actor",
      conversationId: "diag-conv",
      message: prompt,
      history: [],
      settings: { maxNewTokens: 40, deterministic: true, useMemory: false, useRetrieval: false },
    });
    const events = runtime.respondStream(request, { shouldStop: async () => false, shouldStopEvery: 8 });
    let text = "";
    let step = await events.next();
    while (!step.done) {
      const event = step.value;
      if (event.type === "delta") text += event.text;
      step = await events.next();
    }
    return { text, error: null };
  } catch (error) {
    return { text: "", error: error instanceof Error ? error.message : String(error) };
  }
}

async function main(): Promise<void> {
  console.log("=".repeat(74));
  console.log("SERVING CONFIG ACTUALLY IN EFFECT");
  console.log("=".repeat(74));
  const runtime = createServingRuntime({ artifact: STEP5_ARTIFACT, logLevel: "warn" });
  console.log(`import in chat.ts      : src/alpha/serving/step5-artifact.json`);
  console.log(`artifact stage         : ${STEP5_ARTIFACT.stage}`);
  console.log(`artifact model         : ${STEP5_ARTIFACT.model.config.name} v${STEP5_ARTIFACT.model.config.version}`);
  console.log(`artifact params        : ${STEP5_ARTIFACT.model.parameterCount}`);
  console.log(`artifact contextLength : ${STEP5_ARTIFACT.model.config.contextLength}`);
  console.log(`artifact tokenizer     : ${STEP5_ARTIFACT.tokenizer.fingerprint}`);
  console.log(`artifact tokensSeen    : ${STEP5_ARTIFACT.training.tokensSeen}`);
  console.log(`artifact valLoss       : ${STEP5_ARTIFACT.training.validationLoss}`);
  console.log(`runtime modelId        : ${runtime.modelId}`);
  console.log(`runtime tokenizer fp   : ${runtime.tokenizer.fingerprint()}`);
  console.log(`runtime limits         : ${JSON.stringify(runtime.limits)}`);
  console.log(`generation defaults    : ${JSON.stringify(runtime.generationDefaults)}`);

  const ck = parseCheckpoint(readFileSync(CKPT, "utf8"));
  console.log(`\nCLI checkpoint model   : ${ck.config.name} v${ck.config.version}`);
  console.log(`CLI checkpoint ctx     : ${ck.config.contextLength}`);
  console.log(`CLI checkpoint tok fp  : ${ck.tokenizer.fingerprint}`);
  console.log(`CLI checkpoint tokens  : ${ck.tokensSeen}`);
  console.log(`CLI checkpoint valLoss : ${ck.metrics.validationLoss}`);

  console.log(
    `\nSAME MODEL?      ${STEP5_ARTIFACT.model.config.name === ck.config.name ? "same name" : "DIFFERENT"}`,
  );
  console.log(
    `SAME CHECKPOINT? tokensSeen ${STEP5_ARTIFACT.training.tokensSeen} vs ${ck.tokensSeen} -> ${
      STEP5_ARTIFACT.training.tokensSeen === ck.tokensSeen ? "yes" : "NO — backend is a DIFFERENT, older model"
    }`,
  );
  console.log(
    `SAME TOKENIZER?  ${STEP5_ARTIFACT.tokenizer.fingerprint} vs ${ck.tokenizer.fingerprint} -> ${
      STEP5_ARTIFACT.tokenizer.fingerprint === ck.tokenizer.fingerprint
        ? "yes"
        : `NO — ${STEP5_ARTIFACT.tokenizer.fingerprint} is not ${ck.tokenizer.fingerprint}`
    }`,
  );

  const tokenizer = AlphaTokenizer.fromJSON(ck.tokenizer.snapshot);
  const model = new AlphaTransformer(ck.config);
  model.loadWeights(ck.weights);
  const cliEngine = new AlphaInferenceEngine({ model, tokenizer, stage: ck.stage ?? "trained" });

  console.log(`\n${"=".repeat(74)}`);
  console.log("BYTE-FOR-BYTE COMPARISON (deterministic, maxNewTokens=40)");
  console.log("=".repeat(74));

  for (const prompt of PROMPTS) {
    const back = await backendText(prompt);
    const cli = cliEngine.generate(prompt, {
      ...SAMPLING_PRESETS.greedy,
      maxNewTokens: 40,
      seed: 1337,
    });

    console.log(`\nprompt: ${JSON.stringify(prompt)}`);
    console.log(`  backend  text: ${JSON.stringify(back.text)}`);
    if (back.error) console.log(`  backend ERROR : ${back.error}`);
    console.log(`  backend  hex  : ${hex(back.text)}`);
    console.log(`  CLI      text : ${JSON.stringify(cli.text)}`);
    console.log(`  CLI      hex  : ${hex(cli.text)}`);
    console.log(`  backend bytes : ${Buffer.byteLength(back.text, "utf8")}`);
    console.log(`  CLI     bytes : ${Buffer.byteLength(cli.text, "utf8")}`);
    console.log(`  BYTE-IDENTICAL: ${back.text === cli.text ? "YES" : "NO"}`);
    console.log(`  CLI stopReason: ${cli.stopReason}, generated=${cli.generatedTokens}`);
  }

  console.log(`\n${"=".repeat(74)}`);
  console.log("No weights were modified. No training. No external model.");
  console.log("=".repeat(74));
}

await main();