/**
 * Step 7 — chunked, resumable training runner for the frozen micro-v2 preset.
 *
 * The 180s terminal cap makes a single 64-step run impossible on this CPU
 * (measured ~12 s/step). Instead, the runner trains ONE chunk of
 * `STEPS_PER_CHUNK` steps per invocation and exits 0. A fresh run starts a new
 * deterministic model; a relaunched run resumes from the previous chunk's
 * checkpoint (same run: weights, optimiser moments, RNG stream and sampler
 * state) and continues. In this way the full 64-step budget is reached across
 * several fast invocations instead of one impossible 180s one.
 *
 *   STEPS_PER_CHUNK=10 bun scripts/alpha-train-step7.ts    # chunk 1 of 64, exit 0
 *   STEPS_PER_CHUNK=10 bun scripts/alpha-train-step7.ts    # chunk 2 of 64, exit 0
 *   ...                                                    # ...
 *
 * Checkpoint file (per chunk):  src/alpha/experiments/step7-micro-v2-chunk-<n>.alpha-ckpt.json
 */

import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";

import {
  ALPHA_MODEL_PRESETS_MICRO_V2,
  microV2Config,
} from "../src/alpha/model/micro-v2-config";
import {
  STEP7_CORPUS_VERSION,
  STEP7_RECORDED_AT,
  STEP7_SOURCE_ID,
  buildStep7Corpus,
} from "../src/alpha/datasets/step7-corpus";
import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import { AlphaTransformer } from "../src/alpha/model/transformer";
import { AlphaTrainer, createTrainingConfig, type TrainingConfig } from "../src/alpha/training/trainer";
import { modelConfigFingerprint, validateModelConfig } from "../src/alpha/model/config";
import {
  parseCheckpoint,
  type AlphaCheckpoint,
} from "../src/alpha/training/checkpoint";

const TOTAL_STEPS = 64;
const EXPERIMENTS_DIR = join(__dirname, "..", "src", "alpha", "experiments");

function rssMb(): string {
  if (process.memoryUsage?.rss) return (process.memoryUsage().rss / 1024 / 1024).toFixed(1);
  return "n/a";
}

/** Discover already-finished chunk numbers, oldest first. */
function existingChunkNumbers(): number[] {
  if (!existsSync(EXPERIMENTS_DIR)) return [];
  return readdirSync(EXPERIMENTS_DIR)
    .filter((f) => f.endsWith(".alpha-ckpt.json"))
    .map((f) => {
      const m = f.match(/^step7-micro-v2-chunk-(\d+)\.alpha-ckpt\.json$/);
      if (!m) throw new Error(`unexpected checkpoint name ${f}`);
      return Number(m[1]);
    })
    .sort((a, b) => a - b);
}

function readCheckpoint(path: string): AlphaCheckpoint {
  return parseCheckpoint(readFileSync(path, "utf8"));
}

function main(): void {
  const tStart = Date.now();
  const stepsPerChunk = parseInt(process.env.STEPS_PER_CHUNK ?? "30", 10);
  if (!Number.isInteger(stepsPerChunk) || stepsPerChunk < 1) {
    throw new Error(
      `STEPS_PER_CHUNK must be a positive integer, got ${process.env.STEPS_PER_CHUNK}`,
    );
  }

  console.log("=".repeat(74));
  console.log("Step 7 — chunked micro-v2 training runner");
  console.log(`STEPS_PER_CHUNK=${stepsPerChunk}  rss=${rssMb()}`);
  console.log("=".repeat(74));

  // ------------------------------------------------------------------
  // 1. Frozen micro-v2 config identity
  // ------------------------------------------------------------------
  const frozen = microV2Config();
  validateModelConfig(frozen);
  const configFingerprint = modelConfigFingerprint(frozen);

  console.log("\n[id-1] frozen config preset: alpha-micro-v2@0.1.0");
  console.log(`[id-1] config fingerprint: ${configFingerprint}`);
  console.log(
    `[id-1] name=${frozen.name} version=${frozen.version} ` +
      `vocabSize=${frozen.vocabSize} contextLength=${frozen.contextLength} ` +
      `dModel=${frozen.dModel} nHeads=${frozen.nHeads} nLayers=${frozen.nLayers} ` +
      `dFeedForward=${frozen.dFeedForward} positionalEncoding=${frozen.positionalEncoding} ` +
      `tieEmbeddings=${frozen.tieEmbeddings} initStd=${frozen.initStd}`,
  );

  // ------------------------------------------------------------------
  // 2. Dataset (ids 5-8)
  // ------------------------------------------------------------------
  const built = buildStep7Corpus();
  const corpusDocs = built.documents.length;
  const corpusDuplicates = built.duplicatesRemoved;
  const corpusChars = built.characters;
  const corpusTopics = Object.keys(built.topicCounts).length;
  const corpusLanguages = Object.keys(built.languageCounts).sort().join(",");

  console.log("\n[id-5] dataset version:        " + STEP7_CORPUS_VERSION);
  console.log(`[id-6] dataset source id:      ${STEP7_SOURCE_ID}`);
  console.log("[id-7] dataset recorded at:    " + STEP7_RECORDED_AT);
  console.log(
    `[id-8] built: ${corpusDocs} docs, ${corpusDuplicates} dup, ${corpusChars} chars, ` +
      `${corpusTopics} topics, languages=${corpusLanguages}`,
  );

  const dataset = {
    id: "dataset_step7",
    name: "alpha-step7",
    version: STEP7_CORPUS_VERSION,
    description: "Step 7 authored capability corpus",
    license: "Alpha-owned",
    source: STEP7_SOURCE_ID,
    documents: built.documents.map((d) => d.text),
  };

  // ------------------------------------------------------------------
  // 3. Tokenizer (id 9)
  // ------------------------------------------------------------------
  const tokenizer = AlphaTokenizer.train(dataset.documents, {
    vocabSize: Math.max(frozen.vocabSize, 128),
    version: "1.0.0",
    trainedOn: "alpha-step7@1.0.0",
  });

  console.log("\n[id-9] tokenizer version:      " + tokenizer.version);
  console.log(`[id-9] tokenizer vocabSize:    ${tokenizer.vocabSize}`);
  console.log(`[id-9] tokenizer fingerprint:  ${tokenizer.fingerprint()}`);
  console.log(`[id-9] tokenizer trainedOn:    ${tokenizer.trainedOn}`);

  // ------------------------------------------------------------------
  // 4. Model (id 10)
  // ------------------------------------------------------------------
  const model = new AlphaTransformer(frozen);
  console.log("\n[id-10] model parameterCount:  " + model.parameterCount);
  console.log(
    `[id-10] tokenizer vocabSize <= model vocabSize: ${tokenizer.vocabSize} <= ${frozen.vocabSize}`,
  );

  // ------------------------------------------------------------------
  // 5. Training: one chunk per invocation
  // ------------------------------------------------------------------
  if (!existsSync(EXPERIMENTS_DIR)) mkdirSync(EXPERIMENTS_DIR, { recursive: true });

  const existing = existingChunkNumbers();
  const resumeCheckpoint = existing.length > 0 ? readCheckpoint(
    join(EXPERIMENTS_DIR, `step7-micro-v2-chunk-${existing[existing.length - 1]}.alpha-ckpt.json`),
  ) : null;

  // Where this invocation's chunk starts (steps already completed).
  const chunkStart = resumeCheckpoint ? resumeCheckpoint.step : 0;
  const stepsThisChunk = Math.min(stepsPerChunk, TOTAL_STEPS - chunkStart);
  const chunkNumber = Math.floor(chunkStart / stepsPerChunk) + 1;

  if (stepsThisChunk <= 0) {
    console.log("\nStep 7 — nothing left to train (already at " + TOTAL_STEPS + " steps).");
    return;
  }

  const config: Partial<TrainingConfig> = {
    batchSize: 8,
    seqLen: 256,
    // The trainer's run loop ends at config.totalSteps.  It must be the
    // absolute step the run reaches for this chunk: `chunkStart` steps are
    // already done before this invocation, and `stepsThisChunk` stay.
    // On resume `stepCount` was restored to `chunkStart`, so `totalSteps`
    // must be `chunkStart + stepsThisChunk`; otherwise a fresh chunk (0)
    // would train the whole budget and a resumed chunk would train zero
    // steps, leaving an empty history whose final trainLoss is NaN.
    totalSteps: chunkStart + stepsThisChunk,
    learningRate: 0.002,
    warmupSteps: Math.min(5, stepsThisChunk),
    evalInterval: Math.min(16, stepsThisChunk),
    evalBatches: 2,
    validationFraction: 0.12,
    seed: 1337,
    gradientAccumulationSteps: 1,
  };

  const trainer = new AlphaTrainer({
    model,
    tokenizer,
    dataset: dataset,
    config,
    checkpointLabel: `step7-micro-v2-chunk-${chunkNumber}`,
    runId: "step7-micro-v2-chunked",
  });

  if (resumeCheckpoint) {
    trainer.resumeFrom(resumeCheckpoint);
    console.log(
      `\n  resumed from chunk ${chunkNumber - 1} checkpoint (step ${resumeCheckpoint.step}) ` +
        `-> training ${stepsThisChunk} more step(s) to step ${chunkStart + stepsThisChunk}`,
    );
  } else {
    console.log(`\n  [chunk ${chunkNumber}] fresh deterministic model (seed 1337, step ${chunkStart})`);
  }

  const chunkT0 = Date.now();
  const summary = trainer.trainToCompletion();
  const chunkMs = Date.now() - chunkT0;

  const lossCurve = trainer.history.map((p) => ({ step: p.step, loss: p.loss }));
  const lastHistoryPoint = trainer.history.length > 0 ? trainer.history[trainer.history.length - 1] : null;

  const throughput = chunkMs > 0 ? Math.round((summary.tokensPerStep / chunkMs) * 1000) : 0;
  const rss = rssMb();

  const lastLoss = lastHistoryPoint?.loss;
  console.log(
    `  chunk ${chunkNumber} done: steps ${summary.steps} | ` +
      `train ${lastLoss?.toFixed(4) ?? "n/a"} ` +
      `(best ${summary.bestLoss?.toFixed(4) ?? "n/a"}) | ` +
      `tokens/sec ${throughput} | rss ${rss}`,
  );

  const checkpointPath = join(
    EXPERIMENTS_DIR,
    `step7-micro-v2-chunk-${chunkNumber}.alpha-ckpt.json`,
  );
  if (summary.checkpoint) {
    writeFileSync(checkpointPath, JSON.stringify(summary.checkpoint, null, 2));
    console.log(`  checkpoint written: ${checkpointPath}`);
  }

  console.log("\n[id-11] loss curve:");
  for (const point of lossCurve) {
    console.log(`  step ${String(point.step).padStart(4)}  loss ${point.loss.toFixed(4)}`);
  }

  const overallMs = Date.now() - tStart;
  const overallTPS = overallMs > 0 ? Math.round((stepsThisChunk * 8 * 256) / overallMs * 1000) : 0;

  console.log("\n" + "=".repeat(74));
  console.log("Step 7 — final metrics");
  console.log("=".repeat(74));
  console.log(`total steps:          ${TOTAL_STEPS}`);
  console.log(`this chunk:           ${stepsThisChunk} step(s) (${chunkStart} -> ${chunkStart + stepsThisChunk})`);
  console.log(`chunk number:         ${chunkNumber}`);
  console.log(`final checkpoint:     ${checkpointPath}`);
  if (summary.checkpoint) {
    console.log(`final checkpoint id:  ${summary.checkpoint.id}`);
    console.log(`final checkpoint:     step ${summary.checkpoint.step}`);
    console.log(`final checkpoint:     size ${summary.checkpoint.sizeBytes} bytes`);
  }
  console.log(`chunk tokens/sec:     ${throughput}`);
  console.log(`total run rss:        ${rss}`);
  console.log(`total run duration:   ${Math.round(overallMs)} ms`);
  console.log("=".repeat(74));
  console.log("Step 7 chunked runner finished OK (exit 0).");
}

main();
