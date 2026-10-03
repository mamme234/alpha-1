/**
 * Step 7 — chunked, resumable training runner for the frozen micro-v2 preset.
 *
 * The 180s terminal cap makes a single 64-step run (measured ~15–30 s/step on this
 * CPU) impossible to finish in one invocation. We therefore train in small chunks:
 *
 *   1. Print the frozen micro-v2 config identity (fingerprint, 1,401,280 params).
 *   2. Build the Step 7 corpus, train the BPE tokenizer, build the model.
 *   3. For each chunk of `STEPS_PER_CHUNK` (default 30) steps:
 *        - resume from the previous chunk's checkpoint when one exists (same run:
 *          weights, optimiser moments, RNG stream and sampler state);
 *        - otherwise start a fresh deterministic model (seed 1337);
 *        - train exactly `stepsThisChunk` steps, then write a checkpoint
 *          `src/alpha/experiments/step7-micro-v2-chunk-<n>.alpha-ckpt.json`.
 *   4. Print the loss curve, tokens/sec and the final checkpoint path, then exit 0.
 *
 * The runner is resumable: relaunch it after a crash and it continues from the last
 * chunk checkpoint instead of restarting from scratch.
 */

import { performance } from "node:perf_hooks";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";

import {
  ALPHA_MODEL_PRESETS_MICRO_V2,
  microV2Config,
} from "../src/alpha/model/micro-v2-config";
import {
  STEP7_CORPUS_VERSION,
  STEP7_SOURCE_ID,
  buildStep7Corpus,
} from "../src/alpha/datasets/step7-corpus";
import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import {
  AlphaTrainer,
  createTrainingConfig,
  type TrainingConfig,
} from "../src/alpha/training/trainer";
import { modelConfigFingerprint, validateModelConfig } from "../src/alpha/model/config";
import {
  parseCheckpoint,
  type AlphaCheckpoint,
} from "../src/alpha/training/checkpoint";

const TOTAL_STEPS = 64;
const EXPERIMENTS_DIR = join(__dirname, "..", "src", "alpha", "experiments");
const RESULT_FILE = join(EXPERIMENTS_DIR, "step7-micro-v2-result.json");

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

function logTable(header: string, rows: Array<{ label: string; value: string }>): void {
  console.log(header);
  console.log(rows.map((r) => `  ${r.label.padEnd(22)} ${r.value}`).join("\n"));
}

async function main(): Promise<void> {
  const tStart = performance.now();
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
  console.log(`[id-10] tokenizer vocabSize <= model vocabSize: ${tokenizer.vocabSize} <= ${frozen.vocabSize}`);

  let chunkNumber = 0;
  let lastCheckpoint: AlphaCheckpoint | null = null;
  const chunkRecords: Array<{
    chunk: number;
    steps: number;
    startLoss: number | null;
    lastLoss: number | null;
    bestLoss: number | null;
    tokensPerSecond: number;
    checkpointPath: string;
    rssMb: string;
  }> = [];
  const lossCurve: Array<{ step: number; loss: number }> = [];

  while (remaining > 0) {
    chunkNumber += 1;
    const stepsThisChunk = Math.min(stepsPerChunk, remaining);
    const checkpointPath = join(
      experimentsDir,
      `step7-micro-v2-chunk-${chunkNumber}.alpha-ckpt.json`,
    );
    lastCheckpoint = existing.includes(chunkNumber) ? readCheckpoint(checkpointPath) : null;

    const config: Partial<TrainingConfig> = {
      batchSize: 8,
      seqLen: 256,
      totalSteps: stepsThisChunk,
      learningRate: 0.002,
      warmupSteps: 5,
      evalInterval: 16,
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

    if (lastCheckpoint) {
      trainer.resumeFrom(lastCheckpoint);
      console.log(`\n  resuming from chunk ${chunkNumber - 1} checkpoint (step ${lastCheckpoint.step})`);
    } else {
      console.log(`\n  [chunk ${chunkNumber}] fresh deterministic model (seed 1337, step 0)`);
    }

    const chunkT0 = performance.now();
    const summary = trainer.trainToCompletion();
    const chunkMs = performance.now() - chunkT0;

    for (const point of trainer.history) lossCurve.push({ step: point.step, loss: point.loss });

    const throughput = chunkMs > 0 ? Math.round((summary.tokensPerStep / chunkMs) * 1000) : 0;
    const rss = rssMb();

    console.log(
      `  chunk ${chunkNumber} done: steps ${summary.steps} | ` +
        `train ${summary.lastLoss?.toFixed(4) ?? "n/a"} ` +
        `(best ${summary.bestLoss?.toFixed(4) ?? "n/a"}) | ` +
        `tokens/sec ${throughput} | rss ${rss}`,
    );

    writeFileSync(
      checkpointPath,
      JSON.stringify(summary.checkpoint ?? readCheckpoint(checkpointPath), null, 2),
    );

    chunkRecords.push({
      chunk: chunkNumber,
      steps: summary.steps,
      startLoss: summary.firstLoss,
      lastLoss: summary.lastLoss,
      bestLoss: summary.bestLoss,
      tokensPerSecond: throughput,
      checkpointPath,
      rssMb: rss,
    });

    remaining -= stepsThisChunk;
  }

  // ------------------------------------------------------------------
  // 6. Final report
  // ------------------------------------------------------------------
  const finalCheckpointPath = chunkRecords[chunkRecords.length - 1]!.checkpointPath;
  const finalRss = chunkRecords[chunkRecords.length - 1]!.rssMb;

  console.log("\n" + "=".repeat(74));
  console.log("Step 7 — final metrics");
  console.log("=".repeat(74));
  console.log(`total steps:          ${TOTAL_STEPS}`);
  console.log(`total tokens:         ${TOTAL_STEPS * 8 * 256}`);
  console.log(`final checkpoint:     ${finalCheckpointPath}`);
  console.log(`final checkpoint:     ${summary.checkpoint?.id} at step ${summary.checkpoint?.step ?? "n/a"}`);

  console.log("\n[id-11] loss curve:");
  for (const point of lossCurve) {
    console.log(`  step ${String(point.step).padStart(3)}  loss ${point.loss.toFixed(4)}`);
  }

  const overallMs = performance.now() - tStart;
  const overallTPS = overallMs > 0 ? Math.round((TOTAL_STEPS * 8 * 256) / overallMs * 1000) : 0;
  console.log(`\noverall tokens/sec:   ${overallTPS}`);
  console.log(`total run rss:        ${finalRss}`);
  console.log(`total run duration:   ${Math.round(overallMs)} ms`);

  // Save a JSON audit record.
  const result = {
    runner: "scripts/alpha-train-step7.ts",
    frozen: {
      preset: frozen.name,
      version: frozen.version,
      fingerprint: configFingerprint,
      parameterCount: model.parameterCount,
    },
    dataset: {
      version: STEP7_CORPUS_VERSION,
      sourceId: STEP7_SOURCE_ID,
      recordedAt: STEP7_RECORDED_AT,
      documents: corpusDocs,
      duplicatesRemoved: corpusDuplicates,
      characters: corpusChars,
      topics: corpusTopics,
      languages: corpusLanguages,
    },
    tokenizer: {
      version: tokenizer.version,
      vocabSize: tokenizer.vocabSize,
      fingerprint: tokenizer.fingerprint(),
      trainedOn: tokenizer.trainedOn,
    },
    training: {
      totalSteps: TOTAL_STEPS,
      batchSize: 8,
      seqLen: 256,
      learningRate: 0.002,
      warmupSteps: 5,
      evalInterval: 16,
      evalBatches: 2,
      validationFraction: 0.12,
      seed: 1337,
      gradientAccumulationSteps: 1,
      chunkSize: stepsPerChunk,
      chunks: chunkRecords,
      lossCurve,
      tokensPerSecond: overallTPS,
      durationMs: Math.round(overallMs),
      finalCheckpoint: finalCheckpointPath,
      finalCheckpointId: summary.checkpoint?.id ?? null,
      finalCheckpointStep: summary.checkpoint?.step ?? null,
      finalCheckpointSizeBytes: summary.checkpoint?.sizeBytes ?? null,
    },
  };
  writeFileSync(RESULT_FILE, JSON.stringify(result, null, 2));
  console.log(`\naudit record written: ${RESULT_FILE}`);
  console.log("=".repeat(74));
  console.log("Step 7 chunked runner finished OK.");
}

main().catch((error) => {
  console.error("FATAL:", error instanceof Error ? error.message : String(error));
  process.exit(1);
});
