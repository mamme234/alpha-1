/**
 * Alpha — Language training runner (chunked, resumable).
 *
 * ONE experiment, one purpose: continue training the frozen micro-v2
 * architecture on a much larger Alpha-owned corpus, in resumable chunks that
 * each finish well inside the Freebuff terminal cap (max 180 s), so the host
 * never has to hold the whole run at once.
 *
 * This script does NOT decide the budget; it executes the one that was
 * measured on this host:
 *   - model: alpha-micro-v2 (1,401,280 params, context 256, vocab 768)
 *   - corpus: Alpha-owned mixture, 4,289 docs / 2.13 M chars / 1.16 M tokens
 *   - batch 4 x seq 256 = 1,024 tokens/step at ~5.9 s/step (~170 tok/s)
 *   - GC between steps keeps RSS ~450 MB (batch 8 OOM-killed at 1.6 GB)
 *
 * Modes
 * -----
 *   PREP=1   Build the corpus documents, train the tokenizer once, cache the
 *            tokenizer snapshot next to the checkpoints. Writes no model
 *            weights. Safe to re-run.
 *   default  Train ONE chunk of STEPS_PER_CHUNK optimiser steps, resume from
 *            the latest checkpoint, save the latest checkpoint (overwritten),
 *            append a log line. Re-run until it prints COMPLETE.
 *
 * Env
 * ---
 *   STEPS_PER_CHUNK   steps this invocation          (default 18)
 *   TARGET_STEPS      total run length in steps      (default 1020)
 *   PREP=1            prepare corpus + tokenizer only
 *
 * Files (all under src/alpha/experiments/language/)
 * -------------------------------------------------
 *   tokenizer.json                          cached tokenizer snapshot
 *   language-mv2-latest.alpha-ckpt.json     latest checkpoint (overwritten)
 *   language-mv2-final.alpha-ckpt.json      written when TARGET_STEPS is reached
 *   language-run-log.jsonl                  one JSON line per finished chunk
 */

import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";

import { microV2Config } from "../src/alpha/model/micro-v2-config";
import { AlphaTransformer } from "../src/alpha/model/transformer";
import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import { AlphaTrainer, type TrainingConfig } from "../src/alpha/training/trainer";
import {
  parseCheckpoint,
  type AlphaCheckpoint,
} from "../src/alpha/training/checkpoint";
import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { buildGeneratedCorpus } from "../src/alpha/datasets/generated-corpus";
import { buildStep7Corpus } from "../src/alpha/datasets/step7-corpus";

const ROOT = join(__dirname, "..");
const DIR = join(ROOT, "src", "alpha", "experiments", "language");
const TOKENIZER_PATH = join(DIR, "tokenizer.json");
const LATEST_CKPT = join(DIR, "language-mv2-latest.alpha-ckpt.json");
const FINAL_CKPT = join(DIR, "language-mv2-final.alpha-ckpt.json");
const LOG_PATH = join(DIR, "language-run-log.jsonl");

const STEPS_PER_CHUNK = intEnv("STEPS_PER_CHUNK", 18);
const TARGET_STEPS = intEnv("TARGET_STEPS", 1020);
const PREP = process.env.PREP === "1";

const DATASET_ID = "dataset_language_mixture";
const DATASET_NAME = "alpha-language-mixture";
const DATASET_VERSION = "1.0.0";
const DATASET_SOURCE = "alpha-authored+generated+step7+repo-prose";

const EPOCHS_NOTE =
  "two authored seeds, generated corpus, step7 corpus and repository prose";

function intEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive integer, got ${raw}`);
  }
  return value;
}

function rssMb(): number {
  return Number((process.memoryUsage().rss / 1048576).toFixed(1));
}

type BunLike = { gc?: (force?: boolean) => void };
const bun = (globalThis as { Bun?: BunLike }).Bun;

/** Force a GC between steps: without it Bun's heap grew until the OOM killer hit. */
function collect(): void {
  bun?.gc?.(true);
}

/** The Alpha-owned corpus mixture. Order is fixed so chunk runs are identical. */
function buildMixture(): { documents: string[]; chars: number } {
  const authoredA = buildAuthoredCorpus(2400, 20260101).map((d) => d.text);
  const authoredB = buildAuthoredCorpus(1200, 20260202).map((d) => d.text);
  const generated = buildGeneratedCorpus(600, 20250930) as unknown as {
    documents?: unknown[];
  };
  const generatedDocs = (generated.documents ?? []).map((d) =>
    typeof d === "string" ? d : (d as { text: string }).text,
  );
  const step7Docs = buildStep7Corpus().documents.map((d) => d.text);
  const proseFiles = [
    join(ROOT, "README.md"),
    join(ROOT, "CONTRIBUTING.md"),
    ...readdirSync(join(ROOT, "docs"))
      .filter((f) => f.endsWith(".md"))
      .map((f) => join(ROOT, "docs", f)),
  ];
  const prose = proseFiles.map((f) => readFileSync(f, "utf8")).join("\n\n");

  const documents = [...authoredA, ...authoredB, ...generatedDocs, ...step7Docs, prose];
  return { documents, chars: documents.reduce((n, d) => n + d.length, 0) };
}

/**
 * Learning rate of the *global* run at a 1-indexed step: warmup 40 steps to
 * 0.002, then cosine down to 0.0002 at TARGET_STEPS. Each chunk's trainer uses
 * a constant rate (this value), so the schedule decays across chunks instead of
 * resetting each chunk.
 */
function globalLearningRate(step: number): number {
  const peak = 0.002;
  const floor = 0.0002;
  const warmup = 40;
  if (step <= warmup) return peak * (step / warmup);
  const progress = Math.min(1, (step - warmup) / Math.max(1, TARGET_STEPS - warmup));
  return floor + (peak - floor) * 0.5 * (1 + Math.cos(Math.PI * progress));
}

function readLatestCheckpoint(): AlphaCheckpoint | null {
  if (!existsSync(LATEST_CKPT)) return null;
  return parseCheckpoint(readFileSync(LATEST_CKPT, "utf8"));
}

function main(): void {
  const tStart = Date.now();
  if (!existsSync(DIR)) mkdirSync(DIR, { recursive: true });

  console.log("=".repeat(74));
  console.log("Alpha — language training runner (micro-v2, chunked)");
  console.log(
    `${PREP ? "mode: PREP (corpus + tokenizer only)" : `mode: TRAIN STEPS_PER_CHUNK=${STEPS_PER_CHUNK} TARGET_STEPS=${TARGET_STEPS}`}` +
      `  rss=${rssMb()} MB`,
  );
  console.log("=".repeat(74));

  // ------------------------------------------------------------------
  // Corpus (deterministic every run)
  // ------------------------------------------------------------------
  const tBuild = Date.now();
  const { documents, chars } = buildMixture();
  console.log(
    `corpus: ${documents.length} docs, ${chars} chars built in ${Date.now() - tBuild} ms`,
  );

  if (PREP) {
    // ------------------------------------------------------------------
    // Tokenizer: train once, cache the snapshot for every later chunk
    // ------------------------------------------------------------------
    const tTok = Date.now();
    const tokenizer = AlphaTokenizer.train(documents, {
      vocabSize: 768,
      version: "1.0.0",
      trainedOn: `${DATASET_NAME}@${DATASET_VERSION}`,
    });
    console.log(
      `tokenizer: trained in ${Date.now() - tTok} ms — vocab ${tokenizer.vocabSize}, ` +
        `trainedOn ${tokenizer.trainedOn}`,
    );
    console.log(`tokenizer: fingerprint ${tokenizer.fingerprint()}`);
    writeFileSync(TOKENIZER_PATH, JSON.stringify(tokenizer.toJSON()));
    console.log(`tokenizer: snapshot written to ${TOKENIZER_PATH}`);

    // ------------------------------------------------------------------
    // Prove encoding + trainer setup work and measure the per-chunk cost
    // ------------------------------------------------------------------
    const tEncode = Date.now();
    const dataset = {
      id: DATASET_ID,
      name: DATASET_NAME,
      version: DATASET_VERSION,
      description: `Alpha-owned language mixture: ${EPOCHS_NOTE}`,
      license: "Alpha-owned",
      source: DATASET_SOURCE,
      documents,
    };
    const model = new AlphaTransformer(microV2Config());
    const trainer = new AlphaTrainer({
      model,
      tokenizer,
      dataset,
      config: {
        batchSize: 4,
        seqLen: 256,
        totalSteps: 1,
        learningRate: 0.002,
        warmupSteps: 0,
        schedule: "constant",
        minFactor: 1,
        evalInterval: 0,
        evalBatches: 1,
        validationFraction: 0.12,
        seed: 1337,
        checkpointInterval: 0,
        gradientAccumulationSteps: 1,
      },
      checkpointLabel: "alpha-language-prep",
      runId: "alpha-language-prep",
    });
    const stats = trainer.corpus.stats;
    console.log(
      `encode: corpus encoded in ${Date.now() - tEncode} ms — ` +
        `total ${stats.totalTokens} tokens, train ${stats.trainTokens}, validation ${stats.validationTokens}`,
    );
    console.log(
      `PREP done in ${Date.now() - tStart} ms. Run training with:\n` +
        `  STEPS_PER_CHUNK=${STEPS_PER_CHUNK} bun scripts/alpha-language.ts`,
    );
    return;
  }

  // ------------------------------------------------------------------
  // TRAIN — one chunk
  // ------------------------------------------------------------------
  if (!existsSync(TOKENIZER_PATH)) {
    throw new Error(
      `tokenizer cache missing at ${TOKENIZER_PATH}; run PREP=1 bun scripts/alpha-language.ts first`,
    );
  }
  const snapshot = JSON.parse(readFileSync(TOKENIZER_PATH, "utf8"));
  const tokenizer = AlphaTokenizer.fromJSON(snapshot);
  console.log(
    `tokenizer: loaded from cache — vocab ${tokenizer.vocabSize}, fingerprint ${tokenizer.fingerprint()}`,
  );

  const dataset = {
    id: DATASET_ID,
    name: DATASET_NAME,
    version: DATASET_VERSION,
    description: `Alpha-owned language mixture: ${EPOCHS_NOTE}`,
    license: "Alpha-owned",
    source: DATASET_SOURCE,
    documents,
  };

  const resumed = readLatestCheckpoint();
  if (resumed) {
    console.log(
      `resume: latest checkpoint ${resumed.id} at step ${resumed.step} ` +
        `(metrics: train ${resumed.metrics.trainLoss}, val ${resumed.metrics.validationLoss}, ` +
        `sampler state: ${resumed.sampler !== undefined && resumed.sampler !== null ? "present" : "MISSING"})`,
    );
  } else {
    console.log("resume: none — starting a fresh deterministic model (seed 1337)");
  }

  const chunkStart = resumed ? resumed.step : 0;
  if (chunkStart >= TARGET_STEPS) {
    console.log(`nothing left to train (step ${chunkStart} >= target ${TARGET_STEPS})`);
    return;
  }
  const stepsThisChunk = Math.min(STEPS_PER_CHUNK, TARGET_STEPS - chunkStart);
  const chunkNumber = Math.floor(chunkStart / STEPS_PER_CHUNK) + 1;
  const chunkEnd = chunkStart + stepsThisChunk;

  const learningRate = Number(globalLearningRate(chunkStart + 1).toFixed(6));
  const config: Partial<TrainingConfig> = {
    batchSize: 4,
    seqLen: 256,
    // Loop control: the trainer stops at config.totalSteps. On resume the step
    // counter is already chunkStart, so the target for this invocation is the
    // end of this chunk (same pattern as the Step 7 chunked runner).
    totalSteps: chunkEnd,
    learningRate,
    warmupSteps: 0,
    schedule: "constant",
    minFactor: 1,
    evalInterval: Math.min(10, stepsThisChunk),
    evalBatches: 1,
    validationFraction: 0.12,
    seed: 1337,
    checkpointInterval: 0,
    gradientAccumulationSteps: 1,
  };

  const model = new AlphaTransformer(microV2Config());
  const trainer = new AlphaTrainer({
    model,
    tokenizer,
    dataset,
    config,
    checkpointLabel: `alpha-language-mv2-chunk-${chunkNumber}`,
    runId: "alpha-language-mv2",
    isFineTune: chunkStart > 0,
  });
  const stats = trainer.corpus.stats;
  console.log(
    `corpus: encoded — total ${stats.totalTokens}, train ${stats.trainTokens}, ` +
      `validation ${stats.validationTokens} | model params ${model.parameterCount}`,
  );
  if (resumed) trainer.resumeFrom(resumed);

  console.log(
    `\nchunk ${chunkNumber}: steps ${chunkStart} -> ${chunkEnd} ` +
      `(lr ${learningRate})  tokensSoFar ${chunkStart * 1024}`,
  );

  const chunkT0 = Date.now();
  const stepMs: number[] = [];
  let lastTrainLoss: number | null = null;
  let lastValidationLoss: number | null = null;
  let peakRss = rssMb();

  const iterator = trainer.run();
  let result = iterator.next();
  while (!result.done) {
    const event = result.value;
    if (event.type === "step") {
      const rss = rssMb();
      peakRss = Math.max(peakRss, rss);
      stepMs.push(event.point.elapsedMs);
      lastTrainLoss = event.point.loss;
      console.log(
        `  step ${event.point.step}: loss ${event.point.loss.toFixed(4)} ` +
          `| ${event.point.elapsedMs - (stepMs.length > 1 ? stepMs[stepMs.length - 2] : 0)} ms ` +
          `| rss ${rss} MB`,
      );
      collect();
    } else if (event.type === "eval") {
      lastValidationLoss = event.evaluation.loss;
      console.log(
        `  eval @ ${event.evaluation.step}: val loss ${event.evaluation.loss.toFixed(4)} ` +
          `(ppl ${event.evaluation.perplexity.toFixed(2)}, uniform ${event.evaluation.uniformLoss.toFixed(3)})`,
      );
    }
    result = iterator.next();
  }
  const summary = result.value;
  const chunkMs = Date.now() - chunkT0;

  const checkpoint = summary.checkpoint;
  if (!checkpoint) throw new Error("training finished without a checkpoint");
  const serialized = JSON.stringify(checkpoint);
  writeFileSync(LATEST_CKPT, serialized);
  const complete = checkpoint.step >= TARGET_STEPS;
  if (complete) writeFileSync(FINAL_CKPT, serialized);

  const logLine = {
    at: new Date().toISOString(),
    chunk: chunkNumber,
    step: checkpoint.step,
    targetSteps: TARGET_STEPS,
    tokensSeen: checkpoint.tokensSeen,
    stepsThisChunk: summary.steps,
    learningRate,
    trainLoss: lastTrainLoss,
    validationLoss: lastValidationLoss ?? checkpoint.metrics.validationLoss,
    validationPerplexity: checkpoint.metrics.validationPerplexity,
    uniformLoss: checkpoint.metrics.uniformLoss,
    chunkMs,
    peakRssMb: peakRss,
    checkpointBytes: serialized.length,
    samplerPersisted: checkpoint.sampler !== undefined && checkpoint.sampler !== null,
    complete,
  };
  appendFileSync(LOG_PATH, JSON.stringify(logLine) + "\n");

  console.log("\n" + "=".repeat(74));
  console.log(
    `chunk ${chunkNumber} done: step ${checkpoint.step}/${TARGET_STEPS} ` +
      `| train ${lastTrainLoss?.toFixed(4)} | val ${logLine.validationLoss?.toFixed?.(4) ?? logLine.validationLoss} ` +
      `| ${Math.round(chunkMs / 1000)} s | peak rss ${peakRss} MB`,
  );
  console.log(`checkpoint:  ${LATEST_CKPT} (${serialized.length} bytes)`);
  console.log(
    `sampler state persisted: ${logLine.samplerPersisted ? "yes" : "NO — investigate"}`,
  );
  console.log(
    complete
      ? `COMPLETE — final checkpoint also at ${FINAL_CKPT}`
      : `next chunk: run the same command again (${Math.ceil((TARGET_STEPS - checkpoint.step) / STEPS_PER_CHUNK)} chunk(s) left)`,
  );
  console.log("=".repeat(74));
}

main();
