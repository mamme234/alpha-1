/**
 * Alpha — training budget benchmark.
 *
 * Runs a small, honest measurement so the training budget is based on this
 * host's real speed and memory, not a guess. It writes NO files.
 *
 * The first version of this script was OOM-killed (1.6 GB anon RSS) at the
 * Step 7 configuration. That is why this version:
 *   - pulls the trainer's generator one step at a time,
 *   - forces a synchronous GC between steps (Bun.gc(true)), and
 *   - prints RSS / heap after every step.
 *
 * Env knobs:
 *   BENCH_BATCH   rows per step            (default 4)
 *   BENCH_SEQ     tokens per row           (default 256)
 *   BENCH_STEPS   optimiser steps to run   (default 2)
 *   BENCH_EVAL    validation batches       (default 1)
 */

import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import { AlphaTransformer } from "../src/alpha/model/transformer";
import { AlphaTrainer } from "../src/alpha/training/trainer";
import { microV2Config } from "../src/alpha/model/micro-v2-config";
import { buildStep7Corpus } from "../src/alpha/datasets/step7-corpus";

const BATCH = Number(process.env.BENCH_BATCH ?? "4");
const SEQ = Number(process.env.BENCH_SEQ ?? "256");
const STEPS = Number(process.env.BENCH_STEPS ?? "2");
const EVAL_BATCHES = Number(process.env.BENCH_EVAL ?? "1");

type BunLike = { gc?: (force?: boolean) => void };
const bun = (globalThis as { Bun?: BunLike }).Bun;

function rssMb(): number {
  return Number((process.memoryUsage().rss / 1048576).toFixed(1));
}
function heapMb(): number {
  return Number((process.memoryUsage().heapUsed / 1048576).toFixed(1));
}
function collect(label: string): void {
  const before = rssMb();
  bun?.gc?.(true);
  console.log(
    `  [gc] ${label}: rss ${before} -> ${rssMb()} MB, heap ${heapMb()} MB`,
  );
}

const t0 = Date.now();
const built = buildStep7Corpus();
const dataset = {
  id: "alpha-budget-bench",
  name: "alpha-budget-bench",
  version: "1.0.0",
  description: "benchmark corpus (step7)",
  license: "Alpha-owned",
  source: "alpha-step7-corpus",
  documents: built.documents.map((d) => d.text),
};
const tokenizer = AlphaTokenizer.train(dataset.documents, {
  vocabSize: 768,
  version: "1.0.0",
  trainedOn: "alpha-budget-bench",
});
console.log(`setup: tokenizer trained in ${Date.now() - t0} ms`);

const model = new AlphaTransformer(microV2Config());
console.log(
  `setup: model ${model.config.name}@${model.config.version} ` +
    `params=${model.parameterCount} rss=${rssMb()} MB`,
);

const trainer = new AlphaTrainer({
  model,
  tokenizer,
  dataset,
  config: {
    batchSize: BATCH,
    seqLen: SEQ,
    totalSteps: STEPS,
    learningRate: 0.002,
    warmupSteps: 1,
    evalInterval: 0,
    evalBatches: EVAL_BATCHES,
    validationFraction: 0.12,
    seed: 1337,
    checkpointInterval: 0,
    gradientAccumulationSteps: 1,
  },
  checkpointLabel: "alpha-budget-bench",
  runId: "alpha-budget-bench",
});
console.log(
  `setup: trainer ready (corpus tokens ${trainer.corpus.stats.totalTokens}, ` +
    `train ${trainer.corpus.stats.trainTokens}, val ${trainer.corpus.stats.validationTokens}) ` +
    `rss=${rssMb()} MB`,
);
collect("after setup");

console.log(
  `\nbenchmark: batch=${BATCH} seq=${SEQ} steps=${STEPS} tokensPerStep=${BATCH * SEQ}`,
);

const iterator = trainer.run();
const stepMs: number[] = [];
let peakRss = rssMb();
let result = iterator.next();
while (!result.done) {
  const event = result.value;
  if (event.type === "step") {
    const rss = rssMb();
    peakRss = Math.max(peakRss, rss);
    stepMs.push(event.point.elapsedMs);
    console.log(
      `  step ${event.point.step}: loss ${event.point.loss.toFixed(4)} ` +
        `elapsed ${event.point.elapsedMs} ms rss ${rss} MB heap ${heapMb()} MB`,
    );
    bun?.gc?.(true);
    const after = rssMb();
    peakRss = Math.max(peakRss, after);
    console.log(`    post-gc rss ${after} MB`);
  }
  result = iterator.next();
}

const summary = result.value;
const deltas: number[] = [];
for (let i = 1; i < stepMs.length; i++) deltas.push(stepMs[i] - stepMs[i - 1]);
deltas.push(stepMs[0]);
const medianMsPerStep = deltas.slice().sort((a, b) => a - b)[Math.floor(deltas.length / 2)];
const tokensPerSecond = medianMsPerStep > 0 ? (BATCH * SEQ) / (medianMsPerStep / 1000) : 0;

console.log(
  JSON.stringify(
    {
      batch: BATCH,
      seq: SEQ,
      steps: summary.steps,
      tokensPerStep: BATCH * SEQ,
      stepElapsedMs: stepMs,
      medianMsPerStep: Number(medianMsPerStep.toFixed(1)),
      tokensPerSecond: Math.round(tokensPerSecond),
      peakRssMb: peakRss,
      finalRssMb: rssMb(),
      trainLossLast: summary.lastLoss,
      validationLoss: summary.validationLoss,
      validationPerplexity: summary.validationPerplexity,
      totalWallMs: Date.now() - t0,
    },
    null,
    1,
  ),
);
