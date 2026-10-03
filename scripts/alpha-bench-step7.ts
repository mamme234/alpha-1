/**
 * Alpha — Step 7 feasibility bench.
 *
 *   bun scripts/alpha-bench-step7.ts
 *
 * Measures, on this machine, before any training run is committed to:
 *
 *   - parameter count from the config and from the model's actual tensors
 *     (two independent computations that must agree);
 *   - weight / checkpoint / training-memory estimates from real serialized
 *     bytes;
 *   - training throughput at the Step 7 shape (batch x seq x accumulation);
 *   - inference throughput and first-token latency at the 256-token context;
 *   - projected wall-clock for the step budgets under consideration.
 *
 * Nothing here is extrapolated from another machine: every rate printed is
 * measured here, now, by running the real loop.
 */

import { readFileSync } from "node:fs";
import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import { AlphaTransformer } from "../src/alpha/model/transformer";
import { AlphaTrainer, type TrainingConfig } from "../src/alpha/training/trainer";
import { countParameters, validateModelConfig } from "../src/alpha/model/config";
import { ALPHA_STEP7_PRESETS, validateStep7Presets } from "../src/alpha/model/presets-v2";
import { createDataset, datasetFingerprint } from "../src/alpha/datasets/types";
import { ALPHA_SEED_CORPUS } from "../src/alpha/datasets/seed-corpus";
import { estimateCheckpointBytes } from "../src/alpha/training/checkpoint";
import { AlphaInferenceEngine } from "../src/alpha/inference/engine";
import { parseServingArtifact } from "../src/alpha/serving/artifact";

const config = ALPHA_STEP7_PRESETS["micro-v2"];
const baseline = parseServingArtifact(
  JSON.parse(readFileSync("src/alpha/serving/step5-artifact.json", "utf8")),
);
const tokenizer = AlphaTokenizer.fromJSON(baseline.tokenizer.snapshot);

console.log("Alpha — Step 7 feasibility bench");
console.log(`  machine: ${process.platform}/${process.arch} · bun ${process.versions.bun ?? "unknown"}`);

// ---------------------------------------------------------------- config
validateModelConfig(config);
const validated = validateStep7Presets();
console.log("\nArchitecture");
for (const preset of validated) {
  console.log(
    `  ${preset.preset}: ${preset.parameterCount.toLocaleString()} parameters · fingerprint ${preset.fingerprint}`,
  );
}
console.log(
  `  baseline (Step 5): ${countParameters(baseline.model.config).toLocaleString()} parameters · context ${baseline.model.config.contextLength}`,
);
console.log(
  `  step 7 config: context ${config.contextLength} · dModel ${config.dModel} · heads ${config.nHeads} · layers ${config.nLayers} · ff ${config.dFeedForward} · vocab ${config.vocabSize}`,
);

// ------------------------------------------------------- tensors & memory
const model = new AlphaTransformer(config);
const tensorParams = model.parameters().reduce((sum, param) => sum + param.tensor.size, 0);
const configParams = countParameters(config);
console.log("\nIndependent parameter count");
console.log(`  from config formula : ${configParams.toLocaleString()}`);
console.log(`  from tensor shapes   : ${tensorParams.toLocaleString()}`);
console.log(`  agree                : ${tensorParams === configParams ? "yes" : "NO — MISMATCH"}`);

const weights = model.serializeWeights();
const weightBytes = estimateCheckpointBytes(weights);
const checkpointJsonBytes = Buffer.byteLength(JSON.stringify(weights));
console.log("\nMemory and artifacts");
console.log(`  weights (float32)    ${(weightBytes / 1024 / 1024).toFixed(2)} MiB`);
console.log(`  weights as base64    ${(checkpointJsonBytes / 1024 / 1024).toFixed(2)} MiB`);
console.log(
  `  training (w+g+adam)  ${(((weightBytes * 4) / 1024 / 1024)).toFixed(2)} MiB in float32 (weights + gradients + 2x Adam moments)`,
);

// ------------------------------------------------------------- training
const TRAIN_STEPS = 12;
const BATCH = 8;
const SEQ = config.contextLength;
const ACCUM = 1;
const trainingConfig: TrainingConfig = {
  batchSize: BATCH,
  seqLen: SEQ,
  totalSteps: TRAIN_STEPS,
  learningRate: 0.002,
  schedule: "cosine",
  warmupSteps: 4,
  minFactor: 0.1,
  weightDecay: 0.01,
  gradClipNorm: 1,
  evalInterval: 0,
  evalBatches: 2,
  validationFraction: 0.12,
  seed: 777,
  checkpointInterval: 0,
  batchMode: "windows",
  gradientAccumulationSteps: ACCUM,
  earlyStopping: null,
};
const dataset = createDataset({
  name: "step7-bench",
  version: "0.0.0",
  description: "Throughput measurement only (seed corpus, never used for claims).",
  license: "Alpha-owned",
  source: "bench",
  documents: ALPHA_SEED_CORPUS.documents,
});

const heapBefore = process.memoryUsage().heapUsed;
const wallStart = Date.now();
const trainer = new AlphaTrainer({ model, tokenizer, dataset, config: trainingConfig, runId: "step7-bench", checkpointLabel: "step7-bench" });
const summary = trainer.trainToCompletion();
const wallMs = Date.now() - wallStart;
const heapAfter = process.memoryUsage().heapUsed;

const tokensPerStep = BATCH * SEQ * ACCUM;
console.log("\nTraining throughput (measured)");
console.log(
  `  ${TRAIN_STEPS} steps · batch ${BATCH} x seq ${SEQ} x accum ${ACCUM} = ${tokensPerStep.toLocaleString()} tokens/step`,
);
console.log(
  `  ${summary.tokensSeen.toLocaleString()} tokens in ${wallMs} ms → ${(summary.tokensSeen / (wallMs / 1000)).toFixed(1)} tokens/step-sec, reported ${summary.throughputTokensPerSecond} tok/s`,
);
console.log(`  loss ${summary.firstLoss?.toFixed(4)} → ${summary.lastLoss?.toFixed(4)} · heap grew ${((heapAfter - heapBefore) / 1024 / 1024).toFixed(1)} MiB`);

const measuredTokensPerSecond = summary.tokensSeen / (wallMs / 1000);
console.log("\nProjected training wall-clock at this throughput");
for (const steps of [40, 64, 96]) {
  const tokens = steps * tokensPerStep;
  console.log(
    `  ${steps} steps = ${(tokens / 1000).toFixed(0)}K tokens ≈ ${(tokens / measuredTokensPerSecond / 60).toFixed(1)} min`,
  );
}

// ------------------------------------------------------------- inference
const engine = new AlphaInferenceEngine({ model, tokenizer, stage: "architecture" });
const prompt = "Alpha is a self-owned AI system. It trains its own model from its own corpus, and it reports what it measured. ";
const inferStart = Date.now();
const generated = engine.generate(prompt, { temperature: 0, maxNewTokens: 48, topK: 40, topP: 0.95, deterministic: true, seed: 777 });
const inferMs = Date.now() - inferStart;
console.log("\nInference throughput (measured)");
console.log(
  `  48 tokens from a ${(prompt.length > 0 ? tokenizer.encode(prompt).length : 0)}-token prompt in ${inferMs} ms → ${(generated.tokenIds.length / (inferMs / 1000)).toFixed(1)} tok/s`,
);

const longPrompt = (prompt.repeat(8)).slice(0, 1200);
const longIds = tokenizer.encode(longPrompt).slice(0, config.contextLength - 8);
const longStart = Date.now();
const engine2 = new AlphaInferenceEngine({ model, tokenizer, stage: "architecture", maxContextTokens: config.contextLength });
const longGen = engine2.generate(tokenizer.decode(longIds), { temperature: 0, maxNewTokens: 16, deterministic: true, seed: 777 });
console.log(
  `  16 tokens from a ${longIds.length}-token context in ${Date.now() - longStart} ms → stop ${longGen.stopReason}`,
);

console.log("\nNext: train the candidate with alpha:train-step7, then run alpha:verify-intelligence.");
console.log(`  dataset fingerprint for the bench corpus: ${datasetFingerprint(dataset)}`);
process.exit(0);
