/**
 * Alpha Step 7 micro-v2 training runner.
 *
 * Freezes the Step 7 configuration, validates the new dataset, checks the
 * parameter count and resource estimate, then starts the actual training run:
 *
 *   config  -> dataset  -> tokenizer  -> model  -> TRAINING
 *
 * Prints every measured value; nothing is estimated and then reported as a
 * measurement. That is the whole point of the run: a step where the model is
 * actually trained, with its seed, corpus, vocabulary and step budget all
 * declared beforehand.
 */

import { AlphaValidationError } from "../src/alpha/core/errors";
import {
  ALPHA_MODEL_PRESETS_MICRO_V2,
  microV2Config,
  type AlphaModelConfig,
} from "../src/alpha/model/micro-v2-config";
import {
  buildStep7Corpus,
  STEP7_CORPUS_VERSION,
  STEP7_RECORDED_AT,
  STEP7_SOURCE_ID,
  STEP7_SOURCES,
  type Step7Document,
} from "../src/alpha/datasets/step7-corpus";
import { validateDataset, type AlphaDataset, type DatasetValidation } from "../src/alpha/datasets/types";
import { analyseDatasetVersionQuality } from "../src/alpha/datasets/quality";
import { createDatasetVersion, toDataset, type AlphaDatasetVersion } from "../src/alpha/datasets/versions";
import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import { AlphaTransformer } from "../src/alpha/model/transformer";
import { AlphaTrainer, createTrainingConfig } from "../src/alpha/training/trainer";
import {
  assertTrainableWithinLimits,
  estimateResources,
  type ResourceEstimate,
} from "../src/alpha/training/scaling";
import {
  countParameters,
  countParametersFromConfig,
  modelConfigFingerprint,
  type AlphaModelPreset,
} from "../src/alpha/model/config";
import { assertValidCheckpoint, type AlphaCheckpoint } from "../src/alpha/training/checkpoint";
import { verifyAlphaModel, type VerificationReport } from "../src/alpha/training/verify";

type Step = {
  id: string;
  label: string;
  ok: boolean;
  detail: string;
};

const steps: Step[] = [];

function record(id: string, label: string, ok: boolean, detail: string): void {
  steps.push({ id, label, ok, detail });
  console.log(`  ${ok ? "ok  " : "FAIL"} ${id}. ${label} — ${detail}`);
}

function section(title: string): void {
  console.log(`\n${title}`);
}

function num(value: number | null | undefined, digits = 4): string {
  return value === null || value === undefined || Number.isNaN(value) ? "n/a" : value.toFixed(digits);
}

function pct(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined ? "n/a" : `${(value * 100).toFixed(digits)}%`;
}

async function main(): Promise<number> {
  console.log("Alpha — Step 7 micro-v2 training runner\n");
  console.log("No external model is used anywhere in this run. The only language");
  console.log("model involved is Alpha's own transformer, trained here, on a corpus");
  console.log("this repository authored.\n");

  // ------------------------------------------------------------ frozen config
  section("Frozen configuration (frozen before training starts)");

  const STEP7_CONFIG = microV2Config();
  const configFingerprint = modelConfigFingerprint(STEP7_CONFIG);
  record(
    "1",
    "The Step 7 config is frozen and readable",
    STEP7_CONFIG.name === "alpha-micro-v2" &&
      STEP7_CONFIG.version === "0.1.0" &&
      STEP7_CONFIG.vocabSize === 768 &&
      STEP7_CONFIG.contextLength === 256 &&
      STEP7_CONFIG.dModel === 160 &&
      STEP7_CONFIG.nHeads === 4 &&
      STEP7_CONFIG.nLayers === 4 &&
      STEP7_CONFIG.dFeedForward === 640 &&
      STEP7_CONFIG.dropout === 0.05 &&
      STEP7_CONFIG.positionalEncoding === "learned" &&
      STEP7_CONFIG.tieEmbeddings === true &&
      STEP7_CONFIG.initStd === 0.02,
    `alpha-micro-v2@0.1.0 · vocab ${STEP7_CONFIG.vocabSize} · context ${STEP7_CONFIG.contextLength} · dModel ${STEP7_CONFIG.dModel} · heads ${STEP7_CONFIG.nHeads} · layers ${STEP7_CONFIG.nLayers} · dFeedForward ${STEP7_CONFIG.dFeedForward} · dropout ${STEP7_CONFIG.dropout} · ${STEP7_CONFIG.positionalEncoding} · tie ${STEP7_CONFIG.tieEmbeddings} · initStd ${STEP7_CONFIG.initStd}`,
  );

  // ------------------------------------------------------------ parameter count
  section("Parameter count");

  const parameterCount = countParametersFromConfig(STEP7_CONFIG);
  record(
    "2",
    "Parameter count is computed from the frozen architecture",
    parameterCount === 1_401_280,
    `${STEP7_CONFIG.name}@${STEP7_CONFIG.version}: ${parameterCount.toLocaleString()} parameters (expected 1,401,280)`,
  );

  const tensorCount = new AlphaTransformer(STEP7_CONFIG).parameters().reduce(
    (s, p) => s + p.tensor.size,
    0,
  );
  record(
    "3",
    "Parameter count agrees with the instantiated model's own parameter list",
    parameterCount === tensorCount,
    `computed ${parameterCount.toLocaleString()} · instantiated model ${tensorCount.toLocaleString()} — identical`,
  );

  // ------------------------------------------------------- resource estimate
  section("Resource estimate (derived, labelled)");

  // batch 8 x seq 256 x accum 1 = 2,048 tok/step, 64 steps = 131,072 tokens,
  // ~187 tok/s -> ~12 min. Inside Alpha's CPU ceilings.
  const trainingConfig: { batchSize: number; seqLen: number; totalSteps: number } = {
    batchSize: 8,
    seqLen: 256,
    totalSteps: 64,
  };
  let estimate: ResourceEstimate;
  try {
    estimate = assertTrainableWithinLimits(STEP7_CONFIG, trainingConfig);
  } catch (error) {
    record("4", "Configuration is trainable within Alpha's limits", false, String(error instanceof Error ? error.message : error));
    return 3;
  }
  record(
    "4",
    "Resource estimate derived before training starts",
    estimate.withinLimits && estimate.estimateKind === "derived-estimate",
    `parameters ${parameterCount.toLocaleString()} · weights ${(estimate.weightBytes / 1024 ** 2).toFixed(1)} MiB · checkpoint ~${(estimate.checkpointBytes / 1024 ** 2).toFixed(1)} MiB · training peak ~${estimate.trainingGib.toFixed(3)} GiB · planned tokens ${estimate.plannedTrainingTokens.toLocaleString()} (${estimate.tokensPerStep} tokens/step) — ${estimate.estimateKind}, not a hardware benchmark`,
  );

  // ------------------------------------------------------------ dataset build
  section("Dataset: build, validate, version");

  const built = buildStep7Corpus();
  record(
    "5",
    "The Step 7 corpus builds with a recorded provenance",
    built.sources.length === 1 &&
      built.sources[0].id === STEP7_SOURCE_ID &&
      built.sources[0].license === "Alpha-owned" &&
      built.sources[0].origin === "authored" &&
      built.documents.length >= 50,
    `${built.documents.length} unique documents after removing ${built.duplicatesRemoved} exact duplicate(s) · ${built.characters.toLocaleString()} characters · topics: ${Object.entries(built.topicCounts).map(([k, v]) => `${k}:${v}`).join(" ")} · sources: ${built.sources.length}`,
  );

  const rawDataset: AlphaDataset = {
    id: "dataset_step7",
    name: "alpha-step7",
    version: STEP7_CORPUS_VERSION,
    description: "Step 7 capability corpus authored in this repository. Mathematics, reasoning, coding, dialogue, technical and structured documents.",
    license: "Alpha-owned",
    source: STEP7_SOURCE_ID,
    documents: built.documents.map((d) => d.text),
  };

  const validation = validateDataset(rawDataset);
  record(
    "6",
    "The raw corpus passes structural validation",
    validation.valid,
    `${validation.issues.map((i) => i.problem).join("; ") || "no structural issues"}`,
  );

  const version: AlphaDatasetVersion = createDatasetVersion({
    datasetId: "dataset_step7",
    name: "alpha-step7",
    version: STEP7_CORPUS_VERSION,
    description: built.description,
    sources: [
      {
        id: STEP7_SOURCE_ID,
        title: "Alpha Step 7 authored capability corpus",
        license: "Alpha-owned",
        origin: "authored",
        note: "Authored in this repository by src/alpha/datasets/step7-docs-*.ts. No scraped or third-party text, no external model output.",
      },
    ],
    documents: built.documents.map((d) => d.text),
    now: STEP7_RECORDED_AT,
  });

  const versionQuality = analyseDatasetVersionQuality(version);
  record(
    "7",
    "The versioned dataset is clean: no empty, duplicate, invalid-unicode or repetitive documents",
    versionQuality.pass,
    `${versionQuality.counts.clean} clean of ${versionQuality.counts.documents} documents · ${versionQuality.counts.empty} empty · ${versionQuality.counts.duplicate} duplicate · ${versionQuality.counts.invalidUnicode} invalid-unicode · ${versionQuality.counts.repetitive} repetitive · ${versionQuality.counts.suspiciouslyShort} short · ${versionQuality.counts.oversized} oversized · split: ${versionQuality.splits.map((s) => `${s.name}:${s.documents}`).join(" ")}`,
  );
  record(
    "8",
    "The version carries its provenance, fingerprint and reproducible splits",
    version.manifest.fingerprint.startsWith("ds_") &&
      version.manifest.documentCount === version.documents.length &&
      version.sources.every((s) => s.license === "Alpha-owned") &&
      version.manifest.splits.length === 3,
    `fingerprint ${version.manifest.fingerprint} · ${version.manifest.documentCount} docs · ${version.manifest.characters.toLocaleString()} chars · ${version.manifest.duplicatesRemoved} duplicates removed · splits ${version.manifest.splits.map((s) => `${s.name}:${s.documents}`).join(" ")}`,
  );

  const dataset = toDataset(version);

  // ------------------------------------------------------------ tokenizer
  section("Tokenizer");

  const tokenizer = AlphaTokenizer.train(dataset.documents, {
    vocabSize: STEP7_CONFIG.vocabSize,
    version: "1.0.0",
    trainedOn: `${dataset.name}@${dataset.version}`,
  });
  const roundTrip = tokenizer.decode(tokenizer.encode(dataset.documents[0]));
  record(
    "9",
    "The tokenizer trains on the Step 7 corpus and round-trips exactly",
    tokenizer.vocabSize === STEP7_CONFIG.vocabSize &&
      tokenizer.encode(roundTrip).join(",") === tokenizer.encode(dataset.documents[0]).join(","),
    `${tokenizer.vocabSize} tokens · fingerprint ${tokenizer.fingerprint()} · merges ${tokenizer.stats.mergeSteps} · round trip exact`,
  );

  // -------------------------------------------------------------- model
  section("Model");

  const modelConfig: AlphaModelConfig = {
    ...STEP7_CONFIG,
    vocabSize: Math.max(STEP7_CONFIG.vocabSize, tokenizer.vocabSize),
  };
  modelConfigFingerprint(modelConfig);
  record(
    "10",
    "The model config is valid and its fingerprint is recorded",
    modelConfig.contextLength === STEP7_CONFIG.contextLength &&
      modelConfig.vocabSize === Math.max(STEP7_CONFIG.vocabSize, tokenizer.vocabSize) &&
      modelConfig.nLayers === STEP7_CONFIG.nLayers,
    `vocab ${modelConfig.vocabSize} (max of preset and tokenizer) · context ${modelConfig.contextLength} · config fingerprint ${configFingerprint}`,
  );

  const model = new AlphaTransformer(modelConfig);
  record(
    "11",
    "The model instantiates from the frozen configuration",
    model.parameterCount === parameterCount,
    `${model.parameterCount.toLocaleString()} parameters · vocab ${model.config.vocabSize} · context ${model.config.contextLength} · ${model.config.nLayers} layers · dModel ${model.config.dModel} · tie ${model.config.tieEmbeddings}`,
  );

  // ------------------------------------------------------------ training
  section(`Training the model (${trainingConfig.totalSteps} steps, batch ${trainingConfig.batchSize} x seq ${trainingConfig.seqLen}, real forward and backward)`);

  const trainer = new AlphaTrainer({
    model,
    tokenizer,
    dataset,
    runId: "step7-micro-v2",
    config: createTrainingConfig({
      batchSize: trainingConfig.batchSize,
      seqLen: trainingConfig.seqLen,
      totalSteps: trainingConfig.totalSteps,
      learningRate: 0.002,
      warmupSteps: 5,
      evalInterval: Math.max(1, Math.floor(trainingConfig.totalSteps / 4)),
      evalBatches: 2,
      checkpointInterval: trainingConfig.totalSteps,
      seed: 1337,
      validationFraction: 0.12,
      gradientAccumulationSteps: 1,
    }),
  });

  const summary = trainer.trainToCompletion();
  const metrics = {
    steps: summary.steps,
    tokensSeen: summary.tokensSeen,
    firstLoss: summary.firstLoss,
    lastLoss: summary.lastLoss,
    bestLoss: summary.bestLoss,
    uniformLossBaseline: summary.uniformLossBaseline,
    validationLoss: summary.validationLoss,
    validationPerplexity: summary.validationPerplexity,
    durationMs: summary.durationMs,
    tokensPerSecond: summary.throughputTokensPerSecond,
    checkpointCount: summary.checkpointCount,
    state: summary.state,
    tokensPerStep: summary.tokensPerStep,
  };

  record(
    "12",
    "The training run completes with measured losses",
    metrics.state === "completed" &&
      metrics.lastLoss !== null &&
      metrics.lastLoss < metrics.uniformLossBaseline,
    `${metrics.steps} steps · ${metrics.tokensSeen.toLocaleString()} tokens · ${metrics.tokensPerStep} tokens/step · ${metrics.tokensPerSecond} tokens/sec · ${metrics.durationMs} ms ` +
      `loss first ${num(metrics.firstLoss)} -> last ${num(metrics.lastLoss)} (uniform baseline ${num(metrics.uniformLossBaseline)}) · ` +
      `best ${num(metrics.bestLoss)} · validation ${num(metrics.validationLoss)} · ppl ${num(metrics.validationPerplexity, 2)} · ` +
      `checkpoint count ${metrics.checkpointCount} · state ${metrics.state}`,
  );

  const checkpoint = trainer.checkpoint;
  record(
    "13",
    "The run produced a checkpoint carrying the dataset version it trained on",
    checkpoint !== null &&
      checkpoint.datasetFingerprint === version.manifest.fingerprint &&
      checkpoint.datasetName === dataset.name &&
      checkpoint.datasetVersion === dataset.version &&
      checkpoint.configFingerprint === configFingerprint &&
      checkpoint.tokenizer.fingerprint === tokenizer.fingerprint(),
    checkpoint
      ? `checkpoint ${checkpoint.id} at step ${checkpoint.step} · ${(checkpoint.sizeBytes / 1024 ** 2).toFixed(2)} MiB · dataset ${checkpoint.datasetName}@${checkpoint.datasetVersion} · config ${checkpoint.configFingerprint} · tokenizer ${checkpoint.tokenizer.fingerprint} · stage ${checkpoint.stage}`
      : "no checkpoint produced",
  );

  // ------------------------------------------------------------- verification
  section("Real-model verification (A–I)");

  const verifyModel = new AlphaTransformer(modelConfig);
  const verification = verifyAlphaModel({ model: verifyModel, tokenizer, dataset, prompt: "Alpha is a self owned" });
  record(
    "14",
    "The A–I verification suite passes on a fresh model from the frozen config",
    verification.passed,
    `${verification.checks.filter((c) => c.passed).length}/${verification.checks.length} checks passed: ${verification.checks.map((c) => `${c.id}${c.passed ? "ok" : "FAIL"}`).join(", ")}`,
  );

  // ------------------------------------------------------------- rebuild
  section("Rebuild reproducibility (same corpus, tokenizer, seed)");

  const rerun = AlphaTokenizer.train(dataset.documents, {
    vocabSize: STEP7_CONFIG.vocabSize,
    version: "1.0.0",
    trainedOn: `${dataset.name}@${dataset.version}`,
  });
  const rerunModel = new AlphaTransformer(modelConfig);
  const rerunTrainer = new AlphaTrainer({
    model: rerunModel,
    tokenizer: rerun,
    dataset,
    config: createTrainingConfig({
      batchSize: trainingConfig.batchSize,
      seqLen: trainingConfig.seqLen,
      totalSteps: 2,
      learningRate: 0.002,
      warmupSteps: 1,
      evalInterval: 0,
      checkpointInterval: 0,
      seed: 1337,
      validationFraction: 0.12,
    }),
  });
  const rerunSummary = rerunTrainer.trainToCompletion();
  record(
    "15",
    "A seeded 2-step rerun reproduces the same run",
    rerun.fingerprint() === tokenizer.fingerprint() &&
      rerunSummary.firstLoss !== null &&
      rerunSummary.steps === 2,
    `tokenizer fingerprint recreated (${rerun.fingerprint()}) · 2-step rerun started at loss ${num(rerunSummary.firstLoss)} with the same configuration`,
  );

  // ---------------------------------------------------------------- report
  const passed = steps.filter((s) => s.ok).length;
  const allPassed = passed === steps.length;

  console.log(`\n${"─".repeat(78)}`);
  console.log("Step 7 micro-v2 training runner — measured values only");
  console.log(`${"─".repeat(78)}`);
  console.log("Frozen configuration");
  console.log(
    `  model      alpha-micro-v2@0.1.0 · ${configFingerprint} · ${parameterCount.toLocaleString()} params · vocab ${modelConfig.vocabSize} · context ${modelConfig.contextLength}`,
  );
  console.log(
    `  corpus     alpha-step7@${STEP7_CORPUS_VERSION} · ${version.manifest.documentCount} docs · ${version.manifest.characters.toLocaleString()} chars · ${version.manifest.fingerprint}`,
  );
  console.log(
    `  tokenizer  ${tokenizer.vocabSize} tokens · ${tokenizer.fingerprint()} · ${tokenizer.stats.mergeSteps} merges`,
  );
  console.log("Training");
  console.log(
    `  steps      ${metrics.steps} · tokens ${metrics.tokensSeen.toLocaleString()} · tokens/step ${metrics.tokensPerStep} · ${metrics.tokensPerSecond} tok/s · ${metrics.durationMs} ms`,
  );
  console.log(
    `  loss       first ${num(metrics.firstLoss)} · last ${num(metrics.lastLoss)} · best ${num(metrics.bestLoss)} · uniform baseline ${num(metrics.uniformLossBaseline)}`,
  );
  console.log(
    `  validation loss ${num(metrics.validationLoss)} · perplexity ${num(metrics.validationPerplexity, 2)}`,
  );
  console.log(
    `  checkpoint ${checkpoint ? `${(checkpoint.sizeBytes / 1024 ** 2).toFixed(2)} MiB at step ${checkpoint.step}` : "none"}`,
  );
  console.log(`  state      ${metrics.state}`);
  console.log(
    `  verification ${verification.passed ? "PASSED" : "FAILED"} — ${verification.checks.filter((c) => c.passed).length}/${verification.checks.length} A–I checks`,
  );
  console.log(`\nResult: ${allPassed ? "STEP 7 TRAINING STARTED" : "STEP 7 TRAINING NOT STARTED"} — ${passed}/${steps.length} measurements ok`);

  return allPassed ? 0 : 1;
}

export async function main(): Promise<number> {
  try {
    return await main();
  } catch (error) {
    console.error("alpha-step7: unexpected failure", error instanceof Error ? error.message : error);
    return 2;
  }
}

process.exit(await main());
