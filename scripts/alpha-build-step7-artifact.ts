/**
 * Step 7 — build the serving artifact from the verified checkpoint.
 *
 *   bun scripts/alpha-build-step7-artifact.ts
 *
 * Reads the frozen Step 7 checkpoint (step 64, cfg_dc35f555, ckpt_mutfvpv45sqe7),
 * rebuilds the model and the tokenizer from the checkpoint alone, and writes
 * `src/alpha/serving/step7-artifact.json` in the serving-artifact format the
 * chat runtime loads.
 *
 * Every fingerprint inside the artifact is recomputed by `createServingArtifact`
 * and re-checked by `loadServingArtifact` after writing, so a file whose config
 * or weights were edited cannot load. The evaluation block carries the exact
 * numbers the frozen Step 7C run measured on suite evl_b58eacf7 — including the
 * gate verdict, which is recorded as it was: not met.
 *
 * Nothing here trains, and nothing edits a frozen file.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import {
  createServingArtifact,
  describeServingArtifact,
  loadServingArtifact,
  parseServingArtifact,
} from "../src/alpha/serving/artifact";
import { createServingRuntime } from "../src/alpha/serving/runtime";
import { AlphaTransformer } from "../src/alpha/model/transformer";
import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import { assertValidCheckpoint, parseCheckpoint } from "../src/alpha/training/checkpoint";
import { buildStep7Corpus } from "../src/alpha/datasets/step7-corpus";

const ROOT = join(__dirname, "..");
const CHECKPOINT_PATH = join(ROOT, "src/alpha/experiments", "step7-micro-v2-chunk-32.alpha-ckpt.json");
const REPORT_JSON = join(ROOT, "src/alpha/experiments", "step7c-candidate-main.json");
const REPORT_TXT = join(ROOT, "src/alpha/experiments", "step7c-report.txt");
const OUT_PATH = join(ROOT, "src/alpha/serving", "step7-artifact.json");
const FROZEN_SUITE = "evl_b58eacf7";

function fail(problem: string): never {
  console.error(`\nSTEP 7 ARTIFACT BUILD FAILED: ${problem}`);
  process.exit(1);
}

/* ------------------------------------------------------------------ */
/* 1. The checkpoint, validated against its own metadata               */
/* ------------------------------------------------------------------ */

if (!existsSync(CHECKPOINT_PATH)) fail(`checkpoint not found at ${CHECKPOINT_PATH}`);
const checkpoint = parseCheckpoint(readFileSync(CHECKPOINT_PATH, "utf8"));
assertValidCheckpoint(checkpoint);
if (!checkpoint.tokenizer.snapshot) fail("the checkpoint carries no tokenizer snapshot");
if (checkpoint.configFingerprint !== "cfg_dc35f555") {
  fail(`checkpoint config fingerprint is ${checkpoint.configFingerprint}, expected the frozen cfg_dc35f555`);
}

/* ------------------------------------------------------------------ */
/* 2. Rebuild the model exactly as the evaluator did                   */
/* ------------------------------------------------------------------ */

const model = new AlphaTransformer(checkpoint.config);
model.loadWeights(checkpoint.weights);
const tokenizer = AlphaTokenizer.fromJSON(checkpoint.tokenizer.snapshot);
if (tokenizer.fingerprint() !== checkpoint.tokenizer.fingerprint) {
  fail(
    `tokenizer snapshot hashes to ${tokenizer.fingerprint()}, the checkpoint records ${checkpoint.tokenizer.fingerprint}`,
  );
}

/* ------------------------------------------------------------------ */
/* 3. The corpus reference (deterministic rebuild, for counts only)    */
/* ------------------------------------------------------------------ */

const built = buildStep7Corpus();

/* ------------------------------------------------------------------ */
/* 4. The measured Step 7C facts                                       */
/* ------------------------------------------------------------------ */

if (!existsSync(REPORT_JSON) || !existsSync(REPORT_TXT)) {
  fail("the Step 7C evaluation outputs are missing; run the evaluation before building the artifact");
}
const evaluationRun = JSON.parse(readFileSync(REPORT_JSON, "utf8")) as {
  report: {
    suite: { fingerprint: string };
    cases: unknown[];
    languageModeling: { loss: number; perplexity: number; nextTokenTop1Accuracy: number };
  };
  meta: { heldOutDocuments: number; suiteMatchesRecorded: boolean; checkpoint: { id: string } };
};
const reportText = readFileSync(REPORT_TXT, "utf8");
const gateFingerprint = /^\[gate\] (gate_[0-9a-f]+)$/m.exec(reportText)?.[1];
const verdictLine = /^\[verdict\] (.*)$/m.exec(reportText)?.[1] ?? "";
if (!gateFingerprint) fail("the Step 7C report has no gate fingerprint");
if (!evaluationRun.meta.suiteMatchesRecorded) {
  fail("the Step 7C run did not reproduce the recorded frozen suite; refusing to write its numbers into the artifact");
}
if (evaluationRun.report.suite.fingerprint !== FROZEN_SUITE) {
  fail(`evaluation ran on suite ${evaluationRun.report.suite.fingerprint}, expected ${FROZEN_SUITE}`);
}
if (evaluationRun.meta.checkpoint.id !== checkpoint.id) {
  fail(`the evaluation ran checkpoint ${evaluationRun.meta.checkpoint.id}, not ${checkpoint.id}`);
}

/* ------------------------------------------------------------------ */
/* 5. The artifact                                                    */
/* ------------------------------------------------------------------ */

const artifact = createServingArtifact({
  model,
  tokenizer,
  stage: checkpoint.stage,
  training: {
    steps: checkpoint.step,
    tokensSeen: checkpoint.tokensSeen,
    tokensPerStep: checkpoint.trainingConfig.batchSize * checkpoint.trainingConfig.seqLen,
    seed: checkpoint.seed,
    // The chunked runner did not record a first-step loss; a firstLoss of null
    // says "not measured", which is true, instead of inventing a number.
    firstLoss: null,
    lastLoss: checkpoint.metrics.trainLoss,
    validationLoss: checkpoint.metrics.validationLoss,
    validationPerplexity: checkpoint.metrics.validationPerplexity,
    uniformLossBaseline: checkpoint.metrics.uniformLoss ?? Math.log(checkpoint.config.vocabSize),
    // The 64 steps were trained across several rebooted chunk processes; no
    // aggregate wall-clock was recorded, so it is stored as 0, not estimated.
    durationMs: 0,
    tokensPerSecond: 0,
    checkpointId: checkpoint.id,
    gradientAccumulationSteps: checkpoint.trainingConfig.gradientAccumulationSteps,
    config: checkpoint.trainingConfig,
  },
  data: {
    datasetName: checkpoint.datasetName,
    datasetVersion: checkpoint.datasetVersion,
    datasetFingerprint: checkpoint.datasetFingerprint,
    mixtureFingerprint: null,
    documents: built.documents.length,
    characters: built.characters,
  },
  evaluation: {
    suiteFingerprint: evaluationRun.report.suite.fingerprint,
    suiteCases: evaluationRun.report.cases.length,
    heldOutDocuments: evaluationRun.meta.heldOutDocuments,
    loss: evaluationRun.report.languageModeling.loss,
    perplexity: evaluationRun.report.languageModeling.perplexity,
    nextTokenTop1Accuracy: evaluationRun.report.languageModeling.nextTokenTop1Accuracy,
    gateFingerprint,
    gatePassed: verdictLine.startsWith("passed"),
  },
});

const serialised = JSON.stringify(artifact);
writeFileSync(OUT_PATH, serialised);

/* ------------------------------------------------------------------ */
/* 6. Proof: the written file loads in the serving path                */
/* ------------------------------------------------------------------ */

const reparsed = parseServingArtifact(JSON.parse(readFileSync(OUT_PATH, "utf8")));
const loaded = loadServingArtifact(reparsed);
const runtime = createServingRuntime({ artifact: reparsed, logLevel: "warn" });

console.log("Step 7 serving artifact built and verified");
console.log(`  file            ${OUT_PATH} (${serialised.length.toLocaleString()} bytes)`);
console.log(`  ${describeServingArtifact(reparsed)}`);
console.log(
  `  training        ${reparsed.training.steps} steps · ${reparsed.training.tokensSeen.toLocaleString()} tokens · train loss ${reparsed.training.lastLoss} · checkpoint ${reparsed.training.checkpointId}`,
);
console.log(
  `  evaluation      suite ${reparsed.evaluation.suiteFingerprint} (${reparsed.evaluation.suiteCases} cases, ${reparsed.evaluation.heldOutDocuments} held-out docs) · loss ${reparsed.evaluation.loss} · ppl ${reparsed.evaluation.perplexity} · top1 ${reparsed.evaluation.nextTokenTop1Accuracy}`,
);
console.log(
  `  gate            ${reparsed.evaluation.gateFingerprint} ${reparsed.evaluation.gatePassed ? "passed" : "NOT MET (recorded as measured)"}`,
);
console.log(
  `  load            ${loaded.loadMs} ms · context ${runtime.limits.contextLength} tokens · request cap ${runtime.limits.maxRequestTokens} tokens · max new ${runtime.limits.maxNewTokens}`,
);

// One real turn through the same code path the chat backend uses.
const smoke = await runtime.respond({
  message: "Once upon a time",
  actorId: "artifact-smoke",
  conversationId: "artifact-smoke",
  useMemory: false,
  useRetrieval: false,
  useAgent: false,
  allowedTools: [],
  sampling: { maxNewTokens: 8, deterministic: true },
});
if (smoke.error) fail(`the built artifact failed its smoke generation: ${smoke.error.message}`);
console.log(
  `  smoke turn      ${smoke.generation?.generatedTokens ?? 0} tokens in ${smoke.durationMs} ms · "${smoke.response.replace(/\n/g, " ").slice(0, 72)}"`,
);
console.log("STEP 7 ARTIFACT OK");
