/**
 * Alpha — freeze the Step 7 baseline.
 *
 *   bun scripts/alpha-freeze-baseline.ts
 *
 * Runs once before any Step 7 work: it records the exact bytes of the Step 5
 * production serving artifact, every fingerprint inside it, the Step 5
 * evaluation numbers the production gate passed on, and a proof that the
 * serving layer still loads it. The record is written to
 * `src/alpha/experiments/step7-baseline-record.json`.
 *
 * Re-running never rewrites a different record: if the file exists it must
 * match what this run would write, or the script fails. That is the guarantee
 * that the production baseline was not overwritten by Step 7.
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { loadServingArtifact, parseServingArtifact } from "../src/alpha/serving/artifact";
import { createServingRuntime } from "../src/alpha/serving/runtime";

const ARTIFACT_PATH = "src/alpha/serving/step5-artifact.json";
const RECORD_PATH = "src/alpha/experiments/step7-baseline-record.json";

function fail(message: string): never {
  console.error(`\nBASELINE FREEZE FAILED: ${message}`);
  process.exit(1);
}

const raw = readFileSync(ARTIFACT_PATH);
const sha256 = createHash("sha256").update(raw).digest("hex");
const artifact = parseServingArtifact(JSON.parse(raw.toString("utf8")));

// Proof that Step 6's serving layer still loads this exact file.
let loadsInServing = false;
let loadMs = 0;
let contextLength = 0;
try {
  const runtime = createServingRuntime({ artifact });
  loadsInServing = true;
  loadMs = runtime.info.loadMs;
  contextLength = runtime.info.contextLength;
} catch (error) {
  fail(`the serving runtime refused the baseline artifact: ${error instanceof Error ? error.message : String(error)}`);
}

const record = {
  recordedAt: Date.now(),
  purpose: "Step 7 baseline: the production model at the start of Step 7. Never overwritten, always loadable.",
  artifact: {
    path: ARTIFACT_PATH,
    bytes: raw.byteLength,
    sha256,
    formatVersion: artifact.formatVersion,
    createdAt: artifact.createdAt,
  },
  identity: {
    modelId: `alpha-micro@${artifact.model.config.version}+${artifact.tokenizer.fingerprint}`,
    name: artifact.model.config.name,
    version: artifact.model.config.version,
    stage: artifact.stage,
    parameterCount: artifact.model.parameterCount,
    contextLength: artifact.model.config.contextLength,
  },
  fingerprints: {
    config: artifact.model.configFingerprint,
    tokenizer: artifact.tokenizer.fingerprint,
    dataset: artifact.data.datasetFingerprint,
    mixture: artifact.data.mixtureFingerprint,
    suite: artifact.evaluation.suiteFingerprint,
    gate: artifact.evaluation.gateFingerprint,
    checkpoint: artifact.training.checkpointId,
    datasetName: artifact.data.datasetName,
    datasetVersion: artifact.data.datasetVersion,
  },
  training: {
    steps: artifact.training.steps,
    tokensSeen: artifact.training.tokensSeen,
    durationMs: artifact.training.durationMs,
    tokensPerSecond: artifact.training.tokensPerSecond,
    seed: artifact.training.seed,
    firstLoss: artifact.training.firstLoss,
    lastLoss: artifact.training.lastLoss,
    validationLoss: artifact.training.validationLoss,
    validationPerplexity: artifact.training.validationPerplexity,
    gradientAccumulationSteps: artifact.training.gradientAccumulationSteps,
    config: artifact.training.config,
  },
  evaluation: {
    suiteFingerprint: artifact.evaluation.suiteFingerprint,
    suiteCases: artifact.evaluation.suiteCases,
    heldOutDocuments: artifact.evaluation.heldOutDocuments,
    loss: artifact.evaluation.loss,
    perplexity: artifact.evaluation.perplexity,
    nextTokenTop1Accuracy: artifact.evaluation.nextTokenTop1Accuracy,
    gateFingerprint: artifact.evaluation.gateFingerprint,
    gatePassed: artifact.evaluation.gatePassed,
  },
  serving: {
    loads: loadsInServing,
    loadMs,
    contextLength,
  },
};

const serialised = `${JSON.stringify(record, null, 2)}\n`;
if (existsSync(RECORD_PATH)) {
  const existing = JSON.parse(readFileSync(RECORD_PATH, "utf8")) as typeof record;
  // `recordedAt` is the moment of the first freeze and `loadMs` is a timing;
  // everything else must match the artifact, because that is what the record claims.
  const wouldWrite = { ...record, recordedAt: 0, serving: { ...record.serving, loadMs: 0 } };
  const onDisk = { ...existing, recordedAt: 0, serving: { ...existing.serving, loadMs: 0 } };
  if (JSON.stringify(wouldWrite, null, 2) !== JSON.stringify(onDisk, null, 2)) {
    fail(
      `${RECORD_PATH} already exists with different contents. The baseline record is immutable; ` +
        `the production artifact on disk no longer matches what Step 7 froze.`,
    );
  }
  console.log("Baseline record already frozen and matches the artifact on disk — nothing rewritten.");
} else {
  mkdirSync("src/alpha/experiments", { recursive: true });
  writeFileSync(RECORD_PATH, serialised);
  console.log(`Baseline record written to ${RECORD_PATH}`);
}

console.log(`  artifact   ${ARTIFACT_PATH} (${raw.byteLength.toLocaleString()} bytes)`);
console.log(`  sha256     ${sha256}`);
console.log(`  model      ${record.identity.modelId} · stage ${record.identity.stage} · ${record.identity.parameterCount.toLocaleString()} params · ${record.identity.contextLength}-token context`);
console.log(`  fingerprints config ${record.fingerprints.config} · tokenizer ${record.fingerprints.tokenizer} · dataset ${record.fingerprints.dataset} · suite ${record.fingerprints.suite} · gate ${record.fingerprints.gate}`);
console.log(`  training   ${record.training.steps} steps · ${record.training.tokensSeen.toLocaleString()} tokens · validation loss ${record.training.validationLoss} · validation ppl ${record.training.validationPerplexity}`);
console.log(`  evaluation loss ${record.evaluation.loss} · perplexity ${record.evaluation.perplexity} · top1 ${(record.evaluation.nextTokenTop1Accuracy ?? 0) * 100}% · gate ${record.evaluation.gateFingerprint} ${record.evaluation.gatePassed ? "passed" : "FAILED"}`);
console.log(`  serving    loads in the Step 6 runtime: ${record.serving.loads} (${record.serving.loadMs} ms)`);
process.exit(0);
