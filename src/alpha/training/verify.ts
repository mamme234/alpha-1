/**
 * Alpha Training Engine — real model verification.
 *
 * A deterministic procedure that answers, from the code itself, the only
 * questions that matter before Alpha may be called "trained":
 *
 *   A. initial parameters are not all identical
 *   B. one training step changes parameters
 *   C. gradients are non-zero and match numerical differences
 *   D. the loss comes from the model's actual predictions
 *   E. a checkpoint can be saved
 *   F. the checkpoint can be reloaded
 *   G. reloaded parameters match the saved ones
 *   H. training resumes from the checkpoint
 *   I. inference runs on the trained/reloaded model and emits tokens
 *
 * Nothing here manufactures a decreasing loss. If the model fails to learn, the
 * report says so — the checks test mechanism, and the recorded losses are
 * whatever the run actually produced.
 */

import { AlphaRng } from "../core/rng";
import { backward, crossEntropy, maxAbsDiff, setGradEnabled } from "../core/tensor";
import { datasetFingerprint, type AlphaDataset } from "../datasets/types";
import { AlphaInferenceEngine, argmax } from "../inference/engine";
import { AlphaTransformer } from "../model/transformer";
import { modelConfigFingerprint, type AlphaModelConfig } from "../model/config";
import type { AlphaTokenizer } from "../tokenizer/bpe";
import { AlphaTrainer, type TrainingConfig } from "./trainer";
import { numericalGradientCheck } from "./gradients";
import {
  assertValidCheckpoint,
  checkpointToJson,
  parseCheckpoint,
  type AlphaCheckpoint,
} from "./checkpoint";

export type VerificationCheckId = "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I";

export type VerificationCheck = {
  id: VerificationCheckId;
  label: string;
  passed: boolean;
  detail: string;
  data: Record<string, number | string | boolean | null>;
};

export type VerificationReport = {
  passed: boolean;
  checks: VerificationCheck[];
  model: {
    name: string;
    version: string;
    configFingerprint: string;
    parameterCount: number;
    vocabSize: number;
    contextLength: number;
  };
  tokenizer: { version: string; vocabSize: number; fingerprint: string };
  dataset: { name: string; version: string; fingerprint: string; documents: number };
  training: {
    runId: string;
    seed: number;
    batchSize: number;
    seqLen: number;
    steps: number;
    firstLoss: number | null;
    lastLoss: number | null;
    bestLoss: number | null;
    uniformLoss: number;
    validationLoss: number | null;
    resumedSteps: number;
    checkpointId: string | null;
    durationMs: number;
  };
  generation: {
    prompt: string;
    text: string;
    tokens: number;
    stopReason: string;
    deterministic: boolean;
    modelStage: string;
    firstTokenMatchesArgmax: boolean;
  };
  startedAt: number;
  durationMs: number;
  notes: string[];
};

export type VerifyAlphaModelOptions = {
  /** A freshly constructed model — check A inspects the initial weights. */
  model: AlphaTransformer;
  tokenizer: AlphaTokenizer;
  dataset: AlphaDataset;
  /** Training overrides for the verification run. Sized small on purpose. */
  training?: Partial<TrainingConfig>;
  /** Prompt used for check I. */
  prompt?: string;
};

export const DEFAULT_VERIFY_TRAINING: Partial<TrainingConfig> = {
  batchSize: 2,
  seqLen: 16,
  totalSteps: 8,
  learningRate: 3e-3,
  warmupSteps: 2,
  evalInterval: 0,
  evalBatches: 2,
  checkpointInterval: 0,
  validationFraction: 0.25,
  seed: 20260929,
  batchMode: "windows",
};

/** Independent softmax + NLL over raw logits, used to cross-check the loss. */
function meanNllFromLogits(
  logits: Float32Array,
  rows: number,
  vocab: number,
  targets: Int32Array,
  ignoreIndex: number,
): { loss: number; counted: number } {
  let lossSum = 0;
  let counted = 0;
  for (let i = 0; i < rows; i++) {
    const offset = i * vocab;
    const target = targets[i];
    if (target === ignoreIndex || target < 0 || target >= vocab) continue;
    let max = -Infinity;
    for (let j = 0; j < vocab; j++) max = Math.max(max, logits[offset + j]);
    let sum = 0;
    for (let j = 0; j < vocab; j++) sum += Math.exp(logits[offset + j] - max);
    const prob = Math.exp(logits[offset + target] - max) / sum;
    lossSum += -Math.log(Math.max(prob, 1e-12));
    counted++;
  }
  return { loss: counted > 0 ? lossSum / counted : Number.NaN, counted };
}

/** Cheap stable hash of a weight payload — used to prove weights changed. */
function weightsHash(model: AlphaTransformer): string {
  let hash = 0x811c9dc5;
  const feed = (value: number) => {
    hash ^= value & 0xff;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  };
  for (const parameter of model.parameters()) {
    const data = parameter.tensor.data;
    const stride = Math.max(1, Math.floor(data.length / 512));
    for (let i = 0; i < data.length; i += stride) {
      const scaled = Math.round(data[i] * 1e6) | 0;
      feed(scaled);
      feed(scaled >> 8);
      feed(scaled >> 16);
    }
  }
  return `w_${hash.toString(16).padStart(8, "0")}`;
}

/** Largest absolute difference between two models' parameter sets. */
function weightsDifference(a: AlphaTransformer, b: AlphaTransformer): number {
  const other = b.parameterMap();
  let worst = 0;
  for (const parameter of a.parameters()) {
    const match = other.get(parameter.name);
    if (!match || match.size !== parameter.tensor.size) return Number.POSITIVE_INFINITY;
    worst = Math.max(worst, maxAbsDiff(parameter.tensor, match));
  }
  return worst;
}

/** A small deterministic batch used for the gradient and loss checks. */
function buildProbeBatch(
  tokenizer: AlphaTokenizer,
  config: AlphaModelConfig,
  seed = 3,
): { input: Int32Array; target: Int32Array; batch: number; seqLen: number } {
  const rng = new AlphaRng(seed);
  const batch = 2;
  const seqLen = Math.min(8, Math.max(2, config.contextLength - 1));
  const input = new Int32Array(batch * seqLen);
  const target = new Int32Array(batch * seqLen);
  const vocab = tokenizer.vocabSize;
  for (let i = 0; i < input.length; i++) {
    input[i] = 4 + rng.int(Math.max(1, vocab - 4));
    target[i] = 4 + rng.int(Math.max(1, vocab - 4));
  }
  return { input, target, batch, seqLen };
}

export function verifyAlphaModel(options: VerifyAlphaModelOptions): VerificationReport {
  const startedAt = Date.now();
  const { model, tokenizer, dataset } = options;
  const config: AlphaModelConfig = model.config;
  const checks: VerificationCheck[] = [];
  const notes: string[] = [];
  const uniformLoss = Math.log(tokenizer.vocabSize);

  // --- A. initial parameters are not all identical -------------------------
  const initialHash = weightsHash(model);
  const embedding = model.parameterMap().get("token_embedding");
  let distinctSampled = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  if (embedding) {
    const seen = new Set<number>();
    const stride = Math.max(1, Math.floor(embedding.size / 512));
    for (let i = 0; i < embedding.size; i += stride) {
      const value = embedding.data[i];
      seen.add(value);
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    distinctSampled = seen.size;
  }
  checks.push({
    id: "A",
    label: "Initial parameters are not all identical",
    passed: distinctSampled > 1,
    detail:
      distinctSampled > 1
        ? `Sampled ${distinctSampled} distinct values from the token embedding; range [${min.toExponential(3)}, ${max.toExponential(3)}].`
        : "Sampled token-embedding values are all identical — initialisation is degenerate.",
    data: { distinctSampledValues: distinctSampled, min, max, initialWeightsHash: initialHash },
  });

  // --- shared probe batch for C and D -------------------------------------
  const probe = buildProbeBatch(tokenizer, config);

  // --- B. a training step changes parameters -------------------------------
  const stepper = new AlphaTrainer({
    model,
    tokenizer,
    dataset,
    config: {
      ...DEFAULT_VERIFY_TRAINING,
      ...options.training,
      totalSteps: 1,
      evalInterval: 0,
      checkpointInterval: 0,
    },
    checkpointLabel: "verify-step",
    runId: `run_verify_step_${startedAt.toString(36)}`,
  });
  const stepSummary = stepper.trainToCompletion();
  const afterStepHash = weightsHash(model);
  const changed = afterStepHash !== initialHash;
  checks.push({
    id: "B",
    label: "A training step changes parameters",
    passed: changed,
    detail: changed
      ? `Weights hash moved from ${initialHash} to ${afterStepHash} after one real optimiser step (loss ${stepSummary.lastLoss?.toFixed(4) ?? "n/a"}).`
      : "Weights hash is unchanged after an optimiser step — no parameter update happened.",
    data: { before: initialHash, after: afterStepHash, loss: stepSummary.lastLoss },
  });

  // --- C. gradients are non-zero and match numerical differences -----------
  setGradEnabled(true);
  for (const parameter of model.parameters()) parameter.tensor.zeroGrad();
  const forward = model.forward(probe.input, probe.batch, probe.seqLen, { training: false });
  const loss = crossEntropy(forward.logits, probe.target, tokenizer.padId);
  backward(loss.tensor);
  let nonZero = 0;
  let sampledEntries = 0;
  for (const parameter of model.parameters()) {
    const grad = parameter.tensor.grad;
    if (!grad) continue;
    const stride = Math.max(1, Math.floor(grad.length / 64));
    for (let i = 0; i < grad.length; i += stride) {
      sampledEntries++;
      if (Math.abs(grad[i]) > 1e-9) nonZero++;
    }
  }
  const gradientReport = numericalGradientCheck(model, probe, {
    padId: tokenizer.padId,
    samplesPerTensor: 2,
  });
  checks.push({
    id: "C",
    label: "Gradients are non-zero and match numerical differences",
    passed: nonZero > 0 && gradientReport.passed,
    detail:
      nonZero > 0
        ? `${nonZero}/${sampledEntries} sampled gradient entries are non-zero. ${gradientReport.detail}`
        : "Every sampled gradient entry is zero — backpropagation produced nothing.",
    data: {
      nonZeroEntries: nonZero,
      sampledEntries,
      maxRelativeError: gradientReport.maxRelativeError,
      gradientCheckPassed: gradientReport.passed,
    },
  });

  // --- D. loss comes from actual predictions -------------------------------
  setGradEnabled(false);
  const reported = crossEntropy(
    model.forward(probe.input, probe.batch, probe.seqLen, { training: false }).logits,
    probe.target,
    tokenizer.padId,
  );
  const independent = meanNllFromLogits(
    model.forward(probe.input, probe.batch, probe.seqLen, { training: false }).logits.data,
    probe.batch * probe.seqLen,
    config.vocabSize,
    probe.target,
    tokenizer.padId,
  );
  const otherProbe = buildProbeBatch(tokenizer, config, 7);
  const otherLoss = crossEntropy(
    model.forward(otherProbe.input, otherProbe.batch, otherProbe.seqLen, { training: false }).logits,
    otherProbe.target,
    tokenizer.padId,
  ).loss;
  setGradEnabled(true);
  const matches = Math.abs(reported.loss - independent.loss) < 1e-6;
  const discriminates = Math.abs(reported.loss - otherLoss) > 1e-9;
  const notConstant = Math.abs(reported.loss - uniformLoss) > 1e-9;
  checks.push({
    id: "D",
    label: "Loss is calculated from the model's actual predictions",
    passed: matches && discriminates && notConstant,
    detail: matches
      ? `Model loss ${reported.loss.toFixed(6)} equals an independent softmax/NLL recomputation (${independent.loss.toFixed(6)}) over ${reported.tokens} tokens; a different batch gives ${otherLoss.toFixed(6)} and the uniform baseline is ${uniformLoss.toFixed(6)}.`
      : `Model loss ${reported.loss.toFixed(6)} does not match the independent recomputation (${independent.loss.toFixed(6)}).`,
    data: {
      reportedLoss: reported.loss,
      independentLoss: independent.loss,
      otherBatchLoss: otherLoss,
      uniformLoss,
      tokens: reported.tokens,
    },
  });

  // --- the real training run used by E–I ----------------------------------
  const trainingConfig = { ...DEFAULT_VERIFY_TRAINING, ...options.training };
  const trainer = new AlphaTrainer({
    model,
    tokenizer,
    dataset,
    config: { ...trainingConfig, evalInterval: 0, checkpointInterval: 0 },
    checkpointLabel: "verify-run",
    runId: `run_verify_${startedAt.toString(36)}`,
  });
  const summary = trainer.trainToCompletion();
  const validationLoss = trainer.evaluate({ maxBatches: 2 }).loss;

  // --- E. a checkpoint can be saved ---------------------------------------
  let saved = false;
  let saveDetail = "";
  let checkpoint: AlphaCheckpoint | null = null;
  let json = "";
  try {
    checkpoint = trainer.buildCheckpoint();
    json = checkpointToJson(checkpoint);
    saved = json.length > 0 && checkpoint.step === trainer.step;
    saveDetail = `Wrote ${checkpoint.id} at step ${checkpoint.step} (${json.length.toLocaleString()} bytes of JSON, ${checkpoint.sizeBytes.toLocaleString()} bytes of weights).`;
  } catch (error) {
    saveDetail = `Checkpoint failed its own validation: ${error instanceof Error ? error.message : String(error)}`;
  }
  checks.push({
    id: "E",
    label: "A checkpoint can be saved",
    passed: saved,
    detail: saveDetail,
    data: {
      checkpointId: checkpoint?.id ?? null,
      step: checkpoint?.step ?? null,
      jsonBytes: json.length,
      weightBytes: checkpoint?.sizeBytes ?? null,
    },
  });

  // --- F. the checkpoint can be reloaded ----------------------------------
  let restored: AlphaCheckpoint | null = null;
  let reloadDetail = "";
  try {
    restored = parseCheckpoint(json);
    assertValidCheckpoint(restored);
    reloadDetail = `Parsed ${restored.id} back from JSON and re-validated it (format ${restored.formatVersion}, tokenizer ${restored.tokenizer.fingerprint}).`;
  } catch (error) {
    restored = null;
    reloadDetail = `Reload failed: ${error instanceof Error ? error.message : String(error)}`;
  }
  checks.push({
    id: "F",
    label: "The checkpoint can be reloaded",
    passed: restored !== null,
    detail: reloadDetail,
    data: {
      formatVersion: restored?.formatVersion ?? null,
      tokenizerFingerprint: restored?.tokenizer.fingerprint ?? null,
    },
  });

  // --- G. reloaded parameters match the saved ones ------------------------
  const reloadedModel = new AlphaTransformer(config);
  let reloadDelta = Number.POSITIVE_INFINITY;
  if (checkpoint) {
    reloadedModel.loadWeights(checkpoint.weights);
    reloadDelta = weightsDifference(model, reloadedModel);
  }
  const identical = checkpoint !== null && reloadDelta === 0;
  checks.push({
    id: "G",
    label: "Reloaded parameters match the saved parameters",
    passed: identical,
    detail: identical
      ? "Every tensor in the payload reloaded with a maximum absolute difference of exactly 0 across the whole parameter set."
      : `Reloaded weights differ from the trained model's weights (max absolute difference ${reloadDelta}).`,
    data: { maxAbsoluteDifference: Number.isFinite(reloadDelta) ? reloadDelta : null },
  });

  // --- H. training resumes from the checkpoint ----------------------------
  let resumePassed = false;
  let resumeDetail = "";
  let resumedSteps = 0;
  let resumedLoss: number | null = null;
  const resumeTarget = trainer.step + 2;
  if (checkpoint) {
    try {
      const resumedModel = new AlphaTransformer(config);
      const resumed = new AlphaTrainer({
        model: resumedModel,
        tokenizer,
        dataset,
        config: { ...trainingConfig, totalSteps: resumeTarget, evalInterval: 0, checkpointInterval: 0 },
        checkpointLabel: "verify-resume",
        runId: `run_verify_resume_${startedAt.toString(36)}`,
        isFineTune: true,
      });
      resumed.resumeFrom(checkpoint);
      const atResume = resumed.step;
      const resumeSummary = resumed.trainToCompletion();
      resumedSteps = resumeSummary.steps;
      resumedLoss = resumeSummary.lastLoss;
      resumePassed =
        atResume === checkpoint.step &&
        resumed.step === resumeTarget &&
        resumed.optimizer.step === resumeTarget;
      resumeDetail =
        `Restored step ${atResume} and AdamW at step ${checkpoint.step}, then trained ${resumeSummary.steps} more step(s) ` +
        `to ${resumed.step} (loss ${resumeSummary.lastLoss?.toFixed(4) ?? "n/a"}, validation ${resumeSummary.validationLoss?.toFixed(4) ?? "n/a"}).`;
    } catch (error) {
      resumeDetail = `Resume failed: ${error instanceof Error ? error.message : String(error)}`;
    }
  } else {
    resumeDetail = "No checkpoint was produced, so resume could not be attempted.";
  }
  checks.push({
    id: "H",
    label: "Training can resume from the checkpoint",
    passed: resumePassed,
    detail: resumeDetail,
    data: {
      resumedFromStep: checkpoint?.step ?? null,
      stepsAfterResume: resumedSteps,
      lossAfterResume: resumedLoss,
    },
  });

  // --- I. inference uses the trained / reloaded model ---------------------
  const prompt = options.prompt ?? "Alpha is a self owned";
  const engine = new AlphaInferenceEngine({
    model: reloadedModel,
    tokenizer,
    stage: checkpoint?.stage ?? "untrained",
  });
  const greedy = engine.generate(prompt, { temperature: 0, maxNewTokens: 12, repetitionPenalty: 1 });
  const greedyAgain = engine.generate(prompt, { temperature: 0, maxNewTokens: 12, repetitionPenalty: 1 });
  const sampled = engine.generate(prompt, { temperature: 0.8, maxNewTokens: 12, seed: 7 });
  const deterministic = greedy.text === greedyAgain.text && greedy.tokenIds.join(",") === greedyAgain.tokenIds.join(",");
  const producedTokens = greedy.generatedTokens > 0 || sampled.generatedTokens > 0;
  const promptIds = tokenizer.encodeDetailed(prompt, {
    maxLength: engine.maxContextTokens - 1,
    truncation: "left",
    addBos: true,
  }).ids;
  setGradEnabled(false);
  // Mirror the engine: structural tokens are excluded from free generation, so
  // the expectation is the argmax of the same restricted distribution.
  const rawFirst = engine.scoreNextTokens(promptIds);
  const restricted = Float32Array.from(rawFirst);
  if (tokenizer.padId >= 0 && tokenizer.padId < restricted.length) restricted[tokenizer.padId] = -Infinity;
  const expectedFirst = argmax(restricted);
  setGradEnabled(true);
  const firstTokenMatchesArgmax = greedy.tokenIds.length === 0 || greedy.tokenIds[0] === expectedFirst;
  checks.push({
    id: "I",
    label: "Inference runs on the trained/reloaded model and emits tokens",
    passed: producedTokens && deterministic && firstTokenMatchesArgmax,
    detail:
      `Greedy decode produced ${greedy.generatedTokens} token(s) from ${config.name} (${greedy.stopReason}); ` +
      `repeating it gives identical ids: ${deterministic}. The first generated token ${greedy.tokenIds[0] ?? "n/a"} is the argmax ` +
      `${expectedFirst} of the reloaded model's own logits (${firstTokenMatchesArgmax}). ` +
      `Sampled decode with seed 7 produced ${sampled.generatedTokens} token(s).`,
    data: {
      greedyTokens: greedy.generatedTokens,
      sampledTokens: sampled.generatedTokens,
      stopReason: greedy.stopReason,
      deterministic,
      firstToken: greedy.tokenIds[0] ?? null,
      expectedFirst,
      modelStage: greedy.modelStage,
    },
  });

  const passed = checks.every((check) => check.passed);
  if (!passed) notes.push("At least one verification check failed — the report above says which and why.");
  if (summary.lastLoss !== null && summary.lastLoss >= uniformLoss) {
    notes.push(
      "The training loss did not beat the uniform baseline in this short verification run. That is recorded rather than hidden; the mechanism checks (A–I) are independent of how much the model learned.",
    );
  }
  if (options.training === undefined) {
    notes.push(
      `Verification run used its own small configuration (${trainingConfig.batchSize}×${trainingConfig.seqLen}, ${trainingConfig.totalSteps} steps, seed ${trainingConfig.seed}) so the report stays deterministic.`,
    );
  }

  return {
    passed,
    checks,
    model: {
      name: config.name,
      version: config.version,
      configFingerprint: modelConfigFingerprint(config),
      parameterCount: model.parameterCount,
      vocabSize: config.vocabSize,
      contextLength: config.contextLength,
    },
    tokenizer: {
      version: tokenizer.version,
      vocabSize: tokenizer.vocabSize,
      fingerprint: tokenizer.fingerprint(),
    },
    dataset: {
      name: dataset.name,
      version: dataset.version,
      fingerprint: datasetFingerprint(dataset),
      documents: dataset.documents.length,
    },
    training: {
      runId: trainer.runId,
      seed: trainer.config.seed,
      batchSize: trainer.config.batchSize,
      seqLen: trainer.config.seqLen,
      steps: summary.steps,
      firstLoss: summary.firstLoss,
      lastLoss: summary.lastLoss,
      bestLoss: summary.bestLoss,
      uniformLoss: summary.uniformLossBaseline,
      validationLoss,
      resumedSteps,
      checkpointId: checkpoint?.id ?? null,
      durationMs: summary.durationMs,
    },
    generation: {
      prompt,
      text: greedy.text,
      tokens: greedy.generatedTokens,
      stopReason: greedy.stopReason,
      deterministic,
      modelStage: greedy.modelStage,
      firstTokenMatchesArgmax,
    },
    startedAt,
    durationMs: Date.now() - startedAt,
    notes,
  };
}
