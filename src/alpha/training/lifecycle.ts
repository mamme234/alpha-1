/**
 * Alpha Training Engine — the full lifecycle run.
 *
 * This is the procedure that decides whether Alpha may be called
 * TRAINED (FROM SCRATCH). It performs, in order and for real:
 *
 *   1. initialise            architecture + tokenizer trained on the corpus
 *   2. corpus                validation, encoding, split, statistics
 *   3. train                 real optimiser steps (forward, loss, backward, AdamW)
 *   4. loss                  the loss curve the run actually produced
 *   5. parameters            proof that the weights moved
 *   6. checkpoint            write it, validate it, measure it
 *   7. reload                parse it, check compatibility, load the weights
 *   8. resume                continue the run from the checkpoint
 *   9. inference             run the reloaded model
 *  10. generated tokens      decode from the model's own argmax / sampling
 *
 * Every number in the returned report is measured, not asserted. If the
 * platform cannot execute a stage, the stage is reported as not ok with the
 * failure attached — the run is never reported as successful because it was
 * expected to be.
 */

import { ALPHA_MODEL_PRESETS, type AlphaModelConfig, type AlphaModelPreset } from "../model/config";
import { withSpecialTokenIds } from "../model/config";
import { AlphaTransformer } from "../model/transformer";
import { AlphaTokenizer } from "../tokenizer/bpe";
import { ALPHA_SEED_CORPUS } from "../datasets/seed-corpus";
import { assertValidDataset, type AlphaDataset } from "../datasets/types";
import { corpusReport, type CorpusReport } from "../datasets/corpus";
import { modelConfigFingerprint } from "../model/config";
import { AlphaInferenceService } from "../inference/service";
import type { GenerationResult } from "../inference/engine";
import { setGradEnabled } from "../core/tensor";
import { learningRateAt } from "./schedule";
import {
  AlphaTrainer,
  createTrainingConfig,
  type TrainingConfig,
  type TrainingMetricPoint,
} from "./trainer";
import {
  assertCheckpointCompatible,
  checkpointToJson,
  parseCheckpoint,
  validateCheckpoint,
  type AlphaCheckpoint,
} from "./checkpoint";
import { verifyAlphaModel, type VerificationReport } from "./verify";

export type LifecycleStageName =
  | "initialise"
  | "corpus"
  | "train"
  | "checkpoint"
  | "reload"
  | "resume"
  | "inference"
  | "verification";

export type LifecycleStage = {
  name: LifecycleStageName;
  ok: boolean;
  detail: string;
  durationMs: number;
  data: Record<string, string | number | boolean | null>;
};

export type AlphaLifecycleReport = {
  ok: boolean;
  /** Wall-clock time of the run, so a recorded result can be dated. */
  executedAt: string;
  durationMs: number;
  model: {
    name: string;
    version: string;
    preset: AlphaModelPreset;
    configFingerprint: string;
    parameterCount: number;
    vocabSize: number;
    contextLength: number;
    layers: number;
    heads: number;
    dModel: number;
  };
  tokenizer: { version: string; vocabSize: number; fingerprint: string; trainedOn: string; merges: number; characters: number };
  corpus: CorpusReport;
  training: {
    runId: string;
    seed: number;
    batchSize: number;
    seqLen: number;
    totalSteps: number;
    steps: number;
    firstLoss: number;
    lastLoss: number;
    bestLoss: number;
    uniformLoss: number;
    validationLoss: number | null;
    validationPerplexity: number | null;
    finalLearningRate: number;
    meanGradNorm: number;
    durationMs: number;
    tokensSeen: number;
    throughputTokensPerSecond: number;
  };
  checkpoint: {
    id: string;
    runId: string;
    formatVersion: string;
    step: number;
    tokensSeen: number;
    stage: string;
    payloadBytes: number;
    jsonBytes: number;
    tokenizerFingerprint: string;
    datasetFingerprint: string;
    valid: boolean;
    validationIssues: string[];
  };
  reload: {
    maxAbsoluteWeightDifference: number;
    compatible: boolean;
  };
  resume: {
    resumedAtStep: number;
    finalStep: number;
    stepsThisResume: number;
    lossAfterResume: number | null;
    optimizerStep: number;
  };
  generation: {
    prompt: string;
    text: string;
    tokenIds: number[];
    generatedTokens: number;
    stopReason: string;
    modelStage: string;
    deterministic: boolean;
    decodedFromIdsMatches: boolean;
    sampledText: string;
    meanNll: number;
  };
  verification: VerificationReport | null;
  stages: LifecycleStage[];
  notes: string[];
};

export type LifecycleOptions = {
  preset?: AlphaModelPreset;
  dataset?: AlphaDataset;
  /** Tokenizer vocabulary target. Defaults to the preset's model vocabulary. */
  tokenizerVocabSize?: number;
  /** Training overrides. Defaults are small enough to run on a laptop CPU. */
  training?: Partial<TrainingConfig>;
  /** Steps added after resuming from the checkpoint. */
  resumeSteps?: number;
  prompt?: string;
  /** Run the A–I verification suite as well (roughly doubles the work). */
  runVerification?: boolean;
  /** Resource estimate included in the report. */
  onStage?: (stage: LifecycleStage) => void;
};

export const DEFAULT_LIFECYCLE_TRAINING: Partial<TrainingConfig> = {
  batchSize: 8,
  seqLen: 32,
  totalSteps: 60,
  learningRate: 3e-3,
  warmupSteps: 6,
  minFactor: 0.1,
  evalInterval: 15,
  evalBatches: 4,
  checkpointInterval: 30,
  validationFraction: 0.12,
  seed: 1337,
  batchMode: "windows",
};

/**
 * Run the whole lifecycle and return a report of what actually happened.
 * Deterministic for a fixed seed and configuration.
 */
export function runAlphaTrainingLifecycle(options: LifecycleOptions = {}): AlphaLifecycleReport {
  const startedAt = Date.now();
  const executedAt = new Date(startedAt).toISOString();
  const stages: LifecycleStage[] = [];
  const notes: string[] = [];
  const preset = options.preset ?? "nano";
  const dataset = options.dataset ?? ALPHA_SEED_CORPUS;
  const trainingOverrides: Partial<TrainingConfig> = { ...DEFAULT_LIFECYCLE_TRAINING, ...options.training };
  const resumeSteps = options.resumeSteps ?? 2;

  const push = (stage: LifecycleStage) => {
    stages.push(stage);
    options.onStage?.(stage);
  };

  // --- 1. initialise -------------------------------------------------------
  const initStart = Date.now();
  assertValidDataset(dataset);
  const baseConfig = ALPHA_MODEL_PRESETS[preset];
  const tokenizer = AlphaTokenizer.train(dataset.documents, {
    vocabSize: Math.min(options.tokenizerVocabSize ?? baseConfig.vocabSize, baseConfig.vocabSize),
    version: "0.1.0",
    trainedOn: `${dataset.name}@${dataset.version}`,
  });
  const modelConfig: AlphaModelConfig = withSpecialTokenIds(
    { ...baseConfig, vocabSize: Math.max(baseConfig.vocabSize, tokenizer.vocabSize) },
    tokenizer.specialTokenIds,
  );
  const model = new AlphaTransformer(modelConfig);
  const config = createTrainingConfig(trainingOverrides);
  push({
    name: "initialise",
    ok: true,
    detail: `Built ${modelConfig.name} (${model.parameterCount.toLocaleString()} parameters) and trained the tokenizer on ${dataset.documents.length} documents.`,
    durationMs: Date.now() - initStart,
    data: {
      modelName: modelConfig.name,
      parameterCount: model.parameterCount,
      vocabSize: tokenizer.vocabSize,
      specialTokenIds: tokenizer.fingerprint(),
    },
  });

  // --- 2. corpus -----------------------------------------------------------
  const corpusStart = Date.now();
  const report = corpusReport(dataset, tokenizer, {
    seqLen: config.seqLen,
    batchSize: config.batchSize,
    validationFraction: config.validationFraction,
  });
  push({
    name: "corpus",
    ok: report.tokens > 0 && report.trainExamples > 0 && report.validationExamples > 0,
    detail:
      `${report.documents} documents / ${report.characters.toLocaleString()} characters → ${report.tokens.toLocaleString()} tokens ` +
      `(train ${report.trainTokens.toLocaleString()}, validation ${report.validationTokens.toLocaleString()}, ${report.unknownTokens} unknown). ` +
      `${report.trainExamples} training example(s), ${report.validationExamples} validation example(s) at seqLen ${report.sequenceLength}, batch ${report.batchSize}.`,
    durationMs: Date.now() - corpusStart,
    data: { ...report } as unknown as Record<string, string | number | boolean | null>,
  });

  // --- 3–5. train, loss, parameters ---------------------------------------
  const trainStart = Date.now();
  const weightsBefore = firstWeightSample(model);
  const trainer = new AlphaTrainer({
    model,
    tokenizer,
    dataset,
    config,
    checkpointLabel: `${modelConfig.name}-lifecycle`,
  });
  const checkpoints: AlphaCheckpoint[] = [];
  const history: TrainingMetricPoint[] = [];
  const iterator = trainer.run();
  let next = iterator.next();
  while (!next.done) {
    const event = next.value;
    if (event.type === "step") history.push(event.point);
    if (event.type === "checkpoint") checkpoints.push(event.checkpoint);
    next = iterator.next();
  }
  const summary = next.value;
  const weightsAfter = firstWeightSample(model);
  const parametersMoved = weightsBefore !== weightsAfter;
  const losses = history.map((point) => point.loss);
  const gradNorms = history.map((point) => point.gradNorm);
  const meanGradNorm = gradNorms.length ? gradNorms.reduce((a, b) => a + b, 0) / gradNorms.length : 0;
  const finalLearningRate = learningRateAt(trainer.step, trainer.schedule);
  push({
    name: "train",
    ok: summary.steps > 0 && Number.isFinite(losses[0] ?? Number.NaN),
    detail:
      `${summary.steps} optimiser step(s) over ${summary.tokensSeen.toLocaleString()} tokens in ${summary.durationMs} ms ` +
      `(${summary.throughputTokensPerSecond.toLocaleString()} tokens/s). Loss ${losses[0]?.toFixed(4) ?? "n/a"} → ${losses[losses.length - 1]?.toFixed(4) ?? "n/a"} ` +
      `against a uniform baseline of ${summary.uniformLossBaseline.toFixed(4)}; mean gradient norm ${meanGradNorm.toFixed(4)}. ` +
      `Weights ${parametersMoved ? "changed" : "did NOT change"}.`,
    durationMs: Date.now() - trainStart,
    data: {
      steps: summary.steps,
      tokensSeen: summary.tokensSeen,
      firstLoss: losses[0] ?? null,
      lastLoss: losses[losses.length - 1] ?? null,
      bestLoss: losses.length ? Math.min(...losses) : null,
      uniformLoss: summary.uniformLossBaseline,
      validationLoss: summary.validationLoss,
      parametersMoved,
      meanGradNorm,
      checkpointsWritten: checkpoints.length,
    },
  });

  // --- 6. checkpoint -------------------------------------------------------
  const checkpointStart = Date.now();
  const checkpoint = checkpoints[checkpoints.length - 1] ?? trainer.buildCheckpoint();
  const json = checkpointToJson(checkpoint);
  const validation = validateCheckpoint(checkpoint);
  push({
    name: "checkpoint",
    ok: validation.valid && checkpoint.step > 0,
    detail: validation.valid
      ? `Wrote and validated ${checkpoint.id} at step ${checkpoint.step}: ${checkpoint.sizeBytes.toLocaleString()} bytes of weights, ` +
        `${json.length.toLocaleString()} bytes of JSON, tokenizer ${checkpoint.tokenizer.fingerprint}, corpus ${checkpoint.datasetFingerprint}.`
      : `Checkpoint ${checkpoint.id} failed validation: ${validation.issues.join("; ")}`,
    durationMs: Date.now() - checkpointStart,
    data: {
      checkpointId: checkpoint.id,
      step: checkpoint.step,
      stage: checkpoint.stage,
      payloadBytes: checkpoint.sizeBytes,
      jsonBytes: json.length,
      issues: validation.issues.join(" | ") || null,
    },
  });

  // --- 7. reload -----------------------------------------------------------
  const reloadStart = Date.now();
  const restored = parseCheckpoint(json);
  const reloadedModel = new AlphaTransformer(modelConfig);
  let reloadDifference = Number.POSITIVE_INFINITY;
  let compatible = false;
  let reloadOk = false;
  let reloadDetail = "";
  try {
    assertCheckpointCompatible(restored, { config: modelConfig, tokenizer });
    compatible = true;
    reloadedModel.loadWeights(restored.weights);
    reloadDifference = maxWeightDifference(model, reloadedModel);
    reloadOk = reloadDifference === 0;
    reloadDetail = `Parsed ${restored.id}, verified its architecture (${restored.configFingerprint}) and tokenizer (${restored.tokenizer.fingerprint}) against the workspace, and reloaded the weights with a maximum absolute difference of ${reloadDifference}.`;
  } catch (error) {
    reloadDetail = `Reload failed: ${error instanceof Error ? error.message : String(error)}`;
  }
  push({
    name: "reload",
    ok: reloadOk,
    detail: reloadDetail,
    durationMs: Date.now() - reloadStart,
    data: { compatible, maxAbsoluteWeightDifference: Number.isFinite(reloadDifference) ? reloadDifference : null },
  });

  // --- 8. resume -----------------------------------------------------------
  const resumeStart = Date.now();
  const resumeTarget = restored.step + resumeSteps;
  const resumeModel = new AlphaTransformer(modelConfig);
  const resumedTrainer = new AlphaTrainer({
    model: resumeModel,
    tokenizer,
    dataset,
    config: { ...config, totalSteps: resumeTarget },
    checkpointLabel: `${modelConfig.name}-resume`,
    isFineTune: true,
  });
  let resumedAtStep = -1;
  let resumeOk = false;
  let resumeDetail = "";
  let resumeSummarySteps = 0;
  let lossAfterResume: number | null = null;
  let optimizerStep = -1;
  try {
    resumedTrainer.resumeFrom(restored);
    resumedAtStep = resumedTrainer.step;
    const resumedSummary = resumedTrainer.trainToCompletion();
    resumeSummarySteps = resumedSummary.steps;
    lossAfterResume = resumedSummary.lastLoss;
    optimizerStep = resumedTrainer.optimizer.step;
    resumeOk = resumedAtStep === restored.step && resumedTrainer.step === resumeTarget;
    resumeDetail = `Restored step ${resumedAtStep} (AdamW at step ${restored.optimizer.step}) and continued ${resumedSummary.steps} step(s) to ${resumedTrainer.step} — a resumed run, not a fresh model. Loss after resume ${resumedSummary.lastLoss?.toFixed(4) ?? "n/a"}.`;
  } catch (error) {
    resumeDetail = `Resume failed: ${error instanceof Error ? error.message : String(error)}`;
  }
  push({
    name: "resume",
    ok: resumeOk,
    detail: resumeDetail,
    durationMs: Date.now() - resumeStart,
    data: { resumedAtStep, finalStep: resumedTrainer.step, stepsThisResume: resumeSummarySteps, optimizerStep },
  });

  // --- 9–10. inference and generated tokens -------------------------------
  const inferenceStart = Date.now();
  const service = new AlphaInferenceService();
  const handle = service.registerModel({
    id: AlphaInferenceService.modelId(modelConfig.name, modelConfig.version, restored.tokenizer.fingerprint),
    name: modelConfig.name,
    version: modelConfig.version,
    stage: restored.stage,
    model: reloadedModel,
    tokenizer,
    contextLength: modelConfig.contextLength,
    checkpointId: restored.id,
  });
  const prompt = options.prompt ?? "Alpha is a self owned";
  const greedy = service.generate({ modelId: handle.id, prompt, generationConfig: { temperature: 0, maxNewTokens: 24, repetitionPenalty: 1 } });
  const greedyAgain = service.generate({ modelId: handle.id, prompt, generationConfig: { temperature: 0, maxNewTokens: 24, repetitionPenalty: 1 } });
  const sampled = service.generate({ modelId: handle.id, prompt, generationConfig: { temperature: 0.8, maxNewTokens: 24, seed: 7 } });
  const deterministic = greedy.text === greedyAgain.text && greedy.tokenIds.join(",") === greedyAgain.tokenIds.join(",");
  const decodedMatches = tokenizer.decode(greedy.tokenIds) === greedy.text;
  const producedTokens = greedy.generatedTokens > 0;
  push({
    name: "inference",
    ok: producedTokens && deterministic && decodedMatches,
    detail:
      `Greedy decode produced ${greedy.generatedTokens} token(s) from ${modelConfig.name} (${greedy.stopReason}); repeating it returned identical ids (${deterministic}), ` +
      `and decoding those ids reproduces the text exactly (${decodedMatches}). Sampled decode with seed 7 produced ${sampled.generatedTokens} token(s).`,
    durationMs: Date.now() - inferenceStart,
    data: {
      generatedTokens: greedy.generatedTokens,
      stopReason: greedy.stopReason,
      modelStage: greedy.modelStage,
      deterministic,
      decodedFromIdsMatches: decodedMatches,
      meanNll: greedy.meanNll,
    },
  });

  // --- optional: the A–I verification suite -------------------------------
  let verification: VerificationReport | null = null;
  if (options.runVerification) {
    const verifyStart = Date.now();
    // A fresh model, so check A inspects genuine initialisation rather than the
    // weights this run just trained.
    const verifyModel = new AlphaTransformer(modelConfig);
    verification = verifyAlphaModel({ model: verifyModel, tokenizer, dataset, prompt });
    push({
      name: "verification",
      ok: verification.passed,
      detail: `${verification.checks.filter((check) => check.passed).length}/${verification.checks.length} checks passed (A–I).`,
      durationMs: Date.now() - verifyStart,
      data: {
        passed: verification.passed,
        failed: verification.checks.filter((check) => !check.passed).map((check) => check.id).join(",") || null,
      },
    });
  }

  const ok = stages.every((stage) => stage.ok);
  if (!ok) {
    notes.push(
      `Stage(s) not ok: ${stages.filter((stage) => !stage.ok).map((stage) => stage.name).join(", ")}. The details above say why.`,
    );
  }
  if (losses.length > 0 && (losses[losses.length - 1] ?? Infinity) >= summary.uniformLossBaseline) {
    notes.push(
      "The final training loss did not beat the uniform baseline. With a corpus this small that is an honest outcome, not a bug — the report records it instead of hiding it.",
    );
  }
  notes.push(
    "Every loss value in this report comes from Alpha's own forward pass and cross-entropy; nothing is simulated.",
  );
  setGradEnabled(true);

  return {
    ok,
    executedAt,
    durationMs: Date.now() - startedAt,
    model: {
      name: modelConfig.name,
      version: modelConfig.version,
      preset,
      configFingerprint: modelConfigFingerprint(modelConfig),
      parameterCount: model.parameterCount,
      vocabSize: modelConfig.vocabSize,
      contextLength: modelConfig.contextLength,
      layers: modelConfig.nLayers,
      heads: modelConfig.nHeads,
      dModel: modelConfig.dModel,
    },
    tokenizer: {
      version: tokenizer.version,
      vocabSize: tokenizer.vocabSize,
      fingerprint: tokenizer.fingerprint(),
      trainedOn: tokenizer.trainedOn,
      merges: tokenizer.stats.mergeSteps,
      characters: tokenizer.stats.characters,
    },
    corpus: report,
    training: {
      runId: checkpoint.runId,
      seed: config.seed,
      batchSize: config.batchSize,
      seqLen: config.seqLen,
      totalSteps: config.totalSteps,
      steps: summary.steps,
      firstLoss: losses[0] ?? Number.NaN,
      lastLoss: losses[losses.length - 1] ?? Number.NaN,
      bestLoss: losses.length ? Math.min(...losses) : Number.NaN,
      uniformLoss: summary.uniformLossBaseline,
      validationLoss: summary.validationLoss,
      validationPerplexity: summary.validationPerplexity,
      finalLearningRate,
      meanGradNorm,
      durationMs: summary.durationMs,
      tokensSeen: summary.tokensSeen,
      throughputTokensPerSecond: summary.throughputTokensPerSecond,
    },
    checkpoint: {
      id: checkpoint.id,
      runId: checkpoint.runId,
      formatVersion: checkpoint.formatVersion,
      step: checkpoint.step,
      tokensSeen: checkpoint.tokensSeen,
      stage: checkpoint.stage,
      payloadBytes: checkpoint.sizeBytes,
      jsonBytes: json.length,
      tokenizerFingerprint: checkpoint.tokenizer.fingerprint,
      datasetFingerprint: checkpoint.datasetFingerprint,
      valid: validation.valid,
      validationIssues: validation.issues,
    },
    reload: {
      maxAbsoluteWeightDifference: Number.isFinite(reloadDifference) ? reloadDifference : -1,
      compatible,
    },
    resume: {
      resumedAtStep,
      finalStep: resumedTrainer.step,
      stepsThisResume: resumeSummarySteps,
      lossAfterResume,
      optimizerStep,
    },
    generation: {
      prompt,
      text: greedy.text,
      tokenIds: greedy.tokenIds,
      generatedTokens: greedy.generatedTokens,
      stopReason: greedy.stopReason,
      modelStage: greedy.modelStage,
      deterministic,
      decodedFromIdsMatches: decodedMatches,
      sampledText: sampled.text,
      meanNll: greedy.meanNll,
    },
    verification,
    stages,
    notes,
  };
}

/** A small sample of the first parameter tensor's values, for change detection. */
function firstWeightSample(model: AlphaTransformer): string {
  const parameter = model.parameters()[0];
  if (!parameter) return "";
  const data = parameter.tensor.data;
  const stride = Math.max(1, Math.floor(data.length / 64));
  const parts: string[] = [];
  for (let i = 0; i < data.length; i += stride) parts.push(data[i].toFixed(6));
  return parts.join(",");
}

function maxWeightDifference(a: AlphaTransformer, b: AlphaTransformer): number {
  const other = b.parameterMap();
  let worst = 0;
  for (const parameter of a.parameters()) {
    const match = other.get(parameter.name);
    if (!match || match.size !== parameter.tensor.size) return Number.POSITIVE_INFINITY;
    for (let i = 0; i < parameter.tensor.size; i++) {
      const difference = Math.abs(parameter.tensor.data[i] - match.data[i]);
      if (difference > worst) worst = difference;
    }
  }
  return worst;
}

/** Human-readable rendering of a lifecycle report, for the CLI and the docs. */
export function formatLifecycleReport(report: AlphaLifecycleReport): string {
  const lines: string[] = [];
  lines.push(`ALPHA LIFECYCLE REPORT — executed ${report.executedAt} in ${(report.durationMs / 1000).toFixed(2)}s`);
  lines.push(`Overall: ${report.ok ? "OK — every stage completed" : "NOT OK — see the failing stages"}`);
  lines.push("");
  lines.push(
    `Model        ${report.model.name} v${report.model.version} (${report.model.preset}) · ${report.model.parameterCount.toLocaleString()} params · ` +
      `${report.model.layers}L/${report.model.dModel}d/${report.model.heads}h · context ${report.model.contextLength} · vocab ${report.model.vocabSize}`,
  );
  lines.push(
    `Tokenizer    v${report.tokenizer.version} · ${report.tokenizer.vocabSize} tokens · ${report.tokenizer.merges} merges · ${report.tokenizer.fingerprint}`,
  );
  lines.push(
    `Corpus       ${report.corpus.name}@${report.corpus.version} (${report.corpus.license}) · ${report.corpus.documents} docs · ` +
      `${report.corpus.characters.toLocaleString()} chars · ${report.corpus.tokens.toLocaleString()} tokens`,
  );
  lines.push(
    `Split        train ${report.corpus.trainTokens.toLocaleString()} tokens / ${report.corpus.trainExamples} examples · ` +
      `validation ${report.corpus.validationTokens.toLocaleString()} tokens / ${report.corpus.validationExamples} examples`,
  );
  lines.push(
    `Training     ${report.training.steps} steps · batch ${report.training.batchSize} × seq ${report.training.seqLen} · seed ${report.training.seed} · ` +
      `${report.training.tokensSeen.toLocaleString()} tokens · ${report.training.throughputTokensPerSecond} tokens/s`,
  );
  lines.push(
    `Loss         first ${report.training.firstLoss.toFixed(4)} → last ${report.training.lastLoss.toFixed(4)} · best ${report.training.bestLoss.toFixed(4)} · ` +
      `uniform baseline ${report.training.uniformLoss.toFixed(4)} · validation ${report.training.validationLoss?.toFixed(4) ?? "n/a"}`,
  );
  lines.push(
    `Checkpoint   ${report.checkpoint.id} (run ${report.checkpoint.runId}) · step ${report.checkpoint.step} · stage ${report.checkpoint.stage} · ` +
      `${report.checkpoint.payloadBytes.toLocaleString()} bytes · valid ${report.checkpoint.valid}`,
  );
  lines.push(
    `Reload       compatible ${report.reload.compatible} · max absolute weight difference ${report.reload.maxAbsoluteWeightDifference}`,
  );
  lines.push(
    `Resume       step ${report.resume.resumedAtStep} → ${report.resume.finalStep} (+${report.resume.stepsThisResume}) · loss after resume ${report.resume.lossAfterResume?.toFixed(4) ?? "n/a"}`,
  );
  lines.push(
    `Inference    ${report.generation.generatedTokens} token(s) · stop ${report.generation.stopReason} · stage ${report.generation.modelStage} · deterministic ${report.generation.deterministic}`,
  );
  lines.push(`Prompt       ${JSON.stringify(report.generation.prompt)}`);
  lines.push(`Greedy text  ${JSON.stringify(report.generation.text)}`);
  lines.push(`Token ids    [${report.generation.tokenIds.join(", ")}]`);
  if (report.verification) {
    lines.push("");
    lines.push(`Verification ${report.verification.passed ? "PASSED" : "FAILED"}`);
    for (const check of report.verification.checks) {
      lines.push(`  ${check.id}. ${check.passed ? "ok  " : "FAIL"} ${check.label} — ${check.detail}`);
    }
  }
  lines.push("");
  lines.push("Stages:");
  for (const stage of report.stages) {
    lines.push(`  ${stage.ok ? "ok  " : "FAIL"} ${stage.name} (${stage.durationMs} ms) — ${stage.detail}`);
  }
  if (report.notes.length > 0) {
    lines.push("");
    lines.push("Notes:");
    for (const note of report.notes) lines.push(`  · ${note}`);
  }
  return lines.join("\n");
}
