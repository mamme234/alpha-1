/**
 * Model evaluation.
 *
 * Everything in this module is *measured*. There is no intelligence score, no
 * opinionated ranking, and no synthetic number standing in for a benchmark that
 * was not run. A metric that could not be computed is `null` and says why, not
 * a plausible-looking guess.
 *
 * The distinction that matters most here is between the two kinds of capability
 * result. `recall` cases use material Alpha may have trained on, so a pass can
 * mean memorisation. `held-out` cases use material Alpha was never trained on,
 * so a pass means something was generalised. They are reported separately and
 * never averaged into one number.
 */

import { AlphaValidationError } from "../core/errors";
import { AlphaTransformer } from "../model/transformer";
import { modelConfigFingerprint, type AlphaModelConfig } from "../model/config";
import { AlphaTokenizer } from "../tokenizer/bpe";
import { AlphaInferenceEngine, type SamplingConfig } from "../inference/engine";
import { heldOutCases, recallCases } from "./benchmark";

/** Perplexity is exp(loss); beyond this the number is not meaningful to print. */
const MAX_REPORTABLE_PERPLEXITY = 1e12;

export type EvaluationMetric = {
  name: string;
  /** The measured value, or null when the metric could not be computed. */
  value: number | null;
  unit: string;
  /** What was actually done, so the number can be audited. */
  method: string;
  /** Set when `value` is null. */
  note?: string;
};

export type CapabilityResult = {
  kind: "recall" | "held-out";
  prompt: string;
  /** Token ids the model actually produced. */
  generated: number[];
  generatedText: string;
  expected: string[];
  /** True when any expected substring appears in the generated text. */
  matched: boolean;
};

export type EvaluationReport = {
  /** Identity of what was measured. */
  modelName: string;
  modelVersion: string;
  configFingerprint: string;
  parameterCount: number;
  tokenizerVersion: string;
  tokenizerFingerprint: string;
  contextLength: number;
  /** How many parameters were trained, when a checkpoint supplied the model. */
  trainedTokens: number | null;
  metrics: EvaluationMetric[];
  capabilities: CapabilityResult[];
  /** Counts only. No score, no grade, no ranking. */
  capabilityCounts: {
    recall: { total: number; matched: number };
    heldOut: { total: number; matched: number };
  };
  checks: Array<{ name: string; passed: boolean; detail: string }>;
  createdAt: number;
};

export type EvaluationOptions = {
  /** Text used for language-modelling loss. Never the training corpus. */
  evaluationDocuments: string[];
  sampling?: Partial<SamplingConfig>;
  maxNewTokens?: number;
  /** Set when the weights came from a real training run. */
  trainedTokens?: number | null;
  now?: number;
};

/** Cross-entropy of the model's own next-token predictions, in nats per token. */
function languageModelLoss(
  model: AlphaTransformer,
  tokenizer: AlphaTokenizer,
  documents: string[],
  contextLength: number,
): { loss: number; tokens: number; batches: number } {
  let total = 0;
  let tokens = 0;
  let batches = 0;
  const seqLen = Math.min(contextLength, 32);
  for (const doc of documents) {
    const ids = tokenizer.encode(doc);
    if (ids.length < 2) continue;
    for (let start = 0; start + 1 < ids.length; start += seqLen - 1) {
      const window = ids.slice(start, Math.min(ids.length, start + seqLen));
      if (window.length < 2) break;
      const input = Int32Array.from(window.slice(0, -1));
      const target = window.slice(1);
      const out = model.forward(input, 1, input.length, { training: false });
      const vocab = model.config.vocabSize;
      for (let t = 0; t < target.length; t++) {
        // `base` is the first logit of this position; `selected` is the logit
        // the model assigned to the token that actually came next.
        const base = t * vocab;
        const selected = base + target[t];
        // log-sum-exp over the full vocabulary, from the model's own logits.
        let max = -Infinity;
        for (let v = 0; v < vocab; v++) {
          const value = out.logits.data[base + v];
          if (value > max) max = value;
        }
        let sum = 0;
        for (let v = 0; v < vocab; v++) {
          sum += Math.exp(out.logits.data[base + v] - max);
        }
        total += -(out.logits.data[selected] - max - Math.log(sum));
        tokens++;
      }
      batches++;
    }
  }
  return { loss: tokens > 0 ? total / tokens : Number.NaN, tokens, batches };
}

function perplexityFrom(loss: number): number | null {
  if (!Number.isFinite(loss)) return null;
  const value = Math.exp(loss);
  return value <= MAX_REPORTABLE_PERPLEXITY ? value : null;
}

/** Fraction of generated token ids that repeat their immediate predecessor. */
function repetitionOf(ids: number[]): number {
  if (ids.length < 2) return 0;
  let same = 0;
  for (let i = 1; i < ids.length; i++) if (ids[i] === ids[i - 1]) same++;
  return same / (ids.length - 1);
}

/** Longest run of one repeated token id. */
function longestRun(ids: number[]): number {
  let best = 0;
  let current = 0;
  let previous = -1;
  for (const id of ids) {
    current = id === previous ? current + 1 : 1;
    previous = id;
    if (current > best) best = current;
  }
  return best;
}

function toMetric(
  name: string,
  value: number | null,
  unit: string,
  method: string,
  note?: string,
): EvaluationMetric {
  return { name, value, unit, method, ...(note ? { note } : {}) };
}

/**
 * Evaluate a model on a fixed suite. The suite is the benchmark in
 * `./benchmark` plus deterministic behavioural checks; nothing is trained on.
 */
export function evaluateModel(
  model: AlphaTransformer,
  tokenizer: AlphaTokenizer,
  options: EvaluationOptions,
): EvaluationReport {
  const config: AlphaModelConfig = model.config;
  if (options.evaluationDocuments.length === 0) {
    throw new AlphaValidationError(
      "model",
      "evaluation needs at least one document; an empty suite cannot produce a loss",
    );
  }

  const lm = languageModelLoss(model, tokenizer, options.evaluationDocuments, config.contextLength);
  const perplexity = perplexityFrom(lm.loss);

  // Uniform-distribution baseline: what an untrained model with no information
  // would score. The gap between this and the measured loss is the real result.
  const uniformLoss = Math.log(config.vocabSize);
  const uniformPerplexity = config.vocabSize;

  const engine = new AlphaInferenceEngine({
    model,
    tokenizer,
    stage: "trained",
  });
  const maxNewTokens = options.maxNewTokens ?? 24;

  // --- deterministic generation -------------------------------------------
  const determinismPrompt = "alpha";
  const greedy = { temperature: 0, ...options.sampling, maxNewTokens };
  const first = engine.generate(determinismPrompt, greedy);
  const second = engine.generate(determinismPrompt, greedy);
  const deterministic =
    first.tokenIds.length === second.tokenIds.length &&
    first.tokenIds.every((id, i) => id === second.tokenIds[i]);

  // --- tokenizer round trip -----------------------------------------------
  const roundTripSource = options.evaluationDocuments[0];
  const roundTrip = tokenizer.decode(tokenizer.encode(roundTripSource));
  const roundTripOk = tokenizer.encode(roundTrip).join(",") === tokenizer.encode(roundTripSource).join(",");

  // --- stop conditions -----------------------------------------------------
  const eosId = tokenizer.eosId;
  const stoppedCleanly =
    first.stopReason === "eos" || first.stopReason === "max-tokens" || first.tokenIds.length <= maxNewTokens;

  // --- capability cases ----------------------------------------------------
  const capabilities: CapabilityResult[] = [];
  for (const c of recallCases()) {
    const result = engine.generate(c.prompt, { temperature: 0, maxNewTokens: 12 });
    const text = result.text.toLowerCase();
    capabilities.push({
      kind: "recall",
      prompt: c.prompt,
      generated: result.tokenIds,
      generatedText: result.text,
      expected: c.expect,
      matched: c.expect.some((e) => text.includes(e.toLowerCase())),
    });
  }
  for (const c of heldOutCases()) {
    const result = engine.generate(c.prompt, { temperature: 0, maxNewTokens: 12 });
    const text = result.text.toLowerCase();
    capabilities.push({
      kind: "held-out",
      prompt: c.prompt,
      generated: result.tokenIds,
      generatedText: result.text,
      expected: c.expect,
      matched: c.expect.some((e) => text.includes(e.toLowerCase())),
    });
  }

  const recallTotal = capabilities.filter((c) => c.kind === "recall");
  const heldOutTotal = capabilities.filter((c) => c.kind === "held-out");

  const metrics: EvaluationMetric[] = [
    toMetric(
      "languageModelLoss",
      Number.isFinite(lm.loss) ? lm.loss : null,
      "nats/token",
      `cross-entropy over ${lm.tokens} next-token predictions from ${lm.batches} window(s) of held-out text`,
      Number.isFinite(lm.loss) ? undefined : "no usable evaluation window: documents are shorter than 2 tokens",
    ),
    toMetric(
      "perplexity",
      perplexity,
      "exp(loss)",
      "exp(languageModelLoss)",
      perplexity === null ? "loss too large to exponentiate meaningfully" : undefined,
    ),
    toMetric(
      "uniformLossBaseline",
      uniformLoss,
      "nats/token",
      "ln(vocabSize): the loss of a model with no information at all",
    ),
    toMetric("uniformPerplexityBaseline", uniformPerplexity, "probability", "vocabSize"),
    toMetric(
      "tokensPerSecond",
      first.tokensPerSecond,
      "tokens/s",
      `greedy decode of ${first.generatedTokens} token(s) from a ${determinismPrompt.length}-token prompt`,
    ),
    toMetric(
      "generationLatency",
      first.latencyMs,
      "ms",
      `wall clock for ${first.generatedTokens} generated token(s)`,
    ),
    toMetric(
      "repetitionRatio",
      repetitionOf(first.tokenIds),
      "fraction",
      "generated tokens equal to their immediate predecessor, over the deterministic decode",
    ),
    toMetric(
      "longestTokenRun",
      longestRun(first.tokenIds),
      "tokens",
      "longest run of one repeated token id in the deterministic decode",
    ),
    toMetric(
      "contextLengthUsed",
      Math.min(config.contextLength, first.promptTokens + first.generatedTokens),
      "tokens",
      "prompt + generated tokens actually consumed in the deterministic decode",
    ),
  ];

  const checks = [
    {
      name: "deterministic-generation",
      passed: deterministic,
      detail: deterministic
        ? `greedy decode repeated exactly: ${first.tokenIds.length} identical token id(s)`
        : "greedy decode produced different token ids on a second identical run",
    },
    {
      name: "tokenizer-round-trip",
      passed: roundTripOk,
      detail: roundTripOk
        ? "encode(decode(encode(text))) reproduced the original id sequence"
        : "re-encoding the decoded text did not reproduce the original ids",
    },
    {
      name: "generation-stop-conditions",
      passed: stoppedCleanly,
      detail: `stop reason "${first.stopReason}" after ${first.generatedTokens} token(s), cap ${maxNewTokens}`,
    },
    {
      name: "finite-loss",
      passed: Number.isFinite(lm.loss),
      detail: Number.isFinite(lm.loss)
        ? `cross-entropy ${lm.loss.toFixed(4)} nats/token over ${lm.tokens} predictions`
        : "cross-entropy was not finite — the model produced NaN or Inf logits",
    },
  ];

  return {
    modelName: config.name,
    modelVersion: config.version,
    configFingerprint: modelConfigFingerprint(config),
    parameterCount: model.parameterCount,
    tokenizerVersion: tokenizer.version,
    tokenizerFingerprint: tokenizer.fingerprint(),
    contextLength: config.contextLength,
    trainedTokens: options.trainedTokens ?? null,
    metrics,
    capabilities,
    capabilityCounts: {
      recall: {
        total: recallTotal.length,
        matched: recallTotal.filter((c) => c.matched).length,
      },
      heldOut: {
        total: heldOutTotal.length,
        matched: heldOutTotal.filter((c) => c.matched).length,
      },
    },
    checks,
    createdAt: options.now ?? Date.now(),
  };
}

/** Read one metric by name. */
export function metricValue(report: EvaluationReport, name: string): number | null {
  return report.metrics.find((m) => m.name === name)?.value ?? null;
}

/** Count of behavioural checks that passed. */
export function passedChecks(report: EvaluationReport): number {
  return report.checks.filter((c) => c.passed).length;
}

/** One-line summary: measurements only, no verdict on the model. */
export function summariseEvaluation(report: EvaluationReport): string {
  const loss = metricValue(report, "languageModelLoss");
  const ppl = metricValue(report, "perplexity");
  const tps = metricValue(report, "tokensPerSecond");
  return (
    `${report.modelName}@${report.modelVersion} · ${report.parameterCount.toLocaleString()} params · ` +
    `eval loss ${loss === null ? "n/a" : loss.toFixed(4)} · ` +
    `perplexity ${ppl === null ? "n/a" : ppl.toFixed(2)} · ` +
    `${tps ?? "n/a"} tok/s · ` +
    `recall ${report.capabilityCounts.recall.matched}/${report.capabilityCounts.recall.total} · ` +
    `held-out ${report.capabilityCounts.heldOut.matched}/${report.capabilityCounts.heldOut.total} · ` +
    `checks ${passedChecks(report)}/${report.checks.length}`
  );
}
