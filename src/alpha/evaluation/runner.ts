/**
 * The evaluation runner.
 *
 * This runs a frozen suite against a model and returns *raw results*: what the
 * model actually generated, what was measured, and `null` where a measurement
 * does not apply. There is no aggregation into a single score anywhere in this
 * file, and no case is skipped for producing a bad answer.
 *
 * Three kinds of measurement are made:
 *
 *   **Teacher-forced.** Given a prompt and an expected continuation, the model's
 *   own cross entropy on those tokens is computed directly. This is the most
 *   sensitive measurement available: it moves even when generation does not, and
 *   it cannot be gamed by a lucky argmax.
 *
 *   **Generated.** The text the model actually produces, kept verbatim. Format
 *   compliance, substring matches and list counts are computed from it.
 *
 *   **Structural.** Repetition and distinctness of the generation, which are
 *   properties of the text itself rather than of whether it happens to contain
 *   an expected phrase.
 *
 * Every measurement states its own method, so a number can be re-derived.
 */

import { AlphaValidationError } from "../core/errors";
import { setGradEnabled } from "../core/tensor";
import type { AlphaTransformer } from "../model/transformer";
import type { AlphaTokenizer } from "../tokenizer/bpe";
import {
  AlphaInferenceEngine,
  type GenerationResult,
  type SamplingConfig,
} from "../inference/engine";
import {
  EVAL_CATEGORIES,
  type EvalCase,
  type EvalCategory,
  type EvalSplit,
  type EvalSuite,
  assertSuiteFrozen,
} from "./suite";
import { words } from "../datasets/diversity";

/** Default generation budget. Small, because these are short-form cases. */
export const DEFAULT_EVAL_GENERATION_TOKENS = 48;

export type EvalCaseResult = {
  caseId: string;
  category: EvalCategory;
  split: EvalSplit;
  /** What the model was shown. */
  prompt: string;
  /** The expected continuation, when the case defines one. */
  expected: string | null;
  // --- generated -----------------------------------------------------------
  /** Verbatim generated text. Never replaced or cleaned. */
  generated: string;
  generatedTokens: number;
  stopReason: string;
  /** Substrings the case said would indicate a match, and whether each hit. */
  expect: Array<{ text: string; found: boolean }> | null;
  /** Any expected substring found. null when the case defines no expectations. */
  matched: boolean | null;
  // --- format --------------------------------------------------------------
  /** Format compliance, null when the case defines no format requirement. */
  formatPassed: boolean | null;
  formatDetails: string[];
  // --- teacher-forced ------------------------------------------------------
  /** Cross entropy in nats/token over the expected continuation. null if absent. */
  continuationNll: number | null;
  /** Fraction of continuation tokens the model's argmax got exactly right. */
  continuationTop1Accuracy: number | null;
  /** Tokens scored. */
  continuationTokens: number | null;
  // --- structural ----------------------------------------------------------
  /** Share of generated tokens equal to their immediate predecessor. */
  repetitionRatio: number;
  /** Longest run of one repeated token id. */
  longestTokenRun: number;
  /** Distinct 3-grams over total 3-grams in the generation. null if too short. */
  distinctTrigramRatio: number | null;
  // --- confusion -----------------------------------------------------------
  /**
   * For entity-tracking cases: whether the generation contains the entity the
   * case says the model must *not* confuse it with.
   */
  confusedWith: string | null;
  /** True when a confusion was detected. */
  confused: boolean;
  /** Wall clock for the generation, in milliseconds. */
  latencyMs: number;
};

export type CategoryReport = {
  category: EvalCategory;
  cases: number;
  /** Cases the case defined as needing a match, and how many matched. */
  matched: { total: number; value: number } | null;
  /** Cases that defined a format requirement, and how many passed. */
  formatPassed: { total: number; value: number } | null;
  /** Mean teacher-forced NLL over cases that define a continuation. */
  meanContinuationNll: number | null;
  /** Mean top-1 accuracy over the same cases. */
  meanContinuationTop1: number | null;
  /** Mean repetition over generated text. */
  meanRepetitionRatio: number;
  meanDistinctTrigramRatio: number | null;
  confusions: number;
  /** How many of the measurements above actually had a value. */
  measurementsAvailable: number;
};

export type CapabilityReport = {
  suite: { name: string; version: string; fingerprint: string; frozenAt: number };
  model: {
    name: string;
    version: string;
    parameters: number;
    trainedTokens: number | null;
    tokenizerVersion: string;
    tokenizerFingerprint: string;
  };
  /** Every case result, in suite order. Raw. */
  cases: EvalCaseResult[];
  /** Per-category reports. Separate, never summed. */
  categories: CategoryReport[];
  /**
   * Counts only, split by whether the material was held out. These are two
   * numbers, deliberately not one.
   */
  counts: {
    heldOut: { total: number; matched: number };
    known: { total: number; matched: number };
    format: { total: number; passed: number };
    confusion: { total: number; confusions: number };
  };
  /** Language-modelling measurements over the held-out documents. */
  languageModeling: {
    loss: number | null;
    perplexity: number | null;
    /** Exact next-token argmax accuracy over every scored position. */
    nextTokenTop1Accuracy: number | null;
    positions: number;
    uniformLoss: number;
    tokens: number;
  };
  /** Method strings, so every number above can be re-derived. */
  methods: string[];
  createdAt: number;
};

export type RunSuiteOptions = {
  /** Documents for language-modelling measurement. Must be held out. */
  heldOutDocuments: string[];
  sampling?: Partial<SamplingConfig>;
  maxNewTokens?: number;
  /** Set when the weights came from a real run. */
  trainedTokens?: number | null;
  now?: number;
  /** Refuse to run unless the suite matches this fingerprint. */
  expectedFingerprint?: string;
};

/** Fraction of generated tokens equal to their immediate predecessor. */
function repetitionRatio(tokenIds: number[]): number {
  if (tokenIds.length < 2) return 0;
  let same = 0;
  for (let i = 1; i < tokenIds.length; i++) if (tokenIds[i] === tokenIds[i - 1]) same++;
  return same / (tokenIds.length - 1);
}

function longestRun(tokenIds: number[]): number {
  let best = 0;
  let current = 0;
  let previous = -1;
  for (const id of tokenIds) {
    current = id === previous ? current + 1 : 1;
    previous = id;
    if (current > best) best = current;
  }
  return best;
}

function distinctTrigramRatio(text: string): number | null {
  const tokens = words(text);
  if (tokens.length < 3) return null;
  const trigrams = new Set<string>();
  for (let i = 0; i + 3 <= tokens.length; i++) trigrams.add(tokens.slice(i, i + 3).join(" "));
  const total = tokens.length - 2;
  return total === 0 ? null : trigrams.size / total;
}

/** Tokens per line that start with a list marker. */
function listItemCount(text: string): number {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(
      (line) =>
        /^[-*•]\s/.test(line) || /^\d+[.)]\s/.test(line) || /^\d+:\s/.test(line),
    ).length;
}

/**
 * Check a format requirement against generated text.
 *
 * Each requirement is evaluated independently and reported separately, so a case
 * that failed on length alone says so rather than reporting a bare false.
 */
export function checkFormat(
  text: string,
  requirement: NonNullable<EvalCase["format"]>,
): { passed: boolean; details: string[] } {
  const details: string[] = [];
  let passed = true;

  for (const needle of requirement.mustContain ?? []) {
    const ok = text.toLowerCase().includes(needle.toLowerCase());
    if (!ok) passed = false;
    details.push(`contains "${needle}": ${ok ? "yes" : "no"}`);
  }

  if (requirement.json) {
    const candidate = extractJson(text);
    if (candidate === null) {
      passed = false;
      details.push("parses as JSON: no");
    } else if (typeof candidate !== "object" || candidate === null || Array.isArray(candidate)) {
      passed = false;
      details.push(
        `parses as JSON: yes, but produced ${Array.isArray(candidate) ? "an array" : typeof candidate} rather than the requested shape`,
      );
    } else {
      const record = candidate as Record<string, unknown>;
      details.push("parses as JSON: yes");
      for (const key of requirement.json.keys ?? []) {
        const present = Object.prototype.hasOwnProperty.call(record, key);
        if (!present) passed = false;
        details.push(`has key "${key}": ${present ? "yes" : "no"}`);
      }
      for (const [key, type] of Object.entries(requirement.json.types ?? {})) {
        if (!Object.prototype.hasOwnProperty.call(record, key)) {
          details.push(`type of "${key}": missing`);
          passed = false;
          continue;
        }
        const value = record[key];
        const actual = Array.isArray(value)
          ? "array"
          : value === null
            ? "null"
            : typeof value === "string" ||
                typeof value === "number" ||
                typeof value === "boolean"
              ? typeof value
              : "object";
        const ok =
          actual === type ||
          // A JSON number arriving as a string is a type error we report, but a
          // string requested as a string is a pass.
          (type === "object" && actual === "object");
        if (!ok) passed = false;
        details.push(`type of "${key}": ${actual} (want ${type})`);
      }
    }
  }

  if (requirement.minListItems !== undefined) {
    const count = listItemCount(text);
    const ok = count >= requirement.minListItems;
    if (!ok) passed = false;
    details.push(`list items: ${count} (need ${requirement.minListItems})`);
  }

  if (requirement.minDistinctWords !== undefined) {
    const distinct = new Set(words(text)).size;
    const ok = distinct >= requirement.minDistinctWords;
    if (!ok) passed = false;
    details.push(`distinct words: ${distinct} (need ${requirement.minDistinctWords})`);
  }

  if (requirement.minCharacters !== undefined) {
    const ok = text.length >= requirement.minCharacters;
    if (!ok) passed = false;
    details.push(`length: ${text.length} chars (min ${requirement.minCharacters})`);
  }
  if (requirement.maxCharacters !== undefined) {
    const ok = text.length <= requirement.maxCharacters;
    if (!ok) passed = false;
    details.push(`length: ${text.length} chars (max ${requirement.maxCharacters})`);
  }

  return { passed, details };
}

/** Pull the first balanced JSON object or array out of generated text. */
function extractJson(text: string): unknown {
  const starts: number[] = [];
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === "{" || ch === "[") starts.push(i);
    if (ch !== "}" && ch !== "]") continue;
    const open = starts.pop();
    if (open === undefined) continue;
    const candidate = text.slice(open, i + 1);
    try {
      return JSON.parse(candidate);
    } catch {
      // Not this pair; keep scanning outward.
    }
  }
  // A common failure mode is generating a bare object without a full parse of
  // nested pairs; try the whole text as a last resort.
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/**
 * Teacher-forced cross entropy of the expected continuation given the prompt.
 *
 * Computed directly from the model's own logits, exactly as `languageModelLoss`
 * does in `./framework`, but restricted to the continuation so the number answers
 * "how much did the model expect this specific answer".
 */
function continuationScoring(
  model: AlphaTransformer,
  tokenizer: AlphaTokenizer,
  prompt: string,
  continuation: string,
): { nll: number; top1: number; tokens: number } | null {
  if (continuation.trim().length === 0) return null;
  const promptIds = tokenizer.encode(prompt);
  const continuationIds = tokenizer.encode(continuation);
  if (continuationIds.length === 0) return null;
  const context = model.config.contextLength;
  // Window the pair so it always fits, keeping the last token of the window as
  // the prediction target where possible.
  const available = Math.max(1, context - 1);
  const promptWindow = promptIds.slice(-available);
  const ids = [...promptWindow, ...continuationIds];
  if (ids.length < 2) return null;
  if (ids.length > context) {
    // Too long even after trimming: score the tail of the continuation only,
    // and say so by scoring fewer tokens rather than truncating silently.
    const keep = context;
    const tail = ids.slice(ids.length - keep);
    return scoreIds(model, tail, tail.length - continuationIds.length);
  }
  return scoreIds(model, ids, promptWindow.length);
}

function scoreIds(
  model: AlphaTransformer,
  ids: number[],
  continuationStart: number,
): { nll: number; top1: number; tokens: number } | null {
  if (ids.length < 2) return null;
  const vocab = model.config.vocabSize;
  const input = Int32Array.from(ids.slice(0, -1));
  const targets = ids.slice(1);
  setGradEnabled(false);
  let total = 0;
  let top1 = 0;
  let scored = 0;
  try {
    const out = model.forward(input, 1, input.length, { training: false });
    for (let t = 0; t < targets.length; t++) {
      if (t < continuationStart - 1) continue;
      const base = t * vocab;
      const target = targets[t];
      let max = -Infinity;
      let argmax = 0;
      for (let v = 0; v < vocab; v++) {
        const value = out.logits.data[base + v];
        if (value > max) {
          max = value;
          argmax = v;
        }
      }
      let sum = 0;
      for (let v = 0; v < vocab; v++) sum += Math.exp(out.logits.data[base + v] - max);
      total += -(out.logits.data[base + target] - max - Math.log(sum));
      if (argmax === target) top1 += 1;
      scored += 1;
    }
  } finally {
    setGradEnabled(true);
  }
  if (scored === 0) return null;
  return { nll: total / scored, top1: top1 / scored, tokens: scored };
}

/**
 * Language-modelling metrics over held-out documents: cross entropy, perplexity
 * and exact next-token accuracy.
 */
export function languageModelingMetrics(
  model: AlphaTransformer,
  tokenizer: AlphaTokenizer,
  documents: string[],
): CapabilityReport["languageModeling"] {
  const vocab = model.config.vocabSize;
  const uniformLoss = Math.log(vocab);
  const seqLen = Math.min(model.config.contextLength, 32);
  let lossSum = 0;
  let tokens = 0;
  let top1 = 0;
  let positions = 0;

  for (const document of documents) {
    const ids = tokenizer.encode(document);
    if (ids.length < 2) continue;
    for (let start = 0; start + 1 < ids.length; start += seqLen - 1) {
      const window = ids.slice(start, Math.min(ids.length, start + seqLen));
      if (window.length < 2) break;
      const input = Int32Array.from(window.slice(0, -1));
      const targets = window.slice(1);
      setGradEnabled(false);
      try {
        const out = model.forward(input, 1, input.length, { training: false });
        for (let t = 0; t < targets.length; t++) {
          const base = t * vocab;
          const target = targets[t];
          let max = -Infinity;
          let argmax = 0;
          for (let v = 0; v < vocab; v++) {
            const value = out.logits.data[base + v];
            if (value > max) {
              max = value;
              argmax = v;
            }
          }
          let sum = 0;
          for (let v = 0; v < vocab; v++) sum += Math.exp(out.logits.data[base + v] - max);
          lossSum += -(out.logits.data[base + target] - max - Math.log(sum));
          tokens += 1;
          positions += 1;
          if (argmax === target) top1 += 1;
        }
      } finally {
        setGradEnabled(true);
      }
    }
  }

  const loss = tokens === 0 ? null : lossSum / tokens;
  const perplexity =
    loss === null ? null : Math.exp(Math.min(loss, 20)) <= 1e12 ? Math.exp(Math.min(loss, 20)) : null;

  return {
    loss,
    perplexity,
    nextTokenTop1Accuracy: positions === 0 ? null : top1 / positions,
    positions,
    uniformLoss,
    tokens,
  };
}

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function ratio(m: { total: number; value: number } | null): number | null {
  return m && m.total > 0 ? m.value / m.total : null;
}

/**
 * Run the suite.
 *
 * Every case runs. A case that produces nothing is recorded with an empty
 * generation and its measurements null — it is never dropped from the report.
 */
export function runEvalSuite(
  model: AlphaTransformer,
  tokenizer: AlphaTokenizer,
  suite: EvalSuite,
  options: RunSuiteOptions,
): CapabilityReport {
  assertSuiteFrozen(suite, options.expectedFingerprint);
  if (options.heldOutDocuments.length === 0) {
    throw new AlphaValidationError(
      "evaluation",
      "language-modeling measurement needs at least one held-out document; an empty set cannot produce a loss",
    );
  }

  const engine = new AlphaInferenceEngine({ model, tokenizer, stage: "trained" });
  const maxNewTokens = options.maxNewTokens ?? DEFAULT_EVAL_GENERATION_TOKENS;
  const sampling: Partial<SamplingConfig> = {
    temperature: 0,
    maxNewTokens,
    ...options.sampling,
  };

  const results: EvalCaseResult[] = [];

  for (const evalCase of suite.cases) {
    const generation: GenerationResult = engine.generate(evalCase.prompt, sampling);
    const generated = generation.text;
    const generatedTokens = generation.tokenIds;

    const expect =
      evalCase.expect && evalCase.expect.length > 0
        ? evalCase.expect.map((needle) => ({
            text: needle,
            found: generated.toLowerCase().includes(needle.toLowerCase()),
          }))
        : null;
    const matched = expect ? expect.some((entry) => entry.found) : null;

    const formatPassed = evalCase.format
      ? checkFormat(generated, evalCase.format).passed
      : null;
    const formatDetails = evalCase.format
      ? checkFormat(generated, evalCase.format).details
      : [];

    const scoring = evalCase.continuation
      ? continuationScoring(model, tokenizer, evalCase.prompt, evalCase.continuation)
      : null;

    // Entity-confusion detection: the suite records what the answer must not be.
    const confusable = CONFUSABLE.get(evalCase.id);
    const confused = confusable
      ? generated.toLowerCase().includes(confusable.toLowerCase())
      : false;

    results.push({
      caseId: evalCase.id,
      category: evalCase.category,
      split: evalCase.split,
      prompt: evalCase.prompt,
      expected: evalCase.continuation ?? null,
      generated,
      generatedTokens: generation.generatedTokens,
      stopReason: generation.stopReason,
      expect,
      matched,
      formatPassed,
      formatDetails,
      continuationNll: scoring ? scoring.nll : null,
      continuationTop1Accuracy: scoring ? scoring.top1 : null,
      continuationTokens: scoring ? scoring.tokens : null,
      repetitionRatio: repetitionRatio(generatedTokens),
      longestTokenRun: longestRun(generatedTokens),
      distinctTrigramRatio: distinctTrigramRatio(generated),
      confusedWith: confusable ?? null,
      confused,
      latencyMs: generation.latencyMs,
    });
  }

  // Per-category reports, computed only from the measurements that apply.
  const categories: CategoryReport[] = EVAL_CATEGORIES.map((category) => {
    const own = results.filter((r) => r.category === category);
    const matchedCases = own.filter((r) => r.matched !== null);
    const formatCases = own.filter((r) => r.formatPassed !== null);
    const nlls = own.filter((r) => r.continuationNll !== null);
    const distincts = own.filter((r) => r.distinctTrigramRatio !== null);

    const matchedReport = matchedCases.length
      ? { total: matchedCases.length, value: matchedCases.filter((r) => r.matched === true).length }
      : null;
    const formatReport = formatCases.length
      ? {
          total: formatCases.length,
          value: formatCases.filter((r) => r.formatPassed === true).length,
        }
      : null;

    return {
      category,
      cases: own.length,
      matched: matchedReport,
      formatPassed: formatReport,
      meanContinuationNll: mean(nlls.map((r) => r.continuationNll!)),
      meanContinuationTop1: mean(nlls.map((r) => r.continuationTop1Accuracy!)),
      meanRepetitionRatio: mean(own.map((r) => r.repetitionRatio)) ?? 0,
      meanDistinctTrigramRatio: mean(distincts.map((r) => r.distinctTrigramRatio!)),
      confusions: own.filter((r) => r.confused).length,
      measurementsAvailable:
        (matchedReport ? 1 : 0) +
        (formatReport ? 1 : 0) +
        (nlls.length > 0 ? 2 : 0) +
        (distincts.length > 0 ? 1 : 0),
    };
  });

  const heldOutCases = results.filter((r) => r.split === "held-out" && r.matched !== null);
  const knownCases = results.filter((r) => r.split === "known" && r.matched !== null);
  const formatCases = results.filter((r) => r.formatPassed !== null);
  const confusionCases = results.filter((r) => r.confusedWith !== null);

  const lm = languageModelingMetrics(model, tokenizer, options.heldOutDocuments);

  return {
    suite: {
      name: suite.name,
      version: suite.version,
      fingerprint: suite.fingerprint,
      frozenAt: suite.frozenAt,
    },
    model: {
      name: model.config.name,
      version: model.config.version,
      parameters: model.parameterCount,
      trainedTokens: options.trainedTokens ?? null,
      tokenizerVersion: tokenizer.version,
      tokenizerFingerprint: tokenizer.fingerprint(),
    },
    cases: results,
    categories,
    counts: {
      heldOut: {
        total: heldOutCases.length,
        matched: heldOutCases.filter((r) => r.matched === true).length,
      },
      known: {
        total: knownCases.length,
        matched: knownCases.filter((r) => r.matched === true).length,
      },
      format: {
        total: formatCases.length,
        passed: formatCases.filter((r) => r.formatPassed === true).length,
      },
      confusion: {
        total: confusionCases.length,
        confusions: confusionCases.filter((r) => r.confused).length,
      },
    },
    languageModeling: lm,
    methods: [
      "continuationNll: cross entropy of the model's own logits over the expected continuation tokens only, given the prompt as context, computed by teacher forcing over a window of at most model.contextLength tokens",
      "continuationTop1Accuracy: fraction of those continuation positions where the model's argmax equals the expected token",
      "languageModeling.loss: cross entropy over every next-token position in the held-out documents, windowed at min(contextLength, 32) tokens",
      "languageModeling.nextTokenTop1Accuracy: fraction of positions where the argmax equals the actual next token",
      "matched: any case.expect substring present in the generated text, case-insensitive",
      "formatPassed: every stated format requirement satisfied; each requirement reported individually in formatDetails",
      "repetitionRatio: generated tokens equal to their immediate predecessor, over generated tokens - 1",
      "distinctTrigramRatio: distinct word 3-grams over total word 3-grams in the generated text",
      "confused: the generated text contains the entity the case says must not be confused with the answer",
      "All generation is greedy (temperature 0), so a repeated run reproduces these numbers exactly.",
    ],
    createdAt: options.now ?? Date.now(),
  };
}

/**
 * Entities a case must not be confused with, keyed by case id.
 *
 * Kept beside the suite rather than inside it so the case text stays exactly
 * what was frozen: adding a field to `EvalCase` would change nothing about the
 * measured text, but keeping it here makes it obvious that a confusion is an
 * extra measurement, not an extra expectation.
 */
const CONFUSABLE: Map<string, string> = new Map([
  ["ent-1", "salt"],
  ["ent-2", "Rune"],
]);

/** Category report by name. */
export function categoryReport(report: CapabilityReport, category: EvalCategory): CategoryReport {
  const found = report.categories.find((c) => c.category === category);
  if (!found) {
    throw new AlphaValidationError("evaluation", `no report for category ${category}`);
  }
  return found;
}

/** All generated text, for inspecting what the model actually produced. */
export function generations(report: CapabilityReport): Array<{ id: string; text: string }> {
  return report.cases.map((result) => ({ id: result.caseId, text: result.generated }));
}

/** One-line, measurement-only summary. No verdict. */
export function summariseCapabilityReport(report: CapabilityReport): string {
  const lm = report.languageModeling;
  return (
    `${report.model.name}@${report.model.version} · ${report.model.parameters.toLocaleString()} params · ` +
    `lm loss ${lm.loss === null ? "n/a" : lm.loss.toFixed(4)} · ` +
    `ppl ${lm.perplexity === null ? "n/a" : lm.perplexity.toFixed(2)} · ` +
    `next-token top-1 ${lm.nextTokenTop1Accuracy === null ? "n/a" : (lm.nextTokenTop1Accuracy * 100).toFixed(2)}% · ` +
    `substring held-out ${report.counts.heldOut.matched}/${report.counts.heldOut.total} · ` +
    `known ${report.counts.known.matched}/${report.counts.known.total} · ` +
    `format ${report.counts.format.passed}/${report.counts.format.total} · ` +
    `confusions ${report.counts.confusion.confusions}/${report.counts.confusion.total}`
  );
}
