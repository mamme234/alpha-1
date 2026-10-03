/**
 * Step 7's capability gate — declared before measurement.
 *
 * Written before the Step 7 candidate was trained, and not edited afterwards.
 * `step7GateFingerprint()` is printed by `alpha:verify-intelligence`, so a later
 * reader can prove which bar was in force while the numbers were produced.
 *
 * Step 5's gate (`./gate`) is untouched; this module adds a stricter bar for
 * the larger model:
 *
 *   - three criteria are hard requirements: held-out loss must fall, held-out
 *     perplexity must fall by at least 10%, and generation must not repeat
 *     itself more than the baseline. A candidate that does not move the
 *     language-modelling numbers does not pass, however many small wins it has;
 *   - seven of ten criteria must be satisfied, and the families are spread
 *     across language modelling, comprehension, instruction, extension
 *     (mathematics) and generation quality, so passing cannot come from
 *     improving one behaviour alone;
 *   - every criterion compares against the Step 5 production baseline measured
 *     in the same run on the same frozen suite.
 */

import {
  measurementValue,
  type GateCriterion,
  type GateCriterionResult,
  type GateDirection,
  type GateEvaluation,
} from "./gate";
import type { CapabilityComparisonRow } from "../training/experiments";
import type { CapabilityReport } from "./runner";

/** Re-exported so callers do not need two imports. */
export type { GateCriterion, GateDirection };

type Step7Criterion = GateCriterion;

const CRITERIA: Step7Criterion[] = [
  {
    id: "held-out-loss",
    measurement: "languageModeling.loss",
    description: "frozen held-out language-modelling loss falls by at least 0.10 nats",
    direction: "lower",
    minAbsoluteImprovement: 0.1,
    family: "language-modeling",
  },
  {
    id: "held-out-perplexity",
    measurement: "languageModeling.perplexity",
    description: "frozen held-out perplexity falls by at least 10% (and 0.5 in absolute terms)",
    direction: "lower",
    minAbsoluteImprovement: 0.5,
    minRelativeImprovement: 0.1,
    family: "language-modeling",
  },
  {
    id: "next-token-accuracy",
    measurement: "languageModeling.nextTokenTop1Accuracy",
    description: "exact next-token accuracy rises by at least two percentage points",
    direction: "higher",
    minAbsoluteImprovement: 0.02,
    family: "language-modeling",
  },
  {
    id: "completion",
    measurement: "completion.meanContinuationNll",
    description: "cross entropy on expected completions falls by at least 0.05 nats",
    direction: "lower",
    minAbsoluteImprovement: 0.05,
    family: "comprehension",
  },
  {
    id: "question-answering",
    measurement: "question-answering.meanContinuationNll",
    description: "cross entropy on expected answers to held-out questions falls by 0.05 nats",
    direction: "lower",
    minAbsoluteImprovement: 0.05,
    family: "comprehension",
  },
  {
    id: "context-retention",
    measurement: "context-retention.meanContinuationNll",
    description: "cross entropy on answers drawn from earlier context falls by 0.05 nats",
    direction: "lower",
    minAbsoluteImprovement: 0.05,
    family: "comprehension",
  },
  {
    id: "instruction-following",
    measurement: "instruction-following.meanContinuationNll",
    description: "cross entropy on expected responses to instructions falls by 0.05 nats",
    direction: "lower",
    minAbsoluteImprovement: 0.05,
    family: "instruction",
  },
  {
    id: "mathematics",
    measurement: "mathematics.meanContinuationNll",
    description: "cross entropy on expected arithmetic and reasoning steps falls by 0.05 nats",
    direction: "lower",
    minAbsoluteImprovement: 0.05,
    family: "extension",
  },
  {
    id: "generation-repetition",
    measurement: "generation-quality.meanRepetitionRatio",
    description: "generation does not repeat itself more than the baseline",
    direction: "lower",
    minAbsoluteImprovement: 0,
    family: "generation-quality",
  },
  {
    id: "generation-diversity",
    measurement: "generation-quality.meanDistinctTrigramRatio",
    description: "generated text is no less varied than the baseline",
    direction: "higher",
    minAbsoluteImprovement: 0,
    family: "generation-quality",
  },
];

/** Exposed so the report can print the criteria without re-declaring them. */
export const STEP7_GATE_CRITERIA: readonly Step7Criterion[] = CRITERIA;

/** How many of the ten criteria must be satisfied. Declared before measurement. */
export const STEP7_GATE_REQUIRED_PASSES = 7;

/**
 * Criteria that must individually pass. A candidate that fails one of these
 * fails the gate regardless of how many other criteria it satisfies.
 */
export const STEP7_GATE_MUST_PASS: readonly string[] = [
  "held-out-loss",
  "held-out-perplexity",
  "generation-repetition",
];

/** FNV-1a over the criteria themselves — the same scheme Step 5's gate uses. */
export function step7GateFingerprint(): string {
  let hash = 0x811c9dc5;
  const feed = (text: string) => {
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
  };
  feed(`required=${STEP7_GATE_REQUIRED_PASSES}`);
  feed(`must=${[...STEP7_GATE_MUST_PASS].sort().join(",")}`);
  feed("\u0001");
  for (const criterion of CRITERIA) {
    feed(
      [
        criterion.id,
        criterion.measurement,
        criterion.direction,
        String(criterion.minAbsoluteImprovement),
        String(criterion.minRelativeImprovement ?? ""),
        criterion.family,
      ].join("|"),
    );
    feed("\u0001");
  }
  return `gate_${hash.toString(16).padStart(8, "0")}`;
}

type Step7EvaluationExtras = {
  /** Criteria the gate requires individually, that were not satisfied. */
  mustPassMissing: string[];
  mustPass: readonly string[];
};

export type Step7GateEvaluation = GateEvaluation & Step7EvaluationExtras;

/**
 * Evaluate the Step 7 gate against the same frozen-suite reports the comparison
 * table prints. A measurement missing on either side counts as not satisfied,
 * never as not applicable: an unmeasurable number is not improvement.
 */
export function evaluateStep7Gate(input: {
  baseline: CapabilityReport;
  candidate: CapabilityReport;
  comparison: CapabilityComparisonRow[];
  now?: number;
}): Step7GateEvaluation {
  const { baseline, candidate } = input;

  const results: GateCriterionResult[] = CRITERIA.map((criterion) => {
    const b = measurementValue(baseline, criterion.measurement);
    const c = measurementValue(candidate, criterion.measurement);
    if (b === null || c === null) {
      return {
        criterion,
        baseline: b,
        candidate: c,
        delta: null,
        relative: null,
        absoluteImprovement: null,
        meetsAbsolute: false,
        meetsRelative: false,
        satisfied: false,
        detail:
          b === null
            ? `not evaluated: the baseline produced no value for ${criterion.measurement}`
            : `not evaluated: the candidate produced no value for ${criterion.measurement}`,
      };
    }
    const delta = c - b;
    const absoluteImprovement = criterion.direction === "lower" ? -delta : delta;
    const relative = b === 0 ? null : absoluteImprovement / Math.abs(b);
    const meetsAbsolute = absoluteImprovement >= criterion.minAbsoluteImprovement;
    const meetsRelative =
      criterion.minRelativeImprovement === undefined ||
      relative === null ||
      relative >= criterion.minRelativeImprovement;
    const satisfied = meetsAbsolute && meetsRelative;
    const parts = [
      criterion.description,
      `baseline ${b.toFixed(5)} -> candidate ${c.toFixed(5)}`,
      `improvement ${absoluteImprovement.toFixed(5)}${relative === null ? "" : ` (${(relative * 100).toFixed(2)}%)`}`,
      `required ${criterion.minAbsoluteImprovement}${criterion.minRelativeImprovement === undefined ? "" : ` and ${(criterion.minRelativeImprovement * 100).toFixed(0)}%`}`,
      satisfied ? "SATISFIED" : "not satisfied",
    ];
    return {
      criterion,
      baseline: b,
      candidate: c,
      delta,
      relative,
      absoluteImprovement,
      meetsAbsolute,
      meetsRelative,
      satisfied,
      detail: parts.join(" · "),
    };
  });

  const evaluated = results.length;
  const satisfiedRows = results.filter((r) => r.satisfied);
  const families = [...new Set(satisfiedRows.map((r) => r.criterion.family))];
  const suitesMatch = baseline.suite.fingerprint === candidate.suite.fingerprint;
  const mustPassMissing = STEP7_GATE_MUST_PASS.filter(
    (id) => !results.some((row) => row.criterion.id === id && row.satisfied),
  );
  const passed =
    suitesMatch && satisfiedRows.length >= STEP7_GATE_REQUIRED_PASSES && mustPassMissing.length === 0;

  let verdict: string;
  if (!suitesMatch) {
    verdict = `refused: the two reports used different evaluation suites (${baseline.suite.fingerprint} vs ${candidate.suite.fingerprint}), so the numbers are not comparable`;
  } else if (mustPassMissing.length > 0) {
    verdict = `not met: required criteria failed: ${mustPassMissing.join(", ")} — these must pass whatever else improved`;
  } else if (satisfiedRows.length >= STEP7_GATE_REQUIRED_PASSES) {
    verdict = `passed: ${satisfiedRows.length} of ${evaluated} criteria satisfied (required ${STEP7_GATE_REQUIRED_PASSES}) across ${families.length} independent families (${families.join(", ")})`;
  } else {
    verdict = `not met: ${satisfiedRows.length} of ${evaluated} criteria satisfied, below the required ${STEP7_GATE_REQUIRED_PASSES}`;
  }

  return {
    gateFingerprint: step7GateFingerprint(),
    suiteFingerprint: candidate.suite.fingerprint,
    baselineSuiteFingerprint: baseline.suite.fingerprint,
    candidateSuiteFingerprint: candidate.suite.fingerprint,
    suitesMatch,
    results,
    satisfied: satisfiedRows.length,
    evaluated,
    familiesImproved: families,
    required: STEP7_GATE_REQUIRED_PASSES,
    passed,
    verdict,
    mustPassMissing,
    mustPass: STEP7_GATE_MUST_PASS,
    whatPassingDoesNotMean: [
      "It does not mean the model produces coherent general-purpose language.",
      "It does not mean the model is useful, safe, or ready to serve anyone.",
      "It does not mean the model understands anything; it means specific numbers moved.",
      "It does not mean the measurement suite is a benchmark of intelligence.",
      "It does not authorise promotion to PRODUCTION, which stays an explicit operator decision.",
    ],
    evaluatedAt: input.now ?? Date.now(),
  };
}

/** Refuse to evaluate this gate when its criteria have changed since declaration. */
export function assertStep7GateStable(
  evaluation: Step7GateEvaluation,
  expectedFingerprint?: string,
): { ok: true } {
  const actual = step7GateFingerprint();
  if (evaluation.gateFingerprint !== actual) {
    throw new Error(
      `step 7 gate criteria changed: evaluation used ${evaluation.gateFingerprint}, code now defines ${actual}`,
    );
  }
  if (expectedFingerprint && expectedFingerprint !== actual) {
    throw new Error(
      `step 7 gate mismatch: expected ${expectedFingerprint}, found ${actual}; the bar moved between runs`,
    );
  }
  return { ok: true };
}

/** Printable gate report. Measurements and verdict, no score. */
export function describeStep7Gate(evaluation: Step7GateEvaluation): string {
  const lines = [
    `Step 7 capability gate ${evaluation.passed ? "PASSED" : "NOT MET"} — ${evaluation.gateFingerprint}`,
    `  suite: ${evaluation.suiteFingerprint} (baseline ${evaluation.baselineSuiteFingerprint})`,
    `  ${evaluation.verdict}`,
    `  must pass: ${[...evaluation.mustPass].join(", ")}`,
    "",
    "  criteria (counted, never summed):",
  ];
  for (const result of evaluation.results) {
    lines.push(`    ${result.satisfied ? "[x]" : "[ ]"} ${result.criterion.id}: ${result.detail}`);
  }
  lines.push("");
  lines.push("  passing does not mean:");
  for (const note of evaluation.whatPassingDoesNotMean) lines.push(`    - ${note}`);
  return lines.join("\n");
}
