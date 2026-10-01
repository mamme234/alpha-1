/**
 * The capability gate.
 *
 * A model is not a capability milestone because its loss fell or because its
 * checkpoint exists. The gate states, in advance, which *measurable* things have
 * to improve, by how much, and how many of them have to move at once. It is
 * written here, before any Step 5 run has been measured, and it is not edited
 * afterwards.
 *
 * Three things it deliberately refuses to do:
 *
 *   1. **No composite score.** Criteria are counted, never weighted and summed.
 *      A weighted sum is an intelligence score wearing a lab coat, and it lets
 *      one large improvement hide a regression elsewhere.
 *
 *   2. **No silent threshold changes.** The criteria below are constants with a
 *      declared identity. `gateFingerprint` hashes them, and a gate evaluation
 *      records the fingerprint it used. Re-running against a different
 *      fingerprint is detectable, which is what makes "the bar did not move
 *      after the result was known" checkable rather than merely claimed.
 *
 *   3. **No verdict on intelligence.** Passing means "these measurements
 *      improved by at least this much". It does not mean the model is coherent,
 *      useful, or good, and `whatPassingDoesNotMean` says so in the output
 *      rather than in a footnote.
 *
 * Because thresholds are declared before measurement, they may turn out to be
 * unreachable for a model of this size. That is a legitimate outcome and is
 * reported as a failure, not recalibrated.
 */

import { AlphaValidationError } from "../core/errors";
import type { CapabilityReport } from "./runner";
import type { CapabilityComparisonRow } from "../training/experiments";

export type GateDirection = "lower" | "higher";

export type GateCriterion = {
  /** Stable id; part of the gate fingerprint. */
  id: string;
  /** The measurement it reads, named as it appears in the comparison. */
  measurement: string;
  /** What improvement looks like, in one line. */
  description: string;
  direction: GateDirection;
  /**
   * Minimum absolute improvement. For a fraction, 0.01 means one percentage
   * point. For a loss, 0.05 means 0.05 nats.
   */
  minAbsoluteImprovement: number;
  /**
   * Additional relative requirement. When present, *both* the absolute and the
   * relative bar must be met, so a tiny baseline cannot pass on absolute
   * movement alone.
   */
  minRelativeImprovement?: number;
  /** Which capability family this belongs to, so criteria can be counted as independent. */
  family: string;
};

/**
 * Step 5's criteria, declared before measurement.
 *
 * The families are deliberately spread across independent behaviours —
 * language modelling, comprehension, format control, and generation quality —
 * so that satisfying several cannot be done by getting better at one thing.
 */
export const CAPABILITY_GATE_CRITERIA: GateCriterion[] = [
  {
    id: "held-out-loss",
    measurement: "languageModeling.loss",
    description: "held-out language-modelling loss falls",
    direction: "lower",
    minAbsoluteImprovement: 0.05,
    family: "language-modeling",
  },
  {
    id: "held-out-perplexity",
    measurement: "languageModeling.perplexity",
    description: "held-out perplexity falls by at least 5%",
    direction: "lower",
    minAbsoluteImprovement: 0.5,
    minRelativeImprovement: 0.05,
    family: "language-modeling",
  },
  {
    id: "next-token-accuracy",
    measurement: "languageModeling.nextTokenTop1Accuracy",
    description: "exact next-token accuracy rises by at least one percentage point",
    direction: "higher",
    minAbsoluteImprovement: 0.01,
    family: "language-modeling",
  },
  {
    id: "question-answering",
    measurement: "question-answering.meanContinuationNll",
    description: "cross entropy on the expected answers to held-out questions falls",
    direction: "lower",
    minAbsoluteImprovement: 0.05,
    family: "comprehension",
  },
  {
    id: "context-retention",
    measurement: "context-retention.meanContinuationNll",
    description: "cross entropy on answers drawn from earlier context falls",
    direction: "lower",
    minAbsoluteImprovement: 0.05,
    family: "comprehension",
  },
  {
    id: "instruction-following",
    measurement: "instruction-following.meanContinuationNll",
    description: "cross entropy on expected responses to instructions falls",
    direction: "lower",
    minAbsoluteImprovement: 0.05,
    family: "instruction",
  },
  {
    id: "generation-repetition",
    measurement: "generation-quality.meanRepetitionRatio",
    description: "generation does not repeat itself more than the baseline",
    direction: "lower",
    // Zero tolerance in the direction that matters: this criterion passes if
    // repetition is unchanged or lower, and fails only if it got worse.
    minAbsoluteImprovement: 0,
    family: "generation-quality",
  },
  {
    id: "generation-diversity",
    measurement: "generation-quality.meanDistinctTrigramRatio",
    description: "generated text becomes more varied, not less",
    direction: "higher",
    minAbsoluteImprovement: 0,
    family: "generation-quality",
  },
];

/** How many independent criteria must be satisfied. Declared before measurement. */
export const GATE_REQUIRED_PASSES = 3;

/**
 * The fingerprint of the criteria themselves. Recorded with every gate
 * evaluation so a later reader can tell whether the bar moved.
 */
export function gateFingerprint(): string {
  let hash = 0x811c9dc5;
  const feed = (text: string) => {
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
  };
  feed(`required=${GATE_REQUIRED_PASSES}`);
  for (const criterion of CAPABILITY_GATE_CRITERIA) {
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

export type GateCriterionResult = {
  criterion: GateCriterion;
  baseline: number | null;
  candidate: number | null;
  delta: number | null;
  relative: number | null;
  absoluteImprovement: number | null;
  meetsAbsolute: boolean;
  meetsRelative: boolean;
  satisfied: boolean;
  detail: string;
};

export type GateEvaluation = {
  gateFingerprint: string;
  /** Fingerprint of the evaluation suite both sides were measured against. */
  suiteFingerprint: string;
  baselineSuiteFingerprint: string;
  candidateSuiteFingerprint: string;
  /** True only when both reports came from the same frozen suite. */
  suitesMatch: boolean;
  results: GateCriterionResult[];
  /** Count of satisfied criteria, over how many were evaluated at all. */
  satisfied: number;
  evaluated: number;
  /** Families represented by at least one satisfied criterion. */
  familiesImproved: string[];
  required: number;
  passed: boolean;
  /** Machine-readable reason for the verdict. */
  verdict: string;
  /** Stated plainly, so a pass is not over-read. */
  whatPassingDoesNotMean: string[];
  evaluatedAt: number;
};

/**
 * Read one measurement straight off a report, by the same names the comparison
 * table uses. Only the language-modelling trio is not read from a category, so
 * only those three need a direct accessor here.
 */
function measurementValue(report: CapabilityReport, measurement: string): number | null {
  switch (measurement) {
    case "languageModeling.loss":
      return report.languageModeling.loss;
    case "languageModeling.perplexity":
      return report.languageModeling.perplexity;
    case "languageModeling.nextTokenTop1Accuracy":
      return report.languageModeling.nextTokenTop1Accuracy;
    default: {
      const [categoryName, field] = measurement.split(".");
      const category = report.categories.find((c) => c.category === categoryName);
      if (!category) return null;
      switch (field) {
        case "meanContinuationNll":
          return category.meanContinuationNll;
        case "meanContinuationTop1":
          return category.meanContinuationTop1;
        case "meanRepetitionRatio":
          return category.meanRepetitionRatio;
        case "meanDistinctTrigramRatio":
          return category.meanDistinctTrigramRatio;
        case "passRate":
          return category.formatPassed && category.formatPassed.total > 0
            ? category.formatPassed.value / category.formatPassed.total
            : null;
        default:
          return null;
      }
    }
  }
}

/**
 * Evaluate the gate.
 *
 * `rows` come from `compareCapability`, so the gate reads exactly the numbers
 * the report prints. Nothing is recomputed here, and nothing is excluded: a
 * criterion whose measurement is missing on either side counts as *not*
 * satisfied rather than as not applicable, because a measurement that could not
 * be made is not evidence of improvement.
 */
export function evaluateGate(input: {
  baseline: CapabilityReport;
  candidate: CapabilityReport;
  comparison: CapabilityComparisonRow[];
  now?: number;
}): GateEvaluation {
  const { baseline, candidate, comparison } = input;

  const results: GateCriterionResult[] = CAPABILITY_GATE_CRITERIA.map((criterion) => {
    // Read from the same source the comparison table uses, so the gate and the
    // printed table can never disagree about what a number was.
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
    // "Improvement" is measured in the criterion's own direction, so a
    // lower-is-better criterion treats a negative delta as a gain.
    const absoluteImprovement = criterion.direction === "lower" ? -delta : delta;
    const relative = b === 0 ? null : absoluteImprovement / Math.abs(b);

    const meetsAbsolute = absoluteImprovement >= criterion.minAbsoluteImprovement;
    const meetsRelative =
      criterion.minRelativeImprovement === undefined ||
      relative === null ||
      relative >= criterion.minRelativeImprovement;

    const satisfied = meetsAbsolute && meetsRelative;
    const parts = [
      `${criterion.description}`,
      `baseline ${b.toFixed(5)} -> candidate ${c.toFixed(5)}`,
      `improvement ${absoluteImprovement.toFixed(5)}${relative === null ? "" : ` (${(relative * 100).toFixed(2)}%)`}`,
      `required ${criterion.minAbsoluteImprovement}${criterion.minRelativeImprovement === undefined ? "" : ` and ${(criterion.minRelativeImprovement * 100).toFixed(0)}%`}`,
      meetsAbsolute && meetsRelative ? "SATISFIED" : "not satisfied",
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

  const passed = suitesMatch && satisfiedRows.length >= GATE_REQUIRED_PASSES;

  let verdict: string;
  if (!suitesMatch) {
    verdict = `refused: the two reports used different evaluation suites (${baseline.suite.fingerprint} vs ${candidate.suite.fingerprint}), so the numbers are not comparable`;
  } else if (satisfiedRows.length >= GATE_REQUIRED_PASSES) {
    verdict = `passed: ${satisfiedRows.length} of ${evaluated} criteria satisfied (required ${GATE_REQUIRED_PASSES}) across ${families.length} independent families (${families.join(", ")})`;
  } else {
    verdict = `not met: ${satisfiedRows.length} of ${evaluated} criteria satisfied, below the required ${GATE_REQUIRED_PASSES}`;
  }

  return {
    gateFingerprint: gateFingerprint(),
    suiteFingerprint: candidate.suite.fingerprint,
    baselineSuiteFingerprint: baseline.suite.fingerprint,
    candidateSuiteFingerprint: candidate.suite.fingerprint,
    suitesMatch,
    results,
    satisfied: satisfiedRows.length,
    evaluated,
    familiesImproved: families,
    required: GATE_REQUIRED_PASSES,
    passed,
    verdict,
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

/** Refuse to evaluate a gate whose criteria have changed since they were declared. */
export function assertGateStable(evaluation: GateEvaluation, expectedFingerprint?: string): { ok: true } {
  const actual = gateFingerprint();
  if (evaluation.gateFingerprint !== actual) {
    throw new AlphaValidationError(
      "model",
      `capability gate criteria changed: evaluation used ${evaluation.gateFingerprint}, code now defines ${actual}`,
      { evaluation: evaluation.gateFingerprint, actual },
    );
  }
  if (expectedFingerprint && expectedFingerprint !== actual) {
    throw new AlphaValidationError(
      "model",
      `capability gate mismatch: expected ${expectedFingerprint}, found ${actual}; the bar moved between runs`,
      { expected: expectedFingerprint, actual },
    );
  }
  return { ok: true };
}

/** Printable gate report. Measurements and verdict, no score. */
export function describeGate(evaluation: GateEvaluation): string {
  const lines = [
    `Capability gate ${evaluation.passed ? "PASSED" : "NOT MET"} — ${evaluation.gateFingerprint}`,
    `  suite: ${evaluation.suiteFingerprint} (baseline ${evaluation.baselineSuiteFingerprint})`,
    `  ${evaluation.verdict}`,
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
