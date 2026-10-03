/**
 * Alpha's Step 7 evaluation extension.
 *
 * Mathematics, reasoning and coding cases for the Step 7 model, measured on a
 * SEPARATE frozen suite so that the production capability suite (and Step 5's
 * recorded fingerprint `evl_b58eacf7`) is never rewritten by Step 7 work.
 *
 * The three categories are declared in the shared `EVAL_CATEGORIES` so the
 * runner maps them; the case text below lives only here.
 *
 * Rules:
 *
 *   1. **Frozen before measurement.** `createEvalStep7Suite({ now, freeze: true })`
 *      is called once by `alpha:verify-intelligence`, before either arm is
 *      measured, and `assertSuiteFrozen` re-hashes the cases. Any edit after
 *      measurement changes the fingerprint.
 *   2. **Context in prompt.** Every comprehension-style case supplies its
 *      material in the prompt so the answer is derivable rather than recalled.
 *      That keeps these tests of *capability* rather than training-data recall.
 *   3. **Audit clean.** The case text is checked against BOTH arms' training
 *      data (overlap threshold 0.4 shingle coverage) before it is measured,
 *      exactly like the production suite. No case may overlap training.
 *   4. **Never in the gate's main suite.** These cases are reported as the
 *      step-7 extension report; the 10-step-7-gate criteria deliberately read
 *      only the measurements from the frozen extension suite.
 */

import { AlphaValidationError } from "../core/errors";
import { detectOverlap } from "../datasets/splits";
import { words } from "../datasets/diversity";
import { suiteFingerprint } from "./suite";

export const STEP7_EXTENSION_SUITE_VERSION = "1.0.0";

/** The categories this extension adds, kept separate throughout. */
export const STEP7_EXT_CATEGORIES = ["mathematics", "reasoning", "coding"] as const;

export type Step7ExtCategory = (typeof STEP7_EXT_CATEGORIES)[number];

export type Step7ExtCase = {
  id: string;
  category: Step7ExtCategory;
  split: "held-out";
  /** What the model sees. */
  prompt: string;
  /** Teacher-forced continuation for cross-entropy measurement. */
  continuation?: string;
  /** Substrings whose presence indicates a match (generated checks). */
  expect?: string[];
  /** Format requirement when the case checks format. */
  format?: { minCharacters?: number; maxCharacters?: number };
  language: string;
  measures: string;
};

// ---------------------------------------------------------------------------
// Mathematics cases
// ---------------------------------------------------------------------------

const MATHEMATICS_CASES: Step7ExtCase[] = [
  {
    id: "math-1",
    category: "mathematics",
    split: "held-out",
    prompt:
      "What is 37 multiplied by 28? Work it out step by step, then give the final number only.\nAnswer:",
    continuation: "1036",
    measures: "multi-digit multiplication with a step-by-step worked solution",
    language: "en",
  },
  {
    id: "math-2",
    category: "mathematics",
    split: "held-out",
    prompt:
      "A box contains 14 rows of 15 cans each. 3 jars of 4 cans are removed. How many cans remain?\nAnswer:",
    continuation: "201",
    measures: "multi-step arithmetic with a real-world context",
    language: "en",
  },
  {
    id: "math-3",
    category: "mathematics",
    split: "held-out",
    prompt:
      "What is 5 raised to the power of 4, then taking one third of that? Answer:",
    continuation: "208.333",
    measures: "exponents followed by a fraction",
    language: "en",
  },
  {
    id: "math-4",
    category: "mathematics",
    split: "held-out",
    prompt:
      "A train leaves the station at 65 km/h. Two hours later a second train leaves the same station on the same track at 85 km/h. How many hours after the second train leaves does it catch up?\nAnswer:",
    continuation: "6.5",
    measures: "relative speed over a head start",
    language: "en",
  },
  {
    id: "math-5",
    category: "mathematics",
    split: "held-out",
    prompt:
      "The sum of three consecutive whole numbers is 150. What is the largest of the three?\nAnswer:",
    continuation: "51",
    measures: "solving a simple linear equation in words",
    language: "en",
  },
  {
    id: "math-6",
    category: "mathematics",
    split: "held-out",
    prompt:
      "What is 14.5 plus 3.75, minus 6.125?\nAnswer:",
    continuation: "12.125",
    measures: "decimal arithmetic with two different operations",
    language: "en",
  },
  {
    id: "math-7",
    category: "mathematics",
    split: "held-out",
    prompt:
      "A rectangle has perimeter 48 cm and width 9 cm. What is its area?\nAnswer:",
    continuation: "108",
    measures: "geometry from the perimeter relation",
    language: "en",
  },
  {
    id: "math-8",
    category: "mathematics",
    split: "held-out",
    prompt:
      "A number is doubled and then increased by 14. The result is 40. What was the original number?\nAnswer:",
    continuation: "13",
    measures: "reverse two-step arithmetic",
    language: "en",
  },
  {
    id: "math-9",
    category: "mathematics",
    split: "held-out",
    prompt:
      "What is 3.2 times 4.5, then divided by 1.2? Answer:",
    continuation: "12",
    measures: "decimal multiplication followed by division",
    language: "en",
  },
  {
    id: "math-10",
    category: "mathematics",
    split: "held-out",
    prompt:
      "A square's side is 7 cm. A second square has sides 3 cm longer. How many more square centimetres does the second square have?\nAnswer:",
    continuation: "24",
    measures: "area of two squares with a difference",
    language: "en",
  },
];

// ---------------------------------------------------------------------------
// Reasoning cases
// ---------------------------------------------------------------------------

const REASONING_CASES: Step7ExtCase[] = [
  {
    id: "reason-1",
    category: "reasoning",
    split: "held-out",
    prompt:
      "Each of the four cards on the table has a letter on one side and a number on the other. The visible sides show A, 3, 7 and P. The rule is: if a card has a vowel on one side, it must have an even number on the other. Which cards must you turn over to test the rule?\nAnswer:",
    continuation: "A and 3",
    measures: "selecting exactly the cards that can falsify a stated rule",
    language: "en",
  },
  {
    id: "reason-2",
    category: "reasoning",
    split: "held-out",
    prompt:
      "If a traitor always lies and a guard always tells the truth, and one of the two says 'I am the traitor', which one is it?\nAnswer:",
    continuation: "the guard",
    measures: "self-referential sentence analysis",
    language: "en",
  },
  {
    id: "reason-3",
    category: "reasoning",
    split: "held-out",
    prompt:
      "Every time the server crashed after the update, the queue had grown beyond a thousand. The queue today is eight hundred. Can you conclude the server will crash?\nAnswer:",
    continuation: "no",
    measures: "not conflating correlation with certainty",
    language: "en",
  },
  {
    id: "reason-4",
    category: "reasoning",
    split: "held-out",
    prompt:
      "Five people stand in a line: Ada is next to Ben. Ben is next to Cara. Cara is next to Dan. Dan is next to Ella. Ella is next to Ada. Who stands in the middle?\nAnswer:",
    continuation: "Cara",
    measures: "deducing the centre of a cycle from adjacency",
    language: "en",
  },
  {
    id: "reason-5",
    category: "reasoning",
    split: "held-out",
    prompt:
      "A doctor and a surgeon each have one child. The child of the doctor is the surgeon's son. The surgeon cannot be the doctor's husband. Who is the surgeon?\nAnswer:",
    continuation: "the doctor's wife",
    measures: "avoiding a single bias in family-role reasoning",
    language: "en",
  },
  {
    id: "reason-6",
    category: "reasoning",
    split: "held-out",
    prompt:
      "Each of these three statements is either true or false. If the first is true then the second is false; if the second is false then the third is true. What is the truth value of the first statement?\nAnswer:",
    continuation: "false",
    measures: "three-link conditional chain",
    language: "en",
  },
  {
    id: "reason-7",
    category: "reasoning",
    split: "held-out",
    prompt:
      "A clock is set correctly at noon. It loses 4 minutes every hour. What is the correct time when the faulty clock shows 6 pm?\nAnswer:",
    continuation: "6 hours 24 minutes",
    measures: "inverting a constant-rate drift",
    language: "en",
  },
  {
    id: "reason-8",
    category: "reasoning",
    split: "held-out",
    prompt:
      "Red and blue balls are drawn from a bag one at a time without replacement until one colour runs out. The bag started with 7 red and 5 blue. Which colour is left in the bag when the game ends?\nAnswer:",
    continuation: "red",
    measures: "inverting a draw process with an unequal start",
    language: "en",
  },
];

// ---------------------------------------------------------------------------
// Coding cases
// ---------------------------------------------------------------------------

const CODING_CASES: Step7ExtCase[] = [
  {
    id: "code-1",
    category: "coding",
    split: "held-out",
    prompt:
      "function add(a, b) { return a + b; } console.log(add(2, 3));  // what is printed?\nAnswer:",
    continuation: "5",
    measures: "reading a two-line function and its call",
    language: "en",
  },
  {
    id: "code-2",
    category: "coding",
    split: "held-out",
    prompt:
      "for (let i = 0; i < 4; i++) { console.log(i * i); }  // what is the last number printed?\nAnswer:",
    continuation: "9",
    measures: "looping and squaring in order",
    language: "en",
  },
  {
    id: "code-3",
    category: "coding",
    split: "held-out",
    prompt:
      "const letters = ['a', 'b', 'c']; letters.push('d'); letters[0] = 'z'; console.log(letters.join(''));  // what is printed?\nAnswer:",
    continuation: "zbcd",
    measures: "array mutation and join order",
    language: "en",
  },
  {
    id: "code-4",
    category: "coding",
    split: "held-out",
    prompt:
      "function count(votes) { return votes['yes'] + votes['no']; } console.log(count({ yes: 3, no: 5 }));  // what is printed?\nAnswer:",
    continuation: "8",
    measures: "object property lookup and sum",
    language: "en",
  },
  {
    id: "code-5",
    category: "coding",
    split: "held-out",
    prompt:
      "if (true) { let x = 1; } console.log(typeof x);  // what is printed?\nAnswer:",
    continuation: "undefined",
    measures: "block-scoped variable visibility",
    language: "en",
  },
  {
    id: "code-6",
    category: "coding",
    split: "held-out",
    prompt:
      "function repeat(s, n) { return s.repeat(n); } console.log(repeat('ab', 4));  // what is printed?\nAnswer:",
    continuation: "ab ab ab ab",
    measures: "string repeat semantics",
    language: "en",
  },
  {
    id: "code-7",
    category: "coding",
    split: "held-out",
    prompt:
      "while (false) { console.log('never'); }  // how many lines are printed?\nAnswer:",
    continuation: "0",
    measures: "loop condition evaluation",
    language: "en",
  },
  {
    id: "code-8",
    category: "coding",
    split: "held-out",
    prompt:
      "function triple(x) { return x * 3; } console.log(triple(3 + 1));  // what is printed?\nAnswer:",
    continuation: "12",
    measures: "argument evaluation order before call",
    language: "en",
  },
];

// ---------------------------------------------------------------------------
// Extension suite construction
// ---------------------------------------------------------------------------

export type Step7ExtensionSuite = {
  name: string;
  version: string;
  fingerprint: string;
  frozenAt: number;
  cases: Step7ExtCase[];
  counts: Record<Step7ExtCategory, number>;
  note: string;
};

/** Fingerprint over the extension suite's content. */
export function step7ExtensionSuiteFingerprint(cases: Step7ExtCase[]): string {
  let hash = 0x811c9dc5;
  const feed = (text: string) => {
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
  };
  feed(STEP7_EXTENSION_SUITE_VERSION);
  for (const evalCase of cases) {
    feed(evalCase.id);
    feed("|");
    feed(evalCase.category);
    feed("|");
    feed(evalCase.split);
    feed("|");
    feed(evalCase.prompt);
    feed("|");
    feed(evalCase.continuation ?? "");
    feed("|");
    feed((evalCase.expect ?? []).join(";"));
  }
  return `s7e_${hash.toString(16).padStart(8, "0")}`;
}

/**
 * Build the frozen extension suite.
 *
 * `heldOutDocuments` from the production suite are intentionally NOT reused:
 * the extension suite has its own authored cases. The training documents used
 * for the audit are the union of every document both arms were trained on,
 * exactly as for the main suite.
 */
export function createEvalStep7Suite(options: {
  trainingDocuments: string[];
  now?: number;
  freeze?: boolean;
}): Step7ExtensionSuite {
  const cases: Step7ExtCase[] = [...MATHEMATICS_CASES, ...REASONING_CASES, ...CODING_CASES];

  const counts = {
    mathematics: 0,
    reasoning: 0,
    coding: 0,
  } as Record<Step7ExtCategory, number>;
  for (const evalCase of cases) counts[evalCase.category] += 1;

  const suite: Step7ExtensionSuite = {
    name: "alpha-step7-extension-suite",
    version: STEP7_EXTENSION_SUITE_VERSION,
    fingerprint: step7ExtensionSuiteFingerprint(cases),
    frozenAt: options.freeze === false ? 0 : (options.now ?? Date.now()),
    cases,
    counts,
    note:
      "Step 7 extension: mathematics, reasoning and coding, measured on a separate frozen suite so the production suite fingerprint (evl_b58eacf7) is never rewritten. Context in prompt for every comprehension-style case.",
  };

  return suite;
}

/** Refuse a suite edited after it was frozen. */
export function assertStep7ExtensionSuiteFrozen(suite: Step7ExtensionSuite): { ok: true } {
  if (suite.frozenAt === 0) {
    throw new AlphaValidationError(
      "model",
      `evaluation suite "${suite.name}" is not frozen; it must be frozen before any model is measured against it`,
      { suite: suite.name, version: suite.version },
    );
  }
  const actual = step7ExtensionSuiteFingerprint(suite.cases);
  if (actual !== suite.fingerprint) {
    throw new AlphaValidationError(
      "model",
      `evaluation suite "${suite.name}@${suite.version}" was edited after it was frozen: ` +
        `recorded fingerprint ${suite.fingerprint}, current content hashes to ${actual}`,
      { expected: suite.fingerprint, actual },
    );
  }
  return { ok: true };
}

/** Audit the extension cases against the union of both arms' training documents. */
export function auditStep7ExtensionLeakage(
  suite: Step7ExtensionSuite,
  trainingDocuments: string[],
  options: { threshold?: number } = {},
): { clean: boolean; maximumCoverage: number; audited: number; summary: string } {
  const threshold = options.threshold ?? 0.4;
  let maximumCoverage = 0;
  for (const evalCase of suite.cases) {
    const pieces = [evalCase.prompt];
    if (evalCase.continuation) pieces.push(evalCase.continuation);
    for (const piece of pieces) {
      const report = detectOverlap([{ label: evalCase.id, text: piece }], trainingDocuments, {
        contaminationThreshold: threshold,
      });
      maximumCoverage = Math.max(maximumCoverage, report.maxShingleCoverage);
    }
  }
  const clean = maximumCoverage < threshold;
  return {
    clean,
    maximumCoverage,
    audited: suite.cases.length,
    summary: clean
      ? `${suite.cases.length} extension cases audited against ${trainingDocuments.length} training documents: no overlap at or above ${Math.round(threshold * 100)}% shingle coverage (max observed ${Math.round(maximumCoverage * 100)}%)`
      : `${suite.cases.length} extension cases audited but some reached or exceeded ${Math.round(threshold * 100)}% coverage`,
  };
}

/**
 * Compute the per-category mean teacher-forced NLL from a raw list of
 * continuation scores. Reused so the extension report and the main report use
 * the same aggregation.
 */
export function extensionCategoryScores(cases: Step7ExtCase[], scores: Map<string, number | null>): {
  mathematics: number | null;
  reasoning: number | null;
  coding: number | null;
} {
  const means: Record<Step7ExtCategory, number[]> = {
    mathematics: [],
    reasoning: [],
    coding: [],
  };
  for (const evalCase of cases) {
    const score = scores.get(evalCase.id);
    if (score !== undefined && score !== null) means[evalCase.category].push(score);
  }
  return {
    mathematics: means.mathematics.length ? mean(means.mathematics) : null,
    reasoning: means.reasoning.length ? mean(means.reasoning) : null,
    coding: means.coding.length ? mean(means.coding) : null,
  };
}

function mean(values: number[]): number {
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}
