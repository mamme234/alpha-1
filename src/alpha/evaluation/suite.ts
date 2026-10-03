/**
 * Alpha's expanded evaluation suite.
 *
 * Step 4's capability check was six substring matches. That is a floor, not a
 * benchmark: a model could score 6/6 by regurgitating three stock phrases. This
 * suite measures eight separate behaviours and stores the *raw result* of every
 * case — the generated tokens, the measured numbers, and whether a measurement
 * was applicable at all.
 *
 * Three rules govern everything here:
 *
 *   1. **No composite score.** Categories are reported as counts and raw
 *      numbers, never averaged into one "intelligence" figure. A model that is
 *      good at structured output and poor at retention has not got a middling
 *      score; it has two different results.
 *
 *   2. **Frozen before use.** The suite carries a version and a content
 *      fingerprint. Once a model has been measured against it, changing a case
 *      changes the fingerprint, and a gate that requires the frozen fingerprint
 *      will refuse it. That is the mechanism behind "the evaluation does not
 *      change after seeing the result".
 *
 *   3. **Context-in-prompt.** Every comprehension-style case supplies its
 *      material in the prompt, so the answer is derivable rather than memorised.
 *      That makes these tests of *capability* rather than recall, and it is what
 *      makes a held-out claim checkable: we can audit the case text against the
 *      training corpus for actual overlap.
 *
 * Nothing here is trained on. `assertSuiteNotInTraining` exists so that claim is
 * enforced rather than assumed.
 */

import { AlphaValidationError } from "../core/errors";
import { detectOverlap } from "../datasets/splits";
import { words } from "../datasets/diversity";

export const ALPHA_EVAL_SUITE_VERSION = "1.0.0";

/**
 * The behaviours Alpha measures, kept separate throughout.
 *
 * Step 5's eight are the verified production set. Step 7 adds the three
 * categories below for the new model — mathematics, reasoning and coding —
 * whose cases are measured on a SEPARATE frozen extension suite so that the
 * production suite's fingerprint (and Step 5's recorded results) are never
 * rewritten by Step 7 work.
 */
export const EVAL_CATEGORIES = [
  "language-modeling",
  "completion",
  "instruction-following",
  "question-answering",
  "summarization",
  "structured-output",
  "context-retention",
  "generation-quality",
  "mathematics",
  "reasoning",
  "coding",
] as const;

export type EvalCategory = (typeof EVAL_CATEGORIES)[number];

/**
 * `known` material may appear in training (Alpha's own components); a pass there
 * can be recall. `held-out` material is audited to be absent from training, so a
 * pass there is generalisation. The distinction is reported, never blended.
 */
export type EvalSplit = "known" | "held-out";

export type FormatRequirement = {
  /**
   * The generated text must contain each of these. Used by structured-output
   * and instruction-following to measure format compliance without pretending
   * the content is correct.
   */
  mustContain?: string[];
  /** A JSON.parse attempt must succeed and produce this shape of object. */
  json?: {
    keys?: string[];
    /** Key must map to a value of this type. */
    types?: Record<string, "string" | "number" | "boolean" | "array" | "object">;
  };
  /** The generated text must contain at least this many list items. */
  minListItems?: number;
  /** Must contain at least this many distinct of the given words. */
  minDistinctWords?: number;
  /** Length bounds on the generated text, in characters. */
  minCharacters?: number;
  maxCharacters?: number;
};

export type EvalCase = {
  id: string;
  category: EvalCategory;
  split: EvalSplit;
  /** What the model sees. */
  prompt: string;
  /**
   * The continuation the model would have had to produce. Used for the
   * teacher-forced and overlap measurements; optional for format cases.
   */
  continuation?: string;
  /** Substrings whose presence in the generation counts as a match. */
  expect?: string[];
  format?: FormatRequirement;
  language: string;
  /** One line on what this case is actually testing. */
  measures: string;
};

/** A short passage used for summarization and context cases, authored here. */
type SourcePassage = { id: string; text: string };

/** A context-retention passage: it must also name what to ask and the answer. */
type ContextPassage = SourcePassage & {
  question: string;
  answer: string;
  /** Additional phrases that must appear in a passing answer. */
  alsoExpected: string[];
};

/**
 * Source passages for summarization and context cases. Authored for this suite
 * and audited against the training corpus by `assertSuiteNotInTraining`.
 *
 * The context-retention passages deliberately introduce a *specific* entity in
 * the first sentence and ask about it later, so a model that drops earlier
 * context fails the case rather than passing it by luck.
 */
const SUMMARIZATION_PASSAGES: SourcePassage[] = [
  {
    id: "sum-1",
    text: "A training run was started on a Tuesday with a fixed seed. The first forty steps brought the loss down quickly, then the curve flattened. The team stopped adding steps and instead measured how often the same sentence appeared across the corpus. The repetition rate was higher than anyone expected, so the corpus was rebuilt before training resumed.",
  },
  {
    id: "sum-2",
    text: "The workshop kept its records in a paper ledger for three generations. Entries were written in pencil and the margins carried corrections in four different hands. When the archive was finally digitised, the team transcribed the corrections as well as the entries, because the corrections were the only record of how a decision had changed over time.",
  },
  {
    id: "sum-3",
    text: "A small model can be trained on a laptop in a few minutes, which makes it possible to change one thing and measure the result immediately. The team exploited that: they ran the same configuration twice to confirm it reproduced, then altered a single hyperparameter at a time. Most changes made no difference at all, and the two that mattered were found within a day.",
  },
];

const CONTEXT_PASSAGES: ContextPassage[] = [
  {
    id: "ctx-1",
    text: "The harbour office was run by a woman named Ilse Branning, who had taken the post from her uncle in the spring of 1961. She kept the tide tables in a wooden box on the left of her desk. The box was painted a dull green, and it was the only thing in the office that anyone ever painted.",
    question: "Who ran the harbour office, and where were the tide tables kept?",
    answer: "Ilse Branning",
    alsoExpected: ["tide tables", "wooden box"],
  },
  {
    id: "ctx-2",
    text: "There were two clocks in the reading room, and only one of them worked. The broken clock hung above the door and had not been wound since the war. The working clock sat on the mantelpiece and was wound every Thursday afternoon by the caretaker, Mr Okonjo.",
    question: "Which clock worked, and who wound it?",
    answer: "the mantelpiece clock",
    alsoExpected: ["Okonjo", "Thursday"],
  },
  {
    id: "ctx-3",
    text: "The shipment of paper arrived three days late, which meant the printers could not start until the Friday. Rather than delay the whole run, the foreman switched the order and printed the covers first on the stock he already had. By the time the paper came, the covers were stacked and waiting.",
    question: "What did the foreman print first, and why?",
    answer: "the covers",
    alsoExpected: ["paper", "late"],
  },
];

const QUESTION_PASSAGES: Array<SourcePassage & { question: string; answer: string }> = [
  {
    // Every question-answering case supplies its material in the prompt and
    // asks for an answer that is present verbatim in it. The subject matter is
    // deliberately outside Alpha's own technical vocabulary: an earlier set of
    // these cases was worded from the Step 4 corpus's own definitions, which
    // put the answers into the baseline's training data and turned a
    // comprehension measurement into a recall measurement.
    id: "qa-1",
    text:
      "The orchard's irrigation channel is cleared in early spring, before the buds open, " +
      "so that water reaches the roots without obstruction.",
    question: "When and why is the orchard's irrigation channel cleared?",
    answer: "in early spring, before the buds open, so that water reaches the roots",
  },
  {
    id: "qa-2",
    text:
      "A photographer sets the lens hood rather than a filter, because the hood blocks stray light " +
      "without changing what reaches the film.",
    question: "Why does the photographer use a lens hood instead of a filter?",
    answer: "because the hood blocks stray light without changing what reaches the film",
  },
  {
    id: "qa-3",
    text: "The test split is carved out first and is never handed to the trainer, so evaluation data cannot leak into training.",
    question: "Why is the test split carved out first?",
    answer: "so that evaluation data cannot leak into training",
  },    {
      // Deliberately outside Alpha's own domain vocabulary. An earlier version
      // of this case was worded from the generated corpus's own definition of
      // byte pair encoding, which put its answer verbatim in the Step 4
      // baseline's training data: the case measured recall, not extraction.
      id: "qa-4",
      text:
        "The tide gauge at the harbour records the sea level once every ten minutes, " +
        "and each reading is checked against the chart datum rather than against the previous reading of the harbour.",
      question: "What is each tide gauge reading checked against?",
      answer: "the chart datum rather than against the previous reading of the harbour",
    },
  {
    id: "qa-5",
    text:
      "A seamstress presses each seam flat as she sews it, because a seam pressed later " +
      "never lies as smoothly as one pressed at once.",
    question: "Why does the seamstress press each seam as she sews it?",
    answer: "because a seam pressed later never lies as smoothly as one pressed at once",
  },
  {
    id: "qa-6",
    text:
      "The lighthouse keeper writes the lamp's hours in a bound book, and at the end of every month " +
      "that book is read against the shipping list.",
    question: "What is the lamp book read against at the end of every month?",
    answer: "at the end of every month the book is read against the shipping list",
  },
];

/** Long multi-sentence context, for the "longer context" measurement. */
const LONG_CONTEXT_CASES: Array<{
  id: string;
  text: string;
  question: string;
  answer: string;
}> = [
  {
    id: "long-1",
    text:
      "The archive occupied the whole of the basement and had been left untouched for eleven years. " +
      "The first room held ship logs, arranged by year rather than by port, which nobody could explain. " +
      "The second room held correspondence, much of it water damaged. " +
      "In the third room, at the back, behind a shelving unit that had fallen across the door, the team found a single steel cabinet. " +
      "Inside the cabinet was a ledger written entirely in a hand nobody recognised.",
    question: "What was inside the steel cabinet?",
    answer: "a ledger written entirely in a hand nobody recognised",
  },
  {
    id: "long-2",
    text:
      "Ada arrived first and set out the instruments. " +
      "Rune came second and opened the windows. " +
      "Mira arrived last, carrying a folded map and a brass key. " +
      "The key, she explained, opened the cabinet in the north workshop, and nobody had held it since her father died. " +
      "She put it on the sill where the light was, and then the three of them began.",
    question: "What did Mira bring, and what did the key open?",
    answer: "the cabinet in the north workshop",
  },
];

/**
 * Entity tracking: an entity is named at the start and its property changes
 * later, so a model must track which entity holds which property rather than
 * simply recalling the first noun it saw.
 */
const ENTITY_TRACKING: Array<{
  id: string;
  text: string;
  question: string;
  answer: string;
  wrongIfConfusedWith: string;
}> = [
  {
    id: "ent-1",
    text:
      "The left crate was labelled SALT and the right crate was labelled SUGAR. " +
      "During the night someone moved the labels but not the contents. " +
      "In the morning the crate labelled SUGAR still held sugar, because only the labels had been moved backwards.",
    question: "What did the crate labelled SUGAR hold in the morning?",
    answer: "sugar",
    wrongIfConfusedWith: "salt",
  },
  {
    id: "ent-2",
    text:
      "Rune kept the ledger and Ines kept the keys. " +
      "When Rune left, she gave the ledger to Ines and kept nothing. " +
      "So after Rune left, Ines had both the ledger and the keys.",
    question: "Who had the ledger after Rune left?",
    answer: "Ines",
    wrongIfConfusedWith: "Rune",
  },
];

// ---------------------------------------------------------------------------
// Case construction
// ---------------------------------------------------------------------------

function completionCases(): EvalCase[] {
  const cases: EvalCase[] = [
    {
      id: "comp-1",
      category: "completion",
      split: "held-out",
      prompt: "The tide came in slowly and",
      continuation: "the harbour emptied of everything that had been sitting on the mud.",
      measures: "continuation of an open-ended clause into grammatical English",
      language: "en",
    },
    {
      id: "comp-2",
      category: "completion",
      split: "held-out",
      prompt: "When the loss stops falling on held-out data,",
      continuation: "continuing to train on the same data will not help, and new data will.",
      measures: "continuation of a technical clause that requires an understanding of the premise",
      language: "en",
    },
    {
      id: "comp-3",
      category: "completion",
      split: "held-out",
      prompt: "It had been a long week, and",
      continuation: "nobody wanted to be the one to say that the plan had not worked.",
      measures: "continuation of a narrative clause with correct conjunction",
      language: "en",
    },
    {
      id: "comp-4",
      category: "completion",
      split: "known",
      prompt: "attention lets every position in a sequence",
      continuation: "weigh every other position by relevance",
      measures: "continuation of a definition the model may have been trained on (recall)",
      language: "en",
    },
    {
      id: "comp-5",
      category: "completion",
      split: "held-out",
      // Deliberately not drawn from the corpus: the instruction category in the
      // training data lists software-engineering steps, so a continuation from
      // an unrelated domain is genuinely held out rather than coincidentally so.
      prompt: "1. Boil the water and let it cool slightly. 2.",
      continuation: "Pour the water slowly over the leaves and wait four minutes.",
      measures: "continuation of an ordered list with the expected marker convention, on a domain absent from training",
      language: "en",
    },
    {
      id: "comp-6",
      category: "completion",
      split: "held-out",
      prompt: "Mira: You have been in the archive all morning.\nRavi:",
      continuation: "There was a ledger to sort and nobody else to sort it.",
      measures: "continuation of a dialogue turn with the correct speaker format",
      language: "en",
    },
  ];
  return cases;
}

function instructionCases(): EvalCase[] {
  return [
    {
      id: "inst-1",
      category: "instruction-following",
      split: "held-out",
      prompt:
        "Instruction: List exactly three properties of layer normalisation, one per line, numbered 1 to 3.\nResponse:",
      continuation: "1. It normalises the activations.\n2. It stabilises the variance.\n3. It keeps the scale learnable.",
      format: { minListItems: 3, minDistinctWords: 6 },
      measures: "following a counting and formatting instruction",
      language: "en",
    },
    {
      id: "inst-2",
      category: "instruction-following",
      split: "held-out",
      prompt:
        "Instruction: Repeat the word alpha four times, separated by commas, and nothing else.\nResponse:",
      continuation: "alpha, alpha, alpha, alpha",
      format: { mustContain: ["alpha"], minCharacters: 4, maxCharacters: 40 },
      measures: "following an exact-output instruction (count and delimiter)",
      language: "en",
    },
    {
      id: "inst-3",
      category: "instruction-following",
      split: "held-out",
      prompt:
        "Instruction: Answer in one short sentence. Do not use a list.\nContext: A checkpoint stores the weights and the optimiser state.\nQuestion: What does a checkpoint store?\nResponse:",
      continuation: "A checkpoint stores the weights and the optimiser state.",
      format: { mustContain: ["checkpoint"], minCharacters: 10, maxCharacters: 200 },
      measures: "following a two-part instruction (answer the question, obey the format)",
      language: "en",
    },
    {
      id: "inst-4",
      category: "instruction-following",
      split: "held-out",
      prompt:
        "Instruction: Continue the pattern. Say the next two letters.\nContext: A, B, C, D,\nResponse:",
      continuation: "E, F",
      format: { mustContain: ["E"] },
      measures: "following a pattern-completion instruction with an exact expected symbol",
      language: "en",
    },
    {
      id: "inst-5",
      category: "instruction-following",
      split: "known",      prompt: "Instruction: Define attention in one sentence.\nResponse:",
      continuation: "Attention lets each position decide how much weight to give every other position.",
      format: { mustContain: ["attention"], minCharacters: 15, maxCharacters: 300 },
      measures: "following a definitional instruction on material the model may have seen",
      language: "en",
    },
    {
      id: "inst-6",
      category: "instruction-following",
      split: "held-out",
      prompt:
        "Instruction: Give the answer as two sentences. First state the cause, then the effect.\nContext: The learning rate was set far too high.\nResponse:",
      continuation: "The learning rate was set far too high. The updates grew too large and the run stopped converging.",
      format: { minCharacters: 15, maxCharacters: 400 },
      measures: "following an ordering instruction (cause before effect)",
      language: "en",
    },
  ];
}

function structuredCases(): EvalCase[] {
  return [
    {
      id: "struct-1",
      category: "structured-output",
      split: "held-out",
      prompt:
        'Instruction: Return a JSON object with keys "name" and "colour", both strings.\nContext: The coat was charcoal.\nResponse:',
      format: { json: { keys: ["name", "colour"], types: { name: "string", colour: "string" } } },
      measures: "producing a parsable object with the requested keys and value types",
      language: "en",
    },
    {
      id: "struct-2",
      category: "structured-output",
      split: "held-out",
      prompt:
        'Instruction: Return a JSON object with a single key "count" whose value is a number.\nContext: There were three clocks in the room.\nResponse:',
      format: { json: { keys: ["count"], types: { count: "number" } } },
      measures: "producing a parsable object whose value is the correct JSON type",
      language: "en",
    },
    {
      id: "struct-3",
      category: "structured-output",
      split: "held-out",
      prompt:
        "Instruction: Produce a list of the three colours, one per line, each line starting with a dash.\nContext: The three colours were amber, slate and rust.\nResponse:",
      format: { minListItems: 3, mustContain: ["-"] },
      measures: "producing a list with the requested marker",
      language: "en",
    },
    {
      id: "struct-4",
      category: "structured-output",
      split: "held-out",
      prompt:
        'Instruction: Return the value as a JSON array of two strings.\nContext: The two fields were name and value.\nResponse:',
      format: { json: { types: {} }, minCharacters: 2 },
      measures: "producing a parsable array",
      language: "en",
    },
    {
      id: "struct-5",
      category: "structured-output",
      split: "known",
      prompt:
        'Instruction: Return a JSON object with keys "term" and "definition", both strings.\nContext: The term used throughout is attention.\nResponse:',
      format: { json: { keys: ["term", "definition"] } },
      measures: "producing a parsable object on material the model may have seen",
      language: "en",
    },
  ];
}

function generationQualityCases(): EvalCase[] {
  // These cases exist to expose degenerate generation. They ask for output
  // longer than a single sentence, because repetition loops and premature
  // truncation do not appear in short replies.
  return [
    {
      id: "gen-1",
      category: "generation-quality",
      split: "held-out",
      prompt:
        "Write three sentences describing a room you have never been in before. Begin: The room was",
      format: { minCharacters: 60, maxCharacters: 600 },
      measures:
        "sustained generation over several sentences: repetition, longest token run and distinct 3-gram ratio",
      language: "en",
    },
    {
      id: "gen-2",
      category: "generation-quality",
      split: "held-out",
      prompt:
        "List five objects that might be found on a workshop bench, one per line.",
      format: { minCharacters: 40, maxCharacters: 600 },
      measures: "enumeration without collapsing into a loop of a single repeated item",
      language: "en",
    },
    {
      id: "gen-3",
      category: "generation-quality",
      split: "held-out",
      prompt:
        "Continue this paragraph for four more sentences.\n\nThe tide came in slowly and",
      format: { minCharacters: 80, maxCharacters: 900 },
      measures:
        "whether generation stops on its own or runs to the token cap, and whether it degenerates",
      language: "en",
    },
    {
      id: "gen-4",
      category: "generation-quality",
      split: "held-out",
      prompt:
        "Answer this in two sentences: what is the difference between validation loss and test loss?",
      format: { minCharacters: 30, maxCharacters: 500 },
      measures: "stop behaviour: whether generation terminates before the token cap",
      language: "en",
    },
    {
      id: "gen-5",
      category: "generation-quality",
      split: "known",
      prompt: "Continue this definition: perplexity is",
      measures: "repetition behaviour on material the model may have trained on",
      language: "en",
    },
  ];
}

function summarizationCases(): EvalCase[] {
  return SUMMARIZATION_PASSAGES.map((passage, index) => ({
    id: `summ-${index + 1}`,
    category: "summarization" as const,
    split: "held-out" as const,
    prompt: `Instruction: Summarise the passage in one short sentence.\nPassage: ${passage.text}\nResponse:`,
    continuation: undefined,
    format: { minCharacters: 15, maxCharacters: 260 },
    measures: "condensing a three-to-four sentence passage into one sentence within a length bound",
    language: "en",
  }));
}

function qaCases(): EvalCase[] {
  return QUESTION_PASSAGES.map((item) => ({
    id: item.id,
    category: "question-answering" as const,
    split: "held-out" as const,
    prompt: `Context: ${item.text}\nQuestion: ${item.question}\nAnswer:`,
    continuation: item.answer,
    expect: [item.answer],
    measures: "extracting an answer that is present verbatim in the supplied context",
    language: "en",
  }));
}

function contextRetentionCases(): EvalCase[] {
  const cases: EvalCase[] = CONTEXT_PASSAGES.map((item) => ({
    id: item.id,
    category: "context-retention" as const,
    split: "held-out" as const,
    prompt: `${item.text}\nQuestion: ${item.question}\nAnswer:`,
    continuation: item.answer,
    expect: [item.answer, ...item.alsoExpected],
    measures: "answering from information introduced earlier in the same context",
    language: "en",
  }));

  for (const item of LONG_CONTEXT_CASES) {
    cases.push({
      id: item.id,
      category: "context-retention",
      split: "held-out",
      prompt: `${item.text}\nQuestion: ${item.question}\nAnswer:`,
      continuation: item.answer,
      expect: [item.answer],
      measures: "answering from a longer context (5-7 sentences), measuring where retention degrades",
      language: "en",
    });
  }

  for (const item of ENTITY_TRACKING) {
    cases.push({
      id: item.id,
      category: "context-retention",
      split: "held-out",
      prompt: `${item.text}\nQuestion: ${item.question}\nAnswer:`,
      continuation: item.answer,
      // `expect` deliberately includes the answer only. The confusable entity is
      // recorded separately by the runner so a match on it is reported as a
      // confusion rather than as a pass.
      expect: [item.answer],
      measures:
        "tracking an entity whose property changed, rather than recalling the first entity seen",
      language: "en",
    });
  }

  return cases;
}

function languageModelingCases(documents: string[]): EvalCase[] {
  // Language-modelling cases are drawn from the *held-out documents* the runner
  // supplies, so the measurement is over text rather than over authored prompts.
  return documents.map((text, index) => ({
    id: `lm-${String(index).padStart(3, "0")}`,
    category: "language-modeling" as const,
    split: "held-out" as const,
    prompt: text,
    continuation: text,
    measures: "teacher-forced cross entropy and next-token accuracy over held-out text",
    language: "en",
  }));
}

export type EvalSuite = {
  name: string;
  version: string;
  /** FNV-1a over every case and the passages, so any edit is detectable. */
  fingerprint: string;
  /** Epoch ms at which the suite was frozen. 0 means "not yet frozen". */
  frozenAt: number;
  cases: EvalCase[];
  /** Counts per category, for a report. */
  counts: Record<EvalCategory, number>;
  note: string;
};

/** Fingerprint over the suite's text content. Order-sensitive. */
export function suiteFingerprint(cases: EvalCase[]): string {
  let hash = 0x811c9dc5;
  const feed = (text: string) => {
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
  };
  feed(ALPHA_EVAL_SUITE_VERSION);
  for (const evalCase of cases) {
    feed(evalCase.id);
    feed("\u0000");
    feed(evalCase.category);
    feed("\u0000");
    feed(evalCase.split);
    feed("\u0000");
    feed(evalCase.prompt);
    feed("\u0001");
    feed(evalCase.continuation ?? "");
    feed("\u0001");
    feed((evalCase.expect ?? []).join("\u0002"));
    feed("\u0003");
  }
  return `evl_${hash.toString(16).padStart(8, "0")}`;
}

/**
 * Build the suite.
 *
 * `heldOutDocuments` are the text for language-modelling measurements. They
 * must come from the dataset's TEST split — never from training — and the
 * caller asserts that separately with `assertSuiteNotInTraining`.
 */
export function createEvalSuite(options: {
  heldOutDocuments: string[];
  now?: number;
  /** Freeze the suite immediately. Set false only while drafting it. */
  freeze?: boolean;
}): EvalSuite {
  const cases: EvalCase[] = [
    ...languageModelingCases(options.heldOutDocuments),
    ...completionCases(),
    ...instructionCases(),
    ...qaCases(),
    ...summarizationCases(),
    ...structuredCases(),
    ...generationQualityCases(),
    ...contextRetentionCases(),
  ];

  const counts = Object.fromEntries(EVAL_CATEGORIES.map((c) => [c, 0])) as Record<
    EvalCategory,
    number
  >;
  for (const evalCase of cases) counts[evalCase.category] += 1;

  return {
    name: "alpha-capability-suite",
    version: ALPHA_EVAL_SUITE_VERSION,
    fingerprint: suiteFingerprint(cases),
    frozenAt: options.freeze === false ? 0 : (options.now ?? Date.now()),
    cases,
    counts,
    note:
      "Held out from training. Context-in-prompt for every comprehension case, so answers are " +
      "derived rather than recalled. No composite score is computed over these categories.",
  };
}

/**
 * Refuse a suite that has been edited since it was frozen.
 *
 * This is the mechanism that keeps the evaluation from changing after a result is
 * known: a gate records the fingerprint it evaluated against, and a later run
 * whose fingerprint differs is a different evaluation, not a better one.
 */
export function assertSuiteFrozen(suite: EvalSuite, expectedFingerprint?: string): { ok: true } {
  if (suite.frozenAt === 0) {
    throw new AlphaValidationError(
      "model",
      `evaluation suite "${suite.name}" is not frozen; it must be frozen before any model is measured against it`,
      { suite: suite.name, version: suite.version },
    );
  }
  const actual = suiteFingerprint(suite.cases);
  if (actual !== suite.fingerprint) {
    throw new AlphaValidationError(
      "model",
      `evaluation suite "${suite.name}@${suite.version}" was edited after it was frozen: ` +
        `recorded fingerprint ${suite.fingerprint}, current content hashes to ${actual}`,
      { expected: suite.fingerprint, actual },
    );
  }
  if (expectedFingerprint && expectedFingerprint !== suite.fingerprint) {
    throw new AlphaValidationError(
      "model",
      `this evaluation used suite ${suite.fingerprint}, but the gate expects ${expectedFingerprint}; ` +
        "the two runs are not comparable",
      { expected: expectedFingerprint, actual: suite.fingerprint },
    );
  }
  return { ok: true };
}

export type SuiteLeakageReport = {
  clean: boolean;
  /** Fingerprint of the training corpus that was audited against. */
  trainingFingerprint: string;
  /** Case ids whose prompt or continuation overlaps the training corpus. */
  contaminated: Array<{ caseId: string; coverage: number; exact: boolean }>;
  /** Cases audited in total. */
  audited: number;
  /** Maximum coverage found on any single case. */
  maxCoverage: number;
  summary: string;
};

/**
 * Audit the suite against a training corpus.
 *
 * Both prompts and continuations are checked, because a leaked continuation
 * would let the model score on an answer it was trained to produce. Coverage is
 * measured with word shingles at a conservative threshold, so ordinary shared
 * phrasing is not mistaken for contamination — but the coverage figure is
 * reported either way.
 */
export function auditSuiteLeakage(
  suite: EvalSuite,
  trainingDocuments: string[],
  options: { trainingFingerprint?: string; threshold?: number } = {},
): SuiteLeakageReport {
  const threshold = options.threshold ?? 0.5;
  const contaminated: SuiteLeakageReport["contaminated"] = [];
  let maxCoverage = 0;

  for (const evalCase of suite.cases) {
    // Language-modelling cases carry whole documents; audit them as passages.
    // Every other case is audited on both the prompt and the expected text.
    const pieces = [evalCase.prompt];
    if (evalCase.continuation && evalCase.category !== "language-modeling") {
      pieces.push(evalCase.continuation);
    }
    for (const piece of pieces) {
      const report = detectOverlap([{ label: evalCase.id, text: piece }], trainingDocuments, {
        contaminationThreshold: threshold,
      });
      const coverage = report.maxShingleCoverage;
      if (coverage > maxCoverage) maxCoverage = coverage;
      if (report.shingleMatches.length > 0 || report.exactMatches.length > 0) {
        contaminated.push({
          caseId: evalCase.id,
          coverage,
          exact: report.exactMatches.length > 0,
        });
      }
    }
  }

  const unique = [...new Map(contaminated.map((c) => [c.caseId, c])).values()];
  const clean = unique.length === 0;
  return {
    clean,
    trainingFingerprint: options.trainingFingerprint ?? "unknown",
    contaminated: unique,
    audited: suite.cases.length,
    maxCoverage,
    summary: clean
      ? `${suite.cases.length} case(s) audited against the training corpus: no overlap at or above ${(threshold * 100).toFixed(0)}% shingle coverage (max observed ${(maxCoverage * 100).toFixed(1)}%)`
      : `${unique.length} of ${suite.cases.length} case(s) overlap the training corpus: ${unique
          .slice(0, 6)
          .map((c) => `${c.caseId} (${(c.coverage * 100).toFixed(0)}%${c.exact ? ", exact" : ""})`)
          .join(", ")}`,
  };
}

/** Refuse to evaluate with a suite that is present in the training data. */
export function assertSuiteNotInTraining(
  suite: EvalSuite,
  trainingDocuments: string[],
  options: { trainingFingerprint?: string; threshold?: number } = {},
): SuiteLeakageReport {
  const report = auditSuiteLeakage(suite, trainingDocuments, options);
  if (!report.clean) {
    throw new AlphaValidationError(
      "model",
      `refusing to evaluate: the suite overlaps its own training data — ${report.summary}`,
      { contaminated: report.contaminated },
    );
  }
  return report;
}

/** Categories with at least one case, in declaration order. */
export function populatedCategories(suite: EvalSuite): EvalCategory[] {
  return EVAL_CATEGORIES.filter((category) => suite.counts[category] > 0);
}

/** One-line description for a report. */
export function describeEvalSuite(suite: EvalSuite): string {
  const parts = populatedCategories(suite).map((category) => `${category} ${suite.counts[category]}`);
  const known = suite.cases.filter((c) => c.split === "known").length;
  const heldOut = suite.cases.filter((c) => c.split === "held-out").length;
  return (
    `${suite.name}@${suite.version} · ${suite.cases.length} cases · ${suite.fingerprint} · ` +
    `${heldOut} held-out / ${known} known · frozen ${new Date(suite.frozenAt).toISOString()} · ${parts.join(", ")}`
  );
}

/** Exposed for tests that need to confirm shingle extraction is sane. */
export function caseWords(evalCase: EvalCase): string[] {
  return words(evalCase.prompt);
}
