/**
 * Supervised instruction data.
 *
 * An instruction example is three separate things — an instruction, an optional
 * context/input, and a response/output — kept as three separate fields rather
 * than one concatenated string. That matters for two reasons: it makes a missing
 * response detectable instead of silently becoming a blank continuation, and
 * it lets the evaluation suite ask whether a response used the context it was
 * given.
 *
 * Synthetic material is labelled, never hidden. Alpha does not have a corpus of
 * human-written instruction/response pairs it is entitled to train on, so the
 * examples here are authored in this repository and marked `synthetic: true`.
 * An evaluation result built on synthetic data is reported as such. Training on
 * synthetic responses that were written to match an evaluation answer is exactly
 * the kind of circularity Step 5 forbids, so:
 *
 *   - the instruction corpus and the evaluation suite are built from different
 *     content and never share a fingerprint (asserted, not assumed);
 *   - `assertNoInstructionLeakage` checks it.
 *
 * Validation reports every problem it finds and deletes nothing. A caller that
 * wants a filtered set says so explicitly.
 */

import { AlphaValidationError } from "../core/errors";
import {
  createProvenanceDocument,
  provenanceFingerprint,
  type AcquisitionRecord,
  type MixCategory,
  type ProvenanceDocument,
} from "./provenance";

/** One supervised example, with the three fields kept separate. */
export type InstructionExample = {
  id: string;
  /** What the model is being asked to do. Required, non-empty. */
  instruction: string;
  /** Optional material the response should be based on. May be empty. */
  context: string | null;
  /** What a correct response looks like. Required, non-empty. */
  response: string;
  /** True when the response was generated rather than written by a person. */
  synthetic: boolean;
  /** Which skill this exercises: one of the categories below. */
  skill: InstructionSkill;
  provenance: ProvenanceDocument;
};

/** Skills the instruction set exercises. Reported as counts, never as a score. */
export const INSTRUCTION_SKILLS = [
  "definition",
  "explanation",
  "comparison",
  "summarisation",
  "extraction",
  "transformation",
  "structured-output",
  "question-answering",
  "context-following",
] as const;

export type InstructionSkill = (typeof INSTRUCTION_SKILLS)[number];

export type InstructionDataset = {
  datasetId: string;
  name: string;
  version: string;
  description: string;
  license: string;
  origin: string;
  /** Fingerprint over every example's provenance record. */
  fingerprint: string;
  examples: InstructionExample[];
  createdAt: number;
};

export type InstructionIssue = {
  exampleId: string;
  code:
    | "missing-instruction"
    | "missing-response"
    | "empty-instruction"
    | "empty-response"
    | "excessive-length"
    | "duplicate-example"
    | "duplicate-response"
    | "malformed-record"
    | "unknown-skill"
    | "invalid-unicode"
    | "provenance-mismatch";
  problem: string;
};

export type InstructionValidation = {
  valid: boolean;
  issues: InstructionIssue[];
  counts: {
    examples: number;
    synthetic: number;
    withContext: number;
    bySkill: Record<string, number>;
  };
  /** Mean response length in characters. */
  meanResponseCharacters: number;
};

export type InstructionValidationOptions = {
  /** Instruction + context + response longer than this is rejected. Default 8000. */
  maxTotalCharacters?: number;
  /** Response longer than this is rejected. Default 4000. */
  maxResponseCharacters?: number;
};

const INVALID_UNICODE = /[\uD800-\uDFFF\uFFFD]/;

export type CreateInstructionExampleInput = {
  id: string;
  instruction: string;
  context?: string | null;
  response: string;
  synthetic: boolean;
  skill: InstructionSkill;
  sourceId: string;
  language?: string;
  category?: MixCategory;
  license: string;
  origin: "authored" | "public-domain" | "user-owned" | "licensed";
  acquisition: AcquisitionRecord;
  createdAt: number;
};

/** Build one example, attaching the same per-document provenance the corpus uses. */
export function createInstructionExample(
  input: CreateInstructionExampleInput,
): InstructionExample {
  const provenance = createProvenanceDocument({
    documentId: `instr:${input.id}`,
    text: `${input.instruction}\n${input.context ?? ""}\n${input.response}`,
    sourceId: input.sourceId,
    origin: input.origin,
    license: input.license,
    language: input.language ?? "en",
    category: input.category ?? "instructions",
    acquisition: input.acquisition,
    createdAt: input.createdAt,
  });
  return {
    id: input.id,
    instruction: input.instruction,
    context: input.context ?? null,
    response: input.response,
    synthetic: input.synthetic,
    skill: input.skill,
    provenance,
  };
}

export type CreateInstructionDatasetInput = {
  datasetId: string;
  name: string;
  version: string;
  description: string;
  license: string;
  origin: string;
  examples: InstructionExample[];
  now?: number;
};

/** Wrap examples in a versioned, fingerprinted dataset. */
export function createInstructionDataset(input: CreateInstructionDatasetInput): InstructionDataset {
  return {
    datasetId: input.datasetId,
    name: input.name,
    version: input.version,
    description: input.description,
    license: input.license,
    origin: input.origin,
    fingerprint: instructionFingerprint(input.examples),
    examples: [...input.examples],
    createdAt: input.now ?? Date.now(),
  };
}

/** Fingerprint over every example, including its provenance record. */
export function instructionFingerprint(examples: InstructionExample[]): string {
  let hash = 0x811c9dc5;
  const feed = (text: string) => {
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
  };
  feed(String(examples.length));
  for (const example of examples) {
    feed(example.id);
    feed("\u0000");
    feed(example.instruction);
    feed("\u0001");
    feed(example.context ?? "");
    feed("\u0001");
    feed(example.response);
    feed("\u0002");
  }
  return `instr_${hash.toString(16).padStart(8, "0")}`;
}

/** Validate every example. Reports all problems; removes nothing. */
export function validateInstructionDataset(
  dataset: InstructionDataset,
  options: InstructionValidationOptions = {},
): InstructionValidation {
  const maxTotal = options.maxTotalCharacters ?? 8000;
  const maxResponse = options.maxResponseCharacters ?? 4000;
  const issues: InstructionIssue[] = [];

  const bySkill: Record<string, number> = {};
  for (const skill of INSTRUCTION_SKILLS) bySkill[skill] = 0;

  const seenId = new Set<string>();
  const seenPair = new Map<string, string>();
  const seenResponse = new Map<string, string>();

  let synthetic = 0;
  let withContext = 0;
  let responseCharacters = 0;

  for (const example of dataset.examples) {
    const fail = (code: InstructionIssue["code"], problem: string) =>
      issues.push({ exampleId: String(example?.id ?? "<missing>"), code, problem });

    if (!example || typeof example.id !== "string" || !example.id) {
      fail("malformed-record", "example is not a record with an id");
      continue;
    }
    if (typeof example.instruction !== "string") {
      fail("missing-instruction", "an example without an instruction cannot be followed");
    }
    if (typeof example.response !== "string") {
      fail("missing-response", "an example without a response teaches the model to stop");
    }
    if (typeof example.instruction !== "string" || typeof example.response !== "string") {
      fail("malformed-record", "instruction and response must both be strings");
      continue;
    }
    if (seenId.has(example.id)) {
      fail("duplicate-example", `example id "${example.id}" appears more than once`);
    } else {
      seenId.add(example.id);
    }
    if (example.instruction.trim().length === 0) {
      // Distinct from `missing-instruction`: the field is present but blank,
      // which is a different authoring mistake and a different fix.
      fail("empty-instruction", "the instruction is present but blank");
    }
    if (example.response.trim().length === 0) {
      fail("empty-response", "the response is present but blank");
    }
    if (!INSTRUCTION_SKILLS.includes(example.skill)) {
      fail("unknown-skill", `skill "${String(example.skill)}" is not one Alpha defines`);
    } else {
      bySkill[example.skill] += 1;
    }

    const totalLength =
      example.instruction.length + (example.context?.length ?? 0) + example.response.length;
    if (totalLength > maxTotal) {
      fail("excessive-length", `${totalLength} characters, above the ${maxTotal} limit`);
    }
    if (example.response.length > maxResponse) {
      fail("excessive-length", `response is ${example.response.length} characters, above ${maxResponse}`);
    }
    if (INVALID_UNICODE.test(example.instruction + (example.context ?? "") + example.response)) {
      fail("invalid-unicode", "example contains a lone surrogate or replacement character");
    }

    const pairKey = `${example.instruction.trim().toLowerCase()}\u0000${(example.context ?? "").trim().toLowerCase()}`;
    const earlierPair = seenPair.get(pairKey);
    if (earlierPair) {
      fail("duplicate-example", `same instruction and context as "${earlierPair}"`);
    } else {
      seenPair.set(pairKey, example.id);
    }

    const responseKey = example.response.trim().toLowerCase();
    const earlierResponse = seenResponse.get(responseKey);
    if (earlierResponse && responseKey.length > 0) {
      fail("duplicate-response", `same response as "${earlierResponse}"`);
    } else {
      seenResponse.set(responseKey, example.id);
    }

    // The provenance record must still describe the text it claims to describe.
    const expected = provenanceFingerprint({
      documentId: `instr:${example.id}`,
      text: `${example.instruction}\n${example.context ?? ""}\n${example.response}`,
      sourceId: example.provenance.sourceId,
      origin: example.provenance.origin,
      license: example.provenance.license,
      language: example.provenance.language,
      category: example.provenance.category,
      acquisition: example.provenance.acquisition,
      createdAt: example.provenance.createdAt,
      version: example.provenance.version,
    });
    if (expected !== example.provenance.fingerprint) {
      fail("provenance-mismatch", "the provenance fingerprint does not match the example's text");
    }

    if (example.synthetic) synthetic += 1;
    if (example.context && example.context.trim().length > 0) withContext += 1;
    responseCharacters += example.response.length;
  }

  return {
    valid: issues.length === 0,
    issues,
    counts: {
      examples: dataset.examples.length,
      synthetic,
      withContext,
      bySkill,
    },
    meanResponseCharacters:
      dataset.examples.length === 0 ? 0 : responseCharacters / dataset.examples.length,
  };
}

/** Throwing form, for the training entry point. */
export function assertValidInstructionDataset(
  dataset: InstructionDataset,
  options: InstructionValidationOptions = {},
): InstructionValidation {
  const validation = validateInstructionDataset(dataset, options);
  if (!validation.valid) {
    throw new AlphaValidationError(
      "datasets",
      `instruction dataset "${dataset.name}@${dataset.version}" is invalid: ${validation.issues
        .slice(0, 4)
        .map((i) => `${i.exampleId}: ${i.problem}`)
        .join("; ")}`,
      { issues: validation.issues.slice(0, 10) },
    );
  }
  return validation;
}

/**
 * Render one example as training text.
 *
 * The prompt markers are constant and explicit so that "instruction followed by
 * response" is a *learnable, uniform* surface form rather than a property that
 * depends on how each example happened to be written.
 */
export function renderInstructionPrompt(example: InstructionExample): string {
  const parts = [`Instruction: ${example.instruction.trim()}`];
  if (example.context && example.context.trim().length > 0) {
    parts.push(`Context: ${example.context.trim()}`);
  }
  parts.push("Response:");
  return parts.join("\n");
}

/** The full training text for one example: prompt markers then the response. */
export function renderInstructionExample(example: InstructionExample): string {
  return `${renderInstructionPrompt(example)}\n${example.response.trim()}`;
}

/** Every example as a training document, in dataset order. */
export function instructionDocuments(dataset: InstructionDataset): string[] {
  return dataset.examples.map(renderInstructionExample);
}

export type LeakageReport = {
  clean: boolean;
  /** Example ids present in both sets, matched on instruction+context. */
  sharedExamples: string[];
  /** Individual instruction strings present in both sets. */
  sharedInstructions: string[];
  /** Whole-dataset fingerprint collision. */
  sameFingerprint: boolean;
};

/**
 * Check that two instruction sets do not overlap.
 *
 * Matching is on the normalised instruction+context pair and on the instruction
 * alone, because a leak that changes the context still leaks the question.
 */
export function detectInstructionLeakage(
  training: InstructionDataset,
  heldOut: InstructionDataset,
): LeakageReport {
  const trainingPairs = new Map<string, string>();
  const trainingInstructions = new Set<string>();
  for (const example of training.examples) {
    trainingPairs.set(
      `${example.instruction.trim().toLowerCase()}\u0000${(example.context ?? "").trim().toLowerCase()}`,
      example.id,
    );
    trainingInstructions.add(example.instruction.trim().toLowerCase());
  }

  const sharedExamples: string[] = [];
  const sharedInstructions: string[] = [];
  for (const example of heldOut.examples) {
    const key = `${example.instruction.trim().toLowerCase()}\u0000${(example.context ?? "").trim().toLowerCase()}`;
    if (trainingPairs.has(key)) sharedExamples.push(example.id);
    if (trainingInstructions.has(example.instruction.trim().toLowerCase())) {
      sharedInstructions.push(example.id);
    }
  }

  return {
    clean: sharedExamples.length === 0 && sharedInstructions.length === 0,
    sharedExamples,
    sharedInstructions,
    sameFingerprint: training.fingerprint === heldOut.fingerprint,
  };
}

/** Refuse to train on a held-out set that overlaps the training set. */
export function assertNoInstructionLeakage(
  training: InstructionDataset,
  heldOut: InstructionDataset,
): LeakageReport {
  const report = detectInstructionLeakage(training, heldOut);
  if (!report.clean) {
    throw new AlphaValidationError(
      "datasets",
      `instruction leakage: ${report.sharedExamples.length} shared example(s) and ` +
        `${report.sharedInstructions.length} shared instruction(s) between ` +
        `"${training.name}" and "${heldOut.name}"; the held-out set is not held out`,
      { sharedExamples: report.sharedExamples, sharedInstructions: report.sharedInstructions },
    );
  }
  return report;
}

/** One-line summary. Counts and the synthetic fraction — no score. */
export function summariseInstructionDataset(
  dataset: InstructionDataset,
  validation?: InstructionValidation,
): string {
  const report = validation ?? validateInstructionDataset(dataset);
  const skills = Object.entries(report.counts.bySkill)
    .filter(([, count]) => count > 0)
    .map(([name, count]) => `${name} ${count}`)
    .join(", ");
  const syntheticShare =
    report.counts.examples === 0 ? 0 : report.counts.synthetic / report.counts.examples;
  return (
    `${dataset.name}@${dataset.version} · ${dataset.fingerprint} · ${report.counts.examples} examples ` +
    `(${report.counts.withContext} with context, ${(syntheticShare * 100).toFixed(0)}% synthetic) · ` +
    `mean response ${report.meanResponseCharacters.toFixed(0)} chars · skills: ${skills}` +
    (report.valid ? "" : ` · ${report.issues.length} issue(s)`)
  );
}
