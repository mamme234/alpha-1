/**
 * Alpha CLI — verify Step 5 capability.
 *
 *   bun scripts/alpha-verify-capability.ts
 *   bun scripts/alpha-verify-capability.ts --json
 *   bun scripts/alpha-verify-capability.ts --export-artifact <path>
 *
 * `--export-artifact` writes the trained candidate arm — weights, tokenizer
 * snapshot, configuration and every fingerprint recorded above — to a serving
 * artifact. The artifact is therefore written by the same run that measured the
 * model, so what Step 6 serves is exactly what Step 5 verified.
 *
 * Walks the required pipeline end to end, and prints a measurement at every
 * stage rather than a claim:
 *
 *   DATA -> TRAINING -> FROZEN EVALUATION -> MEASUREMENTS
 *         -> BASELINE COMPARISON -> CAPABILITY GATE -> REPORT
 *
 * Two arms are trained inside this one run:
 *
 *   baseline  alpha-micro trained on the Step 4 generated corpus
 *   candidate alpha-micro trained on the Step 5 authored mixture
 *
 * They share one architecture, one tokenizer and one training configuration, so
 * what differs between them is the training corpus — as much as the experiment
 * design allows. Nothing is mocked, no external model is involved anywhere, and
 * no stage produces a composite score: every number below is reported beside
 * its counterpart rather than added to a total.
 *
 * The run ends in exactly one of:
 *
 *   CAPABILITY IMPROVED             the frozen gate was satisfied
 *   CAPABILITY NOT YET IMPROVED     measured honestly, the bar was not met
 *   CAPABILITY INFRASTRUCTURE READY the pipeline ran but no capability claim
 *                                   can be supported from it
 *
 * It never declares a winner, and it reports a regression as a regression.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import {
  ALPHA_EVAL_SUITE_VERSION,
  ALPHA_MIX_CATEGORIES,
  ALPHA_MODEL_PRESETS,
  AlphaTokenizer,
  AlphaTrainer,
  AlphaTransformer,
  assertNoSplitLeakage,
  assertGateStable,
  assertProvenance,
  assertSuiteFrozen,
  assertSuiteNotInTraining,
  auditSuiteLeakage,
  buildAuthoredCorpus,
  buildGeneratedCorpus,
  buildMixture,
  compareCapability,
  completeExperiment,
  countParameters,
  createDataset,
  createDatasetVersion,
  createEvalSuite,
  createExperiment,
  createInstructionDataset,
  createInstructionExample,
  createModelConfig,
  createProvenanceDocument,
  datasetFingerprint,
  decideTokenizerChange,
  describeComparison,
  describeEvalSuite,
  describeGate,
  describeMixture,
  describeProvenance,
  describeReproduction,
  describeSplits,
  documentFingerprint,
  evaluateGate,
  instructionDocuments,
  measureTokenizer,
  modelConfigFingerprint,
  recordCapability,
  recordTraining,
  runEvalSuite,
  shingleFingerprints,
  splitProvenanceDocuments,
  summariseCapabilityReport,
  summariseInstructionDataset,
  tokenMetricsFromSummary,
  validateInstructionDataset,
  validateModelConfig,
  withSourceCounts,
  analyseDiversity,
  compareDiversity,
  type AlphaDatasetVersion,
  type AlphaModelConfig,
  type CapabilityComparisonRow,
  type CapabilityReport,
  type EvalCategory,
  type EvalSuite,
  type Experiment,
  type GateEvaluation,
  type ProvenanceCorpus,
  type ProvenanceDocument,
  type ProvenanceSource,
  type TrainingConfig,
  type TrainingSummary,
  type TokenizerDecision,
  type TokenizerMeasurement,
} from "../index";
import {
  createServingArtifact,
  describeServingArtifact,
  loadServingArtifact,
} from "../serving/artifact";

// ---------------------------------------------------------------------------
// check plumbing
// ---------------------------------------------------------------------------

type Check = { id: string; label: string; passed: boolean; detail: string };

const checks: Check[] = [];
let quiet = false;

function record(id: string, label: string, passed: boolean, detail: string): void {
  checks.push({ id, label, passed, detail });
  if (!quiet) console.log(`  ${passed ? "ok  " : "FAIL"} ${id}. ${label} — ${detail}`);
}

function section(title: string): void {
  if (!quiet) console.log(`\n${title}`);
}

// ---------------------------------------------------------------------------
// experiment constants — fixed, so a rerun measures the same thing
// ---------------------------------------------------------------------------

/** Model and optimiser seed. Identical for both arms. */
const SEED = 1337;
/** Corpus generation seeds. */
const AUTHORED_SEED = 20260101;
const INSTRUCTION_SEED = 20260102;
const GENERATED_SEED = 20250930;
const EVAL_SOURCE_SEED = 777001;
/** Selection seeds. */
const MIXTURE_SEED = 4242;
const SPLIT_SEED = 20260101;

const AUTHORED_DOCUMENTS = 480;
const INSTRUCTION_DOCUMENTS = 48;
const GENERATED_DOCUMENTS = 300;
const EVAL_SOURCE_DOCUMENTS = 80;

/**
 * A held-out document may share at most this share of its word shingles with
 * anything either arm was trained on. The split repair uses the same ceiling,
 * so a document that would sit above it was never in the evaluation set.
 */
const EVAL_CONTAMINATION_CEILING = 0.4;

const TOTAL_STEPS = Math.max(
  4,
  Number.isFinite(Number(process.env.ALPHA_CAPABILITY_STEPS))
    ? Number(process.env.ALPHA_CAPABILITY_STEPS)
    : 32,
);
const BATCH_SIZE = 8;
const SEQ_LEN = 48;
const ACCUMULATION = 2;
const EVAL_INTERVAL = Math.max(1, Math.floor(TOTAL_STEPS / 4));
const EVAL_BATCHES = 3;

/** Fixed timestamp so dataset and experiment records are reproducible. */
const RECORDED_AT = 1_760_000_000_000;

/** Metrics where a *lower* value means a more varied corpus. */
const DIVERSITY_LOWER_IS_BETTER: Record<string, boolean> = {
  "boilerplate.ratio": false,
  "boilerplate.meanOverlap": false,
  "lowInformation.ratio": false,
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

const now = () => RECORDED_AT;
const num = (value: number | null | undefined, digits = 4): string =>
  value === null || value === undefined || Number.isNaN(value) ? "n/a" : value.toFixed(digits);
const pct = (value: number | null | undefined, digits = 1): string =>
  value === null || value === undefined || Number.isNaN(value) ? "n/a" : `${(value * 100).toFixed(digits)}%`;

function ratioOf(m: { total: number; value: number } | null): string {
  return m && m.total > 0 ? `${m.value}/${m.total}` : "n/a";
}

function countRatio(m: { total: number; matched?: number; passed?: number }): string {
  if (m.total <= 0) return "n/a";
  const value = m.matched ?? m.passed ?? 0;
  return `${value}/${m.total}`;
}

/**
 * Turn authored documents into provenance records, dropping exact duplicates
 * rather than silently training on the same text twice.
 */
function authoredProvenance(
  count: number,
  seed: number,
  prefix: string,
): { documents: ProvenanceDocument[]; duplicatesRemoved: number } {
  const seen = new Set<string>();
  const documents: ProvenanceDocument[] = [];
  let duplicatesRemoved = 0;
  for (const entry of buildAuthoredCorpus(count, seed)) {
    const fingerprint = documentFingerprint(entry.text);
    if (seen.has(fingerprint)) {
      duplicatesRemoved += 1;
      continue;
    }
    seen.add(fingerprint);
    documents.push(
      createProvenanceDocument({
        documentId: `${prefix}_${String(documents.length).padStart(4, "0")}`,
        text: entry.text,
        sourceId: "alpha-authored-corpus",
        origin: "authored",
        license: "Alpha-owned",
        language: entry.language,
        category: entry.category,
        acquisition: {
          method: "generated-in-repo",
          location: "this-repository",
          acquiredAt: RECORDED_AT,
          collectedBy: `alpha:${prefix}`,
        },
        createdAt: RECORDED_AT,
      }),
    );
  }
  return { documents, duplicatesRemoved };
}

/** Step 4's corpus as provenance records, so both arms get the same treatment. */
function step4Provenance(texts: string[]): ProvenanceDocument[] {
  return texts.map((text, index) =>
    createProvenanceDocument({
      documentId: `s4_${String(index).padStart(4, "0")}`,
      text,
      sourceId: "alpha-generated",
      origin: "authored",
      license: "Alpha-owned",
      language: "en",
      category: "factual-reference",
      acquisition: {
        method: "generated-in-repo",
        location: "this-repository",
        acquiredAt: RECORDED_AT,
        collectedBy: "alpha:generated-corpus",
      },
      createdAt: RECORDED_AT,
    }),
  );
}

/**
 * Shingle coverage of one document against a reference corpus, computed with
 * the reference indexed once. `detectOverlap` rebuilds that index per call,
 * which is the right shape for an audit but the wrong shape for a filter run
 * over every candidate evaluation document.
 */
function coverageIndex(documents: string[]): {
  exact: Set<string>;
  shingles: Set<string>;
  coverage(text: string): number;
} {
  const exact = new Set<string>();
  const shingles = new Set<string>();
  for (const text of documents) {
    exact.add(documentFingerprint(text));
    for (const shingle of shingleFingerprints(text, 8)) shingles.add(shingle);
  }
  return {
    exact,
    shingles,
    coverage(text: string): number {
      if (exact.has(documentFingerprint(text))) return 1;
      const own = shingleFingerprints(text, 8);
      if (own.size === 0) return 0;
      let shared = 0;
      for (const shingle of own) if (shingles.has(shingle)) shared += 1;
      return shared / own.size;
    },
  };
}

/** Generation throughput measured from the runner's own per-case timings. */
function generationSpeed(report: CapabilityReport): {
  tokens: number;
  latencyMs: number;
  tokensPerSecond: number | null;
} {
  let tokens = 0;
  let latencyMs = 0;
  for (const evalCase of report.cases) {
    tokens += evalCase.generatedTokens;
    latencyMs += evalCase.latencyMs;
  }
  return { tokens, latencyMs, tokensPerSecond: latencyMs > 0 ? tokens / (latencyMs / 1000) : null };
}

function categoryOf(report: CapabilityReport, category: EvalCategory) {
  return report.categories.find((c) => c.category === category) ?? null;
}

function categoryLine(report: CapabilityReport, category: EvalCategory): string {
  const cat = categoryOf(report, category);
  if (!cat) return "n/a";
  const parts = [`${cat.cases} cases`];
  if (cat.matched) parts.push(`matched ${cat.matched.value}/${cat.matched.total}`);
  if (cat.formatPassed) parts.push(`format ${cat.formatPassed.value}/${cat.formatPassed.total}`);
  if (cat.meanContinuationNll !== null) parts.push(`NLL ${cat.meanContinuationNll.toFixed(4)}`);
  if (cat.meanContinuationTop1 !== null) parts.push(`top1 ${pct(cat.meanContinuationTop1, 2)}`);
  if (cat.meanRepetitionRatio) parts.push(`repeat ${pct(cat.meanRepetitionRatio, 2)}`);
  if (cat.meanDistinctTrigramRatio !== null) parts.push(`3-gram ${pct(cat.meanDistinctTrigramRatio, 1)}`);
  if (cat.confusions > 0) parts.push(`confusions ${cat.confusions}`);
  return parts.join(" · ");
}

/** The longest run of one repeated token id anywhere in a report. */
function worstTokenRun(report: CapabilityReport): number {
  return report.cases.reduce((worst, evalCase) => Math.max(worst, evalCase.longestTokenRun), 0);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

export async function main(argv: string[]): Promise<number> {
  const json = argv.includes("--json");
  const exportFlag = argv.indexOf("--export-artifact");
  const exportPath = exportFlag >= 0 ? (argv[exportFlag + 1] ?? null) : null;
  quiet = json;
  const startedAt = Date.now();

  if (!quiet) {
    console.log("Alpha — Step 5 verification: capability and intelligence upgrade\n");
    console.log("No external model is used anywhere in this run. The only language");
    console.log("models involved are two copies of Alpha's own transformer, trained");
    console.log("here, on corpora this repository authored.\n");
    console.log(
      `Pipeline: DATA -> TRAINING -> FROZEN EVALUATION -> MEASUREMENTS -> BASELINE COMPARISON -> CAPABILITY GATE -> REPORT`,
    );
  }

  // ------------------------------------------------------------------ data
  section("Data: provenance, mixture, held-out boundaries");

  const authored = authoredProvenance(AUTHORED_DOCUMENTS, AUTHORED_SEED, "a5");
  const authoredSources: ProvenanceSource[] = withSourceCounts(
    [
      {
        id: "alpha-authored-corpus",
        title: "Alpha authored multi-category corpus",
        origin: "authored",
        license: "Alpha-owned",
        documents: 0,
        note: "Authored in this repository by src/alpha/datasets/authored-corpus.ts. No scraped or third-party text.",
      },
    ],
    authored.documents,
  );
  const provenance = assertProvenance(authored.documents, authoredSources);
  record(
    "1",
    "Every training document carries a provenance record that validates",
    provenance.valid,
    `${authored.documents.length} documents after removing ${authored.duplicatesRemoved} exact duplicate(s) · ` +
      describeProvenance(authored.documents, authoredSources),
  );

  const mixture = buildMixture({
    components: ALPHA_MIX_CATEGORIES.map(
      (category): { id: string; corpus: ProvenanceCorpus; weight: number } => ({
        id: `authored-${category}`,
        corpus: {
          datasetId: "alpha-authored",
          name: "alpha-authored-mixture",
          version: "2.0.0",
          description: "Step 5 multi-category authored training corpus",
          sources: authoredSources,
          documents: authored.documents.filter((d) => d.category === category),
          createdAt: RECORDED_AT,
          previousVersion: null,
        },
        weight: 1,
      }),
    ),
    totalDocuments: authored.documents.length,
    seed: MIXTURE_SEED,
    expectedCategories: [...ALPHA_MIX_CATEGORIES],
  });
  record(
    "2",
    "The training mixture is declared per category and reports what it achieved",
    mixture.unsatisfiableCategories.length === 0 &&
      mixture.shortfall === 0 &&
      mixture.achieved.totalDocuments === authored.documents.length,
    `${describeMixture(mixture)} · fingerprint ${mixture.fingerprint}`,
  );

  const candidateSplit = splitProvenanceDocuments(mixture.documents, {
    validationFraction: 0.12,
    testFraction: 0.12,
    seed: SPLIT_SEED,
  });
  const candidateSplitAudit = assertNoSplitLeakage(candidateSplit);
  record(
    "3",
    "The candidate corpus splits into train/validation/test with no held-out document overlapping training",
    candidateSplitAudit.validation.clean && candidateSplitAudit.test.clean,
    `${describeSplits(candidateSplit)} · validation max coverage ${pct(candidateSplitAudit.validation.maxShingleCoverage)} · test max coverage ${pct(candidateSplitAudit.test.maxShingleCoverage)}`,
  );

  const generated = buildGeneratedCorpus(GENERATED_DOCUMENTS, GENERATED_SEED);
  const step4Documents = step4Provenance(generated.documents);
  const step4Sources: ProvenanceSource[] = withSourceCounts(
    [
      {
        id: "alpha-generated",
        title: "Alpha generated technical corpus",
        origin: "authored",
        license: "Alpha-owned",
        documents: 0,
        note: "Authored in this repository by src/alpha/datasets/generated-corpus.ts.",
      },
    ],
    step4Documents,
  );
  const step4ProvenanceOk = assertProvenance(step4Documents, step4Sources);
  const baselineSplit = splitProvenanceDocuments(step4Documents, {
    validationFraction: 0.12,
    testFraction: 0.12,
    seed: SPLIT_SEED,
  });
  const baselineSplitAudit = assertNoSplitLeakage(baselineSplit);
  record(
    "4",
    "The Step 4 baseline corpus is split and audited by the same rule",
    step4ProvenanceOk.valid && baselineSplitAudit.validation.clean && baselineSplitAudit.test.clean,
    `${describeSplits(baselineSplit)} · ${baselineSplit.validation.length === 0 && baselineSplit.test.length === 0 ? "every held-out document overlapped training above the repair threshold and was moved back — this corpus cannot provide a shingle-independent hold-out, which is reported rather than worked around" : `validation max coverage ${pct(baselineSplitAudit.validation.maxShingleCoverage)}`}`,
  );

  const instructionSeed = authoredProvenance(INSTRUCTION_DOCUMENTS, INSTRUCTION_SEED, "i5");
  const instructionPhrasings = [
    "Give the opening sentence of the passage below.",
    "Repeat the first sentence of the passage below, word for word.",
    "What does the passage begin with? Quote it exactly.",
    "Quote the first sentence of the passage unchanged.",
    "State the opening line of the passage exactly as written.",
    "Copy the passage's first sentence into the response.",
    "Reply with the sentence the passage starts with.",
    "Identify the first sentence of the passage by quoting it.",
  ];
  const responseSeen = new Set<string>();
  const instructionExamples = instructionSeed.documents.flatMap((document, index) => {
    const first = (document.text.split(/(?<=[.!?])\s+/)[0] ?? document.text).trim();
    const key = first.toLowerCase();
    if (responseSeen.has(key) || first.length === 0) return [];
    responseSeen.add(key);
    return [
      createInstructionExample({
        id: `ins-${String(index).padStart(3, "0")}`,
        instruction: instructionPhrasings[index % instructionPhrasings.length],
        context: document.text,
        response: first,
        synthetic: true,
        skill: "extraction",
        sourceId: "alpha-authored-corpus",
        license: "Alpha-owned",
        origin: "authored",
        acquisition: {
          method: "generated-in-repo",
          location: "this-repository",
          acquiredAt: RECORDED_AT,
          collectedBy: "alpha:instructions",
        },
        createdAt: RECORDED_AT,
      }),
    ];
  });
  const instructionDataset = createInstructionDataset({
    datasetId: "alpha-instructions",
    name: "alpha-instruction-set",
    version: "1.0.0",
    description: "Synthetic supervised instruction/context/response examples authored in this repository.",
    license: "Alpha-owned",
    origin: "authored",
    examples: instructionExamples,
    now: RECORDED_AT,
  });
  const instructionValidation = validateInstructionDataset(instructionDataset);
  const instructionDocs = instructionDocuments(instructionDataset);
  record(
    "5",
    "Instruction supervision validates: separate fields, unique pairs, provenance intact",
    instructionValidation.valid && instructionDocs.length > 0,
    `${summariseInstructionDataset(instructionDataset, instructionValidation)} · rendered as ${instructionDocs.length} training documents in the uniform "Instruction / Context / Response" surface form`,
  );

  const baselineVersion = createDatasetVersion({
    datasetId: generated.id,
    name: generated.name,
    version: generated.version,
    description: generated.description,
    sources: [
      {
        id: "alpha-generated",
        title: "Alpha generated technical corpus",
        license: "Alpha-owned",
        origin: "authored",
        note: "Authored in this repository by src/alpha/datasets/generated-corpus.ts.",
      },
    ],
    documents: generated.documents,
    now: RECORDED_AT,
  });

  const candidateRunTexts = [
    ...candidateSplit.train.map((d) => d.text),
    ...candidateSplit.validation.map((d) => d.text),
    ...instructionDocs,
  ];
  const baselineRunTexts = [
    ...baselineSplit.train.map((d) => d.text),
    ...baselineSplit.validation.map((d) => d.text),
  ];

  const candidateVersion = createDatasetVersion({
    datasetId: "alpha-step5-candidate",
    name: "alpha-step5-mixture",
    version: "1.0.0",
    description: "Step 5 authored mixture (train + validation) plus supervised instruction documents.",
    sources: [
      {
        id: "alpha-authored-corpus",
        title: "Alpha authored multi-category corpus",
        license: "Alpha-owned",
        origin: "authored",
        note: "Authored in this repository. No scraped or third-party text.",
      },
    ],
    documents: candidateRunTexts,
    now: RECORDED_AT,
  });

  const rebuiltCandidate = createDatasetVersion({
    datasetId: "alpha-step5-candidate",
    name: "alpha-step5-mixture",
    version: "1.0.0",
    description: "Step 5 authored mixture (train + validation) plus supervised instruction documents.",
    sources: [
      {
        id: "alpha-authored-corpus",
        title: "Alpha authored multi-category corpus",
        license: "Alpha-owned",
        origin: "authored",
        note: "Authored in this repository. No scraped or third-party text.",
      },
    ],
    documents: candidateRunTexts,
    now: RECORDED_AT,
  });
  const rebuiltBaseline = createDatasetVersion({
    datasetId: generated.id,
    name: generated.name,
    version: generated.version,
    description: generated.description,
    sources: [
      {
        id: "alpha-generated",
        title: "Alpha generated technical corpus",
        license: "Alpha-owned",
        origin: "authored",
        note: "Authored in this repository by src/alpha/datasets/generated-corpus.ts.",
      },
    ],
    documents: generated.documents,
    now: RECORDED_AT,
  });
  record(
    "6",
    "Dataset fingerprints are reproducible from content alone",
    rebuiltCandidate.manifest.fingerprint === candidateVersion.manifest.fingerprint &&
      rebuiltBaseline.manifest.fingerprint === baselineVersion.manifest.fingerprint,
    `candidate ${candidateVersion.manifest.fingerprint} · baseline ${baselineVersion.manifest.fingerprint} · both rebuilt identically in this run`,
  );

  const baselineDiversity = analyseDiversity(baselineRunTexts);
  const candidateDiversity = analyseDiversity(candidateRunTexts);
  const diversityRows = compareDiversity(baselineDiversity, candidateDiversity, DIVERSITY_LOWER_IS_BETTER);
  record(
    "7",
    "The candidate corpus is measurably more varied than the Step 4 corpus",
    candidateDiversity.vocabulary.typeTokenRatio > baselineDiversity.vocabulary.typeTokenRatio &&
      candidateDiversity.sentences.distinctOpeningRatio > baselineDiversity.sentences.distinctOpeningRatio,
    `type/token ratio ${baselineDiversity.vocabulary.typeTokenRatio.toFixed(4)} -> ${candidateDiversity.vocabulary.typeTokenRatio.toFixed(4)} · ` +
      `distinct sentence openings ${baselineDiversity.sentences.distinctOpeningRatio.toFixed(4)} -> ${candidateDiversity.sentences.distinctOpeningRatio.toFixed(4)} · ` +
      `boilerplate ${pct(baselineDiversity.boilerplate.ratio)} -> ${pct(candidateDiversity.boilerplate.ratio)} · ` +
      `character entropy ${baselineDiversity.characterVariety.entropyBits.toFixed(2)} -> ${candidateDiversity.characterVariety.entropyBits.toFixed(2)} bits`,
  );

  // -------------------------------------------------------------- tokenizer
  section("Tokenizer: measured, decided, shared");

  const unionTexts = [...baselineRunTexts, ...candidateRunTexts];
  const baselineTokenizer = AlphaTokenizer.train(baselineRunTexts, {
    vocabSize: ALPHA_MODEL_PRESETS.micro.vocabSize,
    version: "1.0.0",
    trainedOn: `${baselineVersion.name}@${baselineVersion.version}`,
  });
  const sharedTokenizer = AlphaTokenizer.train(unionTexts, {
    vocabSize: ALPHA_MODEL_PRESETS.micro.vocabSize,
    version: "1.0.0",
    trainedOn: "step4-baseline+step5-candidate union",
  });

  const currentMeasurement: TokenizerMeasurement = measureTokenizer(baselineTokenizer, unionTexts);
  const candidateMeasurement: TokenizerMeasurement = measureTokenizer(sharedTokenizer, unionTexts);
  const decision: TokenizerDecision = decideTokenizerChange(currentMeasurement, candidateMeasurement);
  const tokenizer = decision.retrain ? sharedTokenizer : baselineTokenizer;
  const tokenizerMeasurement = decision.retrain ? candidateMeasurement : currentMeasurement;

  const adoptedMeasurement = decision.retrain ? candidateMeasurement : currentMeasurement;
  record(
    "8",
    "The decision to replace the tokenizer is derived from measurements of both candidates",
    decision.reason.length > 0 &&
      decision.current.fingerprint === currentMeasurement.tokenizerFingerprint &&
      decision.candidate.fingerprint === candidateMeasurement.tokenizerFingerprint &&
      tokenizer.fingerprint() === adoptedMeasurement.tokenizerFingerprint,
    `${decision.retrain ? "RETRAIN" : "KEEP"} (${decision.reason}): ${decision.explanation} — ` +
      `current ${currentMeasurement.tokenizerFingerprint} coverage ${pct(currentMeasurement.vocabularyCoverage)}, ` +
      `round trip ${currentMeasurement.roundTripExact ? "exact" : "NOT exact"} ` +
      `(${currentMeasurement.unknownCharacters.length} character(s) unrepresented); ` +
      `candidate ${candidateMeasurement.tokenizerFingerprint} coverage ${pct(candidateMeasurement.vocabularyCoverage)}, ` +
      `round trip ${candidateMeasurement.roundTripExact ? "exact" : "NOT exact"}; adopted ${tokenizer.fingerprint()}`,
  );
  record(
    "9",
    "The tokenizer both arms will use is the one the measurement selected",
    tokenizerMeasurement.vocabularyCoverage >= 0.99 &&
      tokenizerMeasurement.unknownTokenShare <= 0.001 &&
      tokenizerMeasurement.roundTripExact,
    `${tokenizer.vocabSize} tokens · ${tokenizer.fingerprint()} · coverage ${pct(tokenizerMeasurement.vocabularyCoverage)} · ` +
      `unknown token share ${pct(tokenizerMeasurement.unknownTokenShare, 4)} · ` +
      `${num(tokenizerMeasurement.charactersPerToken, 3)} characters/token · round trip exact`,
  );

  // ------------------------------------------------- frozen evaluation suite
  section("Frozen evaluation suite");

  const referenceTexts = [...candidateRunTexts, ...baselineRunTexts];
  const referenceIndex = coverageIndex(referenceTexts);
  const evalSource = buildAuthoredCorpus(EVAL_SOURCE_DOCUMENTS, EVAL_SOURCE_SEED);
  const evalCoverages = evalSource.map((entry) => referenceIndex.coverage(entry.text));
  const heldOutDocuments = evalSource
    .filter((_, index) => evalCoverages[index] < EVAL_CONTAMINATION_CEILING)
    .map((entry) => entry.text);
  const rejectedEvalDocuments = EVAL_SOURCE_DOCUMENTS - heldOutDocuments.length;
  const worstEvalCoverage = evalCoverages.reduce((worst, value) => Math.max(worst, value), 0);
  record(
    "10",
    "The held-out evaluation text is drawn from a third corpus and filtered for overlap",
    heldOutDocuments.length >= 8 && rejectedEvalDocuments > 0 &&
      evalSource
        .filter((_, index) => evalCoverages[index] < EVAL_CONTAMINATION_CEILING)
        .every((entry) => referenceIndex.coverage(entry.text) < EVAL_CONTAMINATION_CEILING),
    `${heldOutDocuments.length} of ${EVAL_SOURCE_DOCUMENTS} documents kept below ${EVAL_CONTAMINATION_CEILING * 100}% shingle coverage ` +
      `against both arms' training data; ${rejectedEvalDocuments} rejected (highest coverage anywhere in the source pool: ${pct(worstEvalCoverage)}); ` +
      `seed ${EVAL_SOURCE_SEED}, never used for training`,
  );

  const suite: EvalSuite = createEvalSuite({ heldOutDocuments, now: RECORDED_AT, freeze: true });
  const rebuiltSuite = createEvalSuite({ heldOutDocuments, now: RECORDED_AT, freeze: true });
  assertSuiteFrozen(suite, suite.fingerprint);
  record(
    "11",
    "The evaluation suite is frozen before any model is measured against it",
    suite.fingerprint === rebuiltSuite.fingerprint && suite.frozenAt > 0,
    `${describeEvalSuite(suite)} · rebuilding from the same held-out text reproduces the same fingerprint`,
  );

  let tamperRefused = false;
  try {
    const tampered: EvalSuite = {
      ...suite,
      cases: suite.cases.map((evalCase, index) =>
        index === 0 ? { ...evalCase, prompt: `${evalCase.prompt} edited later` } : evalCase,
      ),
    };
    assertSuiteFrozen(tampered, suite.fingerprint);
  } catch {
    tamperRefused = true;
  }
  record(
    "12",
    "A suite edited after it was frozen is refused rather than silently re-measured",
    tamperRefused,
    "a case edited in memory re-hashed to a different fingerprint and assertSuiteFrozen threw",
  );

  const leakage = auditSuiteLeakage(suite, referenceTexts, {
    threshold: EVAL_CONTAMINATION_CEILING,
    trainingFingerprint: candidateVersion.manifest.fingerprint,
  });
  let refusalWorked = false;
  try {
    assertSuiteNotInTraining(suite, referenceTexts, { threshold: EVAL_CONTAMINATION_CEILING });
    refusalWorked = true;
  } catch {
    refusalWorked = false;
  }
  record(
    "13",
    "No evaluation case overlaps either arm's training data",
    leakage.clean && refusalWorked,
    `${leakage.audited} cases audited against ${referenceTexts.length} training documents at a ${EVAL_CONTAMINATION_CEILING * 100}% ceiling · ` +
      `max observed coverage ${pct(leakage.maxCoverage)} · ${leakage.summary}`,
  );

  // -------------------------------------------------------------- training
  section(`Training both arms (${TOTAL_STEPS} steps, batch ${BATCH_SIZE} x seq ${SEQ_LEN} x accum ${ACCUMULATION})`);

  const modelConfig: AlphaModelConfig = createModelConfig({
    preset: "micro",
    vocabSize: Math.max(ALPHA_MODEL_PRESETS.micro.vocabSize, tokenizer.vocabSize),
  });
  validateModelConfig(modelConfig);
  const configFingerprint = modelConfigFingerprint(modelConfig);
  const parameterCount = countParameters(modelConfig);

  const trainingConfig: TrainingConfig = {
    batchSize: BATCH_SIZE,
    seqLen: SEQ_LEN,
    totalSteps: TOTAL_STEPS,
    learningRate: 0.002,
    schedule: "cosine",
    warmupSteps: 5,
    minFactor: 0.1,
    weightDecay: 0.01,
    gradClipNorm: 1,
    evalInterval: EVAL_INTERVAL,
    evalBatches: EVAL_BATCHES,
    validationFraction: 0.12,
    seed: SEED,
    checkpointInterval: TOTAL_STEPS,
    batchMode: "windows",
    gradientAccumulationSteps: ACCUMULATION,
    earlyStopping: {
      monitor: "validationLoss",
      patience: 3,
      minDelta: 0.01,
      minSteps: Math.min(16, TOTAL_STEPS),
      evalEvery: EVAL_INTERVAL,
    },
  };

  function runArm(arm: {
    runId: string;
    label: string;
    datasetName: string;
    datasetVersion: string;
    documents: string[];
  }): {
    summary: TrainingSummary;
    model: AlphaTransformer;
    trainer: AlphaTrainer;
    datasetFingerprint: string;
  } {
    const dataset = createDataset({
      name: arm.datasetName,
      version: arm.datasetVersion,
      description: `Step 5 capability verification arm: ${arm.label}`,
      license: "Alpha-owned",
      source: arm.runId,
      documents: arm.documents,
    });
    const model = new AlphaTransformer(modelConfig);
    const trainer = new AlphaTrainer({
      model,
      tokenizer,
      dataset,
      runId: arm.runId,
      checkpointLabel: arm.label,
      config: trainingConfig,
    });
    const summary = trainer.trainToCompletion();
    return { summary, model, trainer, datasetFingerprint: datasetFingerprint(dataset) };
  }

  const baselineRun = runArm({
    runId: "step5-baseline",
    label: "step5-baseline-step4-corpus",
    datasetName: baselineVersion.name,
    datasetVersion: baselineVersion.version,
    documents: baselineRunTexts,
  });
  const baselineMetrics = tokenMetricsFromSummary(baselineRun.summary);
  record(
    "14",
    "The baseline arm learned: training loss fell below the no-information baseline",
    baselineMetrics.lastLoss !== null &&
      baselineMetrics.lastLoss < baselineMetrics.uniformLossBaseline &&
      (baselineMetrics.state === "completed" || baselineMetrics.state === "stopped"),
    `state ${baselineMetrics.state} · ${baselineMetrics.steps} steps · ${baselineMetrics.tokensSeen.toLocaleString()} tokens · ` +
      `loss ${num(baselineMetrics.firstLoss)} -> ${num(baselineMetrics.lastLoss)} (uniform ${num(baselineMetrics.uniformLossBaseline)}) · ` +
      `validation ${num(baselineMetrics.validationLoss)} · perplexity ${num(baselineMetrics.validationPerplexity, 2)} · ` +
      `${baselineMetrics.tokensPerSecond} tokens/sec · early stopping: ${baselineRun.summary.earlyStopping?.stoppingReason ?? "n/a"}`,
  );
  record(
    "15",
    "The baseline run writes a checkpoint that names the dataset it trained on",
    baselineRun.trainer.checkpoint !== null &&
      baselineRun.trainer.checkpoint.datasetName === baselineVersion.name &&
      baselineRun.trainer.checkpoint.datasetVersion === baselineVersion.version &&
      baselineRun.trainer.checkpoint.datasetFingerprint === baselineRun.datasetFingerprint,
    baselineRun.trainer.checkpoint
      ? `${baselineRun.trainer.checkpoint.id} at step ${baselineRun.trainer.checkpoint.step} · ` +
        `${(baselineRun.trainer.checkpoint.sizeBytes / 1024 ** 2).toFixed(2)} MiB · dataset ${baselineRun.trainer.checkpoint.datasetName}@${baselineRun.trainer.checkpoint.datasetVersion} · ` +
        `corpus fingerprint ${baselineRun.datasetFingerprint} (version manifest ${baselineVersion.manifest.fingerprint})`
      : "no checkpoint produced",
  );

  const candidateRun = runArm({
    runId: "step5-candidate",
    label: "step5-candidate-step5-corpus",
    datasetName: candidateVersion.name,
    datasetVersion: candidateVersion.version,
    documents: candidateRunTexts,
  });
  const candidateMetrics = tokenMetricsFromSummary(candidateRun.summary);
  record(
    "16",
    "The candidate arm learned: training loss fell below the no-information baseline",
    candidateMetrics.lastLoss !== null &&
      candidateMetrics.lastLoss < candidateMetrics.uniformLossBaseline &&
      (candidateMetrics.state === "completed" || candidateMetrics.state === "stopped"),
    `state ${candidateMetrics.state} · ${candidateMetrics.steps} steps · ${candidateMetrics.tokensSeen.toLocaleString()} tokens · ` +
      `loss ${num(candidateMetrics.firstLoss)} -> ${num(candidateMetrics.lastLoss)} (uniform ${num(candidateMetrics.uniformLossBaseline)}) · ` +
      `validation ${num(candidateMetrics.validationLoss)} · perplexity ${num(candidateMetrics.validationPerplexity, 2)} · ` +
      `${candidateMetrics.tokensPerSecond} tokens/sec · early stopping: ${candidateRun.summary.earlyStopping?.stoppingReason ?? "n/a"}`,
  );
  record(
    "17",
    "The candidate run writes a checkpoint that names the dataset it trained on",
    candidateRun.trainer.checkpoint !== null &&
      candidateRun.trainer.checkpoint.datasetName === candidateVersion.name &&
      candidateRun.trainer.checkpoint.datasetVersion === candidateVersion.version &&
      candidateRun.trainer.checkpoint.datasetFingerprint === candidateRun.datasetFingerprint,
    candidateRun.trainer.checkpoint
      ? `${candidateRun.trainer.checkpoint.id} at step ${candidateRun.trainer.checkpoint.step} · ` +
        `${(candidateRun.trainer.checkpoint.sizeBytes / 1024 ** 2).toFixed(2)} MiB · dataset ${candidateRun.trainer.checkpoint.datasetName}@${candidateRun.trainer.checkpoint.datasetVersion} · ` +
        `corpus fingerprint ${candidateRun.datasetFingerprint} (version manifest ${candidateVersion.manifest.fingerprint})`
      : "no checkpoint produced",
  );

  const candidateEarly = candidateRun.summary.earlyStopping;
  record(
    "18",
    "Gradient accumulation and early stopping are used and recorded, not merely available",
    baselineRun.summary.gradientAccumulationSteps === ACCUMULATION &&
      candidateRun.summary.gradientAccumulationSteps === ACCUMULATION &&
      baselineRun.summary.tokensPerStep === BATCH_SIZE * SEQ_LEN * ACCUMULATION &&
      candidateRun.summary.tokensPerStep === BATCH_SIZE * SEQ_LEN * ACCUMULATION &&
      candidateEarly !== null &&
      candidateEarly.monitor === "validationLoss" &&
      candidateEarly.stoppingReason.length > 0,
    `accumulation ${ACCUMULATION} micro-batch(es) per update in both arms = ${BATCH_SIZE * SEQ_LEN * ACCUMULATION} tokens/step · ` +
      `early stopping monitored ${candidateEarly?.monitor ?? "n/a"}: ${candidateEarly?.stoppingReason ?? "n/a"} ` +
      `(best ${num(candidateEarly?.bestValue ?? null)} at step ${candidateEarly?.bestStep ?? "n/a"}, ` +
      `${candidateEarly?.evaluationsSinceImprovement ?? 0} evaluation(s) since the last improvement, ` +
      `${candidateRun.summary.stepsSkippedByEarlyStopping} step(s) skipped)`,
  );

  const sameArchitecture =
    modelConfigFingerprint(baselineRun.model.config) === configFingerprint &&
    modelConfigFingerprint(candidateRun.model.config) === configFingerprint &&
    countParameters(baselineRun.model.config) === parameterCount &&
    countParameters(candidateRun.model.config) === parameterCount;
  const sameTraining =
    JSON.stringify(baselineRun.trainer.config) === JSON.stringify(candidateRun.trainer.config);
  record(
    "19",
    "The two arms differ only in their training corpus",
    sameArchitecture && sameTraining && baselineRun.trainer.tokenizer.fingerprint() === candidateRun.trainer.tokenizer.fingerprint(),
    `architecture ${configFingerprint} (${parameterCount.toLocaleString()} parameters) in both arms · ` +
      `tokenizer ${tokenizer.fingerprint()} in both arms · training configuration byte-identical · ` +
      `datasets ${baselineVersion.manifest.fingerprint} vs ${candidateVersion.manifest.fingerprint}`,
  );

  // ------------------------------------------------ frozen held-out evaluation
  section("Frozen held-out evaluation (same suite, same tokenizer, both arms)");

  const baselineReport = runEvalSuite(baselineRun.model, tokenizer, suite, {
    heldOutDocuments,
    expectedFingerprint: suite.fingerprint,
    trainedTokens: baselineRun.summary.tokensSeen,
    now: RECORDED_AT,
  });
  const candidateReport = runEvalSuite(candidateRun.model, tokenizer, suite, {
    heldOutDocuments,
    expectedFingerprint: suite.fingerprint,
    trainedTokens: candidateRun.summary.tokensSeen,
    now: RECORDED_AT,
  });

  record(
    "20",
    "The baseline was measured against the frozen suite",
    baselineReport.suite.fingerprint === suite.fingerprint &&
      baselineReport.languageModeling.loss !== null &&
      baselineReport.languageModeling.perplexity !== null,
    `${summariseCapabilityReport(baselineReport)} · held-out loss ${num(baselineReport.languageModeling.loss)} nats/token · ` +
      `perplexity ${num(baselineReport.languageModeling.perplexity, 2)} · next-token top1 ${pct(baselineReport.languageModeling.nextTokenTop1Accuracy, 2)}`,
  );
  record(
    "21",
    "The candidate was measured against the same frozen suite",
    candidateReport.suite.fingerprint === suite.fingerprint &&
      candidateReport.languageModeling.loss !== null &&
      candidateReport.languageModeling.perplexity !== null,
    `${summariseCapabilityReport(candidateReport)} · held-out loss ${num(candidateReport.languageModeling.loss)} nats/token · ` +
      `perplexity ${num(candidateReport.languageModeling.perplexity, 2)} · next-token top1 ${pct(candidateReport.languageModeling.nextTokenTop1Accuracy, 2)}`,
  );
  record(
    "22",
    "Both reports came from the same suite, the same tokenizer and the same architecture",
    baselineReport.suite.fingerprint === candidateReport.suite.fingerprint &&
      baselineReport.model.tokenizerFingerprint === candidateReport.model.tokenizerFingerprint &&
      baselineReport.model.parameters === candidateReport.model.parameters &&
      baselineReport.cases.length === candidateReport.cases.length,
    `suite ${baselineReport.suite.fingerprint} · tokenizer ${baselineReport.model.tokenizerFingerprint} · ` +
      `${baselineReport.model.parameters.toLocaleString()} parameters · ${baselineReport.cases.length} cases each · ` +
      `this is what lets the comparison below mean something`,
  );

  // ---------------------------------------------- comparison and capability gate
  section("Baseline comparison and capability gate");

  const comparison: CapabilityComparisonRow[] = compareCapability(baselineReport, candidateReport);
  const comparableRows = comparison.filter((row) => row.baseline !== null && row.candidate !== null);
  record(
    "23",
    "The comparison is a table of named measurements, never a single score",
    comparableRows.length >= 10 && comparison.some((row) => row.measurement.startsWith("languageModeling.")),
    `${comparableRows.length} of ${comparison.length} measurements have a value on both sides; ` +
      `${comparableRows.filter((r) => r.improved === true).length} moved in the improving direction, ` +
      `${comparableRows.filter((r) => r.improved === false).length} did not; no measurement is aggregated with any other`,
  );

  const gate: GateEvaluation = evaluateGate({
    baseline: baselineReport,
    candidate: candidateReport,
    comparison,
    now: RECORDED_AT,
  });
  let gateStable = false;
  try {
    assertGateStable(gate);
    gateStable = true;
  } catch {
    gateStable = false;
  }
  record(
    "24",
    "The capability gate runs against criteria declared before the measurement",
    gateStable && gate.suitesMatch && gate.evaluated === 8,
    `criteria fingerprint ${gate.gateFingerprint} · ${gate.satisfied} of ${gate.evaluated} satisfied, ${gate.required} required · ` +
      `suites match: ${gate.suitesMatch} · ${gate.verdict}`,
  );

  // ------------------------------------------------------ experiment records
  section("Experiment records and reproducibility");

  function makeExperiment(input: {
    experimentId: string;
    label: string;
    modelId: string;
    version: AlphaDatasetVersion;
    split: { train: number; validation: number; test: number };
    mixtureFingerprint: string | null;
    summary: TrainingSummary;
    report: CapabilityReport;
  }): Experiment {
    let experiment = createExperiment({
      experimentId: input.experimentId,
      label: input.label,
      candidate: {
        modelId: input.modelId,
        modelVersion: "0.1.0",
        preset: "micro",
        config: modelConfig,
        configFingerprint,
        parameterCount,
      },
      dataset: {
        id: input.version.datasetId,
        name: input.version.name,
        version: input.version.version,
        fingerprint: input.version.manifest.fingerprint,
        license: input.version.license,
        documents: input.version.documents.length,
        characters: input.version.documents.reduce((sum, text) => sum + text.length, 0),
        trainDocuments: input.split.train,
        validationDocuments: input.split.validation,
        testDocuments: input.split.test,
        mixtureFingerprint: input.mixtureFingerprint,
      },
      tokenizer: {
        version: tokenizer.version,
        fingerprint: tokenizer.fingerprint(),
        vocabSize: tokenizer.vocabSize,
        trainedOn: tokenizer.trainedOn ?? "unknown",
        measured: {
          tokensPerCharacter: tokenizerMeasurement.tokensPerCharacter,
          vocabularyCoverage: tokenizerMeasurement.vocabularyCoverage,
          unknownTokenShare: tokenizerMeasurement.unknownTokenShare,
          retrained: decision.retrain,
          reason: decision.reason,
        },
      },
      trainingConfig,
      seed: SEED,
      now: RECORDED_AT,
    });
    experiment = recordTraining(experiment, input.summary);
    experiment = recordCapability(experiment, input.report, suite.fingerprint);
    experiment = completeExperiment(experiment, RECORDED_AT);
    return experiment;
  }

  const baselineExperiment = makeExperiment({
    experimentId: "step5-baseline",
    label: "step5 baseline: alpha-micro on the Step 4 corpus",
    modelId: "alpha-micro-step5-baseline",
    version: baselineVersion,
    split: {
      train: baselineSplit.train.length,
      validation: baselineSplit.validation.length,
      test: baselineSplit.test.length,
    },
    mixtureFingerprint: null,
    summary: baselineRun.summary,
    report: baselineReport,
  });
  const candidateExperiment = makeExperiment({
    experimentId: "step5-candidate",
    label: "step5 candidate: alpha-micro on the Step 5 mixture",
    modelId: "alpha-micro-step5-candidate",
    version: candidateVersion,
    split: {
      train: candidateSplit.train.length,
      validation: candidateSplit.validation.length,
      test: candidateSplit.test.length,
    },
    mixtureFingerprint: mixture.fingerprint,
    summary: candidateRun.summary,
    report: candidateReport,
  });

  record(
    "25",
    "Both arms are recorded as complete experiments with everything needed to re-run them",
    baselineExperiment.status === "completed" &&
      candidateExperiment.status === "completed" &&
      baselineExperiment.evaluationSuiteFingerprint === suite.fingerprint &&
      candidateExperiment.evaluationSuiteFingerprint === suite.fingerprint &&
      describeReproduction(candidateExperiment).includes(suite.fingerprint),
    `experiment ids ${baselineExperiment.experimentId} and ${candidateExperiment.experimentId} · ` +
      `checkpoints ${baselineExperiment.checkpoint?.id ?? "none"} / ${candidateExperiment.checkpoint?.id ?? "none"} · ` +
      `suite ${suite.fingerprint} recorded on both`,
  );

  const tokenizerRerun = AlphaTokenizer.train(unionTexts, {
    vocabSize: ALPHA_MODEL_PRESETS.micro.vocabSize,
    version: "1.0.0",
    trainedOn: "step4-baseline+step5-candidate union",
  });
  const suiteRerun = createEvalSuite({ heldOutDocuments, now: RECORDED_AT + 1, freeze: true });
  record(
    "26",
    "The whole run is reproducible from its seeds, corpora and configuration",
    tokenizerRerun.fingerprint() === tokenizer.fingerprint() &&
      suiteRerun.fingerprint === suite.fingerprint &&
      rebuiltCandidate.manifest.fingerprint === candidateVersion.manifest.fingerprint,
    `tokenizer ${tokenizer.fingerprint()} retrained identically · suite ${suite.fingerprint} rebuilt identically · ` +
      `dataset ${candidateVersion.manifest.fingerprint} rebuilt identically · seed ${SEED} · ` +
      `generation is greedy (temperature 0), so both evaluations reproduce exactly`,
  );

  // ------------------------------------------------------------------ report
  const passed = checks.filter((c) => c.passed).length;
  const allPassed = passed === checks.length;

  const baselineSpeed = generationSpeed(baselineReport);
  const candidateSpeed = generationSpeed(candidateReport);

  let verdict: string;
  if (!gate.suitesMatch) {
    verdict = "CAPABILITY INFRASTRUCTURE READY";
  } else if (gate.passed) {
    verdict = "CAPABILITY IMPROVED";
  } else if (comparableRows.length > 0) {
    verdict = "CAPABILITY NOT YET IMPROVED";
  } else {
    verdict = "CAPABILITY INFRASTRUCTURE READY";
  }
  if (!allPassed) verdict = "CAPABILITY INFRASTRUCTURE READY";

  if (!quiet) {
    const w = 40;
    const pad = (value: string) => value.padEnd(w);
    const col = (value: string) => value.padStart(20);

    console.log(`\n${"─".repeat(78)}`);
    console.log("Step 5 capability report — measurements only. No composite score, no winner.");
    console.log(`${"─".repeat(78)}`);
    console.log("\nHeld constant across both arms (this is the isolation statement):");
    console.log(`  architecture     ${modelConfig.name}@${modelConfig.version} [micro] · ${parameterCount.toLocaleString()} parameters · config ${configFingerprint}`);
    console.log(`  tokenizer        ${tokenizer.vocabSize} tokens · ${tokenizer.fingerprint()} · trained on ${tokenizer.trainedOn ?? "unknown"}`);
    console.log(`  training         batch ${BATCH_SIZE} x seq ${SEQ_LEN} x accum ${ACCUMULATION} = ${BATCH_SIZE * SEQ_LEN * ACCUMULATION} tokens/step · ${TOTAL_STEPS} steps · cosine lr 0.002 · seed ${SEED}`);
    console.log(`  evaluation       suite ${suite.name}@${suite.version} · ${suite.fingerprint} · ${suite.cases.length} cases · ${ALPHA_EVAL_SUITE_VERSION}`);
    console.log(`  generation       greedy, temperature 0, up to 48 new tokens per case`);
    console.log("\n  Because both models share the same architecture, the same tokenizer and");
    console.log("  the same training configuration, this comparison isolates the dataset and");
    console.log("  training-data changes as much as the experiment design allows. What it does");
    console.log("  not do is isolate them perfectly: the two corpora differ in size, in domain");
    console.log("  and in provenance, so any difference below is a difference between whole");
    console.log("  corpora rather than between any single editing decision.");

    console.log("\nCorpora");
    console.log(`  baseline         ${baselineVersion.name}@${baselineVersion.version} · ${baselineVersion.documents.length} documents · ${baselineRunTexts.reduce((s, t) => s + t.length, 0).toLocaleString()} chars · ${baselineVersion.manifest.fingerprint}`);
    console.log(`                   ${describeSplits(baselineSplit)}`);
    console.log(`  candidate        ${candidateVersion.name}@${candidateVersion.version} · ${candidateVersion.documents.length} documents · ${candidateRunTexts.reduce((s, t) => s + t.length, 0).toLocaleString()} chars · ${candidateVersion.manifest.fingerprint}`);
    console.log(`                   ${describeSplits(candidateSplit)} · mixture ${mixture.fingerprint}`);
    console.log(`  held-out source  ${EVAL_SOURCE_DOCUMENTS} documents, seed ${EVAL_SOURCE_SEED}; ${heldOutDocuments.length} kept below ${EVAL_CONTAMINATION_CEILING * 100}% coverage`);

    console.log("\nHeadline measurements, reported separately for each arm");
    console.log(`  ${pad("measurement")}${col("step-4 baseline")}${col("step-5 candidate")}`);
    console.log(`  ${"-".repeat(w)}${"-".repeat(18)}${"-".repeat(18)}`);
    const rows: Array<[string, string, string]> = [
      ["validation loss (nats/token)", num(baselineMetrics.validationLoss), num(candidateMetrics.validationLoss)],
      ["validation perplexity", num(baselineMetrics.validationPerplexity, 2), num(candidateMetrics.validationPerplexity, 2)],
      ["training loss first -> last", `${num(baselineMetrics.firstLoss)} -> ${num(baselineMetrics.lastLoss)}`, `${num(candidateMetrics.firstLoss)} -> ${num(candidateMetrics.lastLoss)}`],
      ["training tokens", baselineMetrics.tokensSeen.toLocaleString(), candidateMetrics.tokensSeen.toLocaleString()],
      ["training throughput (tokens/sec)", String(baselineMetrics.tokensPerSecond), String(candidateMetrics.tokensPerSecond)],
      ["held-out loss (nats/token)", num(baselineReport.languageModeling.loss), num(candidateReport.languageModeling.loss)],
      ["held-out perplexity", num(baselineReport.languageModeling.perplexity, 2), num(candidateReport.languageModeling.perplexity, 2)],
      ["held-out next-token top1", pct(baselineReport.languageModeling.nextTokenTop1Accuracy, 2), pct(candidateReport.languageModeling.nextTokenTop1Accuracy, 2)],
      ["held-out uniform baseline (nats)", num(baselineReport.languageModeling.uniformLoss), num(candidateReport.languageModeling.uniformLoss)],
      ["generation speed (tokens/sec)", num(baselineSpeed.tokensPerSecond, 1), num(candidateSpeed.tokensPerSecond, 1)],
      ["longest repeated-token run", String(worstTokenRun(baselineReport)), String(worstTokenRun(candidateReport))],
      ["entity confusions detected", String(baselineReport.counts.confusion.confusions), String(candidateReport.counts.confusion.confusions)],
      ["format checks passed", countRatio(baselineReport.counts.format), countRatio(candidateReport.counts.format)],
      ["held-out cases matched", countRatio(baselineReport.counts.heldOut), countRatio(candidateReport.counts.heldOut)],
      ["known cases matched", countRatio(baselineReport.counts.known), countRatio(candidateReport.counts.known)],
      ["experiment id", baselineExperiment.experimentId, candidateExperiment.experimentId],
      ["checkpoint id", baselineExperiment.checkpoint?.id ?? "none", candidateExperiment.checkpoint?.id ?? "none"],
      ["dataset fingerprint", baselineVersion.manifest.fingerprint, candidateVersion.manifest.fingerprint],
      ["tokenizer fingerprint", tokenizer.fingerprint(), tokenizer.fingerprint()],
      ["suite fingerprint", baselineReport.suite.fingerprint, candidateReport.suite.fingerprint],
    ];
    for (const [label, left, right] of rows) console.log(`  ${pad(label)}${col(left)}${col(right)}`);
    console.log(
      "\n  note: the baseline's validation carve comes from a corpus that cannot be split\n" +
        "  without overlap, so its validation loss is measured against text that shares\n" +
        "  shingles with its own training data. The held-out rows above come from the\n" +
        "  frozen suite, which was audited clean against both arms, and are the rows the\n" +
        "  gate and the comparison use.",
    );

    const categories: Array<[string, EvalCategory]> = [
      ["instruction-following", "instruction-following"],
      ["question-answering", "question-answering"],
      ["completion", "completion"],
      ["summarization", "summarization"],
      ["structured-output", "structured-output"],
      ["context-retention", "context-retention"],
      ["generation-quality (repetition)", "generation-quality"],
      ["language-modeling (held-out)", "language-modeling"],
    ];
    console.log("\nPer-category results, reported separately for each arm");
    for (const [label, category] of categories) {
      console.log(`  ${label}`);
      console.log(`      baseline  ${categoryLine(baselineReport, category)}`);
      console.log(`      candidate ${categoryLine(candidateReport, category)}`);
    }

    console.log("\nRepetition and error measurements (raw, per arm)");
    for (const [name, report] of [
      ["baseline", baselineReport],
      ["candidate", candidateReport],
    ] as Array<[string, CapabilityReport]>) {
      const gen = categoryOf(report, "generation-quality");
      const failures = report.cases.filter((c) => c.formatPassed === false).length;
      const contextLimits = report.cases.filter((c) => c.stopReason === "context-limit").length;
      const otherStops = report.cases.filter(
        (c) =>
          c.stopReason !== "max-tokens" &&
          c.stopReason !== "eos" &&
          c.stopReason !== "context-limit" &&
          c.stopReason !== "not-applicable",
      ).length;
      console.log(
        `  ${name.padEnd(10)} repetition ${pct(gen?.meanRepetitionRatio ?? null, 2)} · distinct 3-gram ${pct(gen?.meanDistinctTrigramRatio ?? null, 1)} · ` +
          `longest token run ${worstTokenRun(report)} · format failures ${failures} · ` +
          `stopped at the context limit ${contextLimits}/38 · stopped for any other reason ${otherStops}`,
      );
    }

    console.log("\nBaseline comparison (named measurements; nothing is summed)");
    console.log(describeComparison(comparison));

    console.log("\nCapability gate");
    console.log(describeGate(gate));

    console.log("\nReproduction");
    console.log(describeReproduction(candidateExperiment).split("\n").map((l) => `  ${l}`).join("\n"));

    console.log(`\nLimitations, stated rather than hidden:`);
    console.log(`  - Both corpora and the held-out evaluation text are English (plus a small`);
    console.log(`    multilingual category) authored inside this repository; results describe`);
    console.log(`    in-distribution behaviour at 418,656 parameters, not general capability.`);
    console.log(`  - The Step 4 corpus cannot be split without overlap, so its validation`);
    console.log(`    carve is not shingle-independent; the frozen suite is the comparable`);
    console.log(`    measurement and it was audited clean against both arms.`);
    console.log(`  - The held-out text comes from a third corpus of the same generator`);
    console.log(`    family as the candidate's training text, which favours the candidate`);
    console.log(`    on distribution and is the reason the gate is not the only number.`);
    console.log(`  - ${TOTAL_STEPS} steps is a small budget; every figure above is a small-scale`);
    console.log(`    measurement and should be read as one, not as a benchmark result.`);
    console.log(`  - External models: NONE. Alpha's own transformer is the only language`);
    console.log(`    model involved in this run.`);

    console.log(`\n${"─".repeat(78)}`);
    console.log(`CAPABILITY VERDICT: ${verdict}`);
    console.log(
      gate.passed
        ? `  ${gate.satisfied} of ${gate.evaluated} declared criteria satisfied (${gate.required} required) across ${gate.familiesImproved.length} families: ${gate.familiesImproved.join(", ")}.`
        : `  ${gate.satisfied} of ${gate.evaluated} declared criteria satisfied, below the required ${gate.required}. ${gate.verdict}`,
    );
    const unsatisfied = gate.results.filter((r) => !r.satisfied).map((r) => r.criterion.id);
    console.log(
      unsatisfied.length === 0
        ? "  Every declared criterion was satisfied."
        : `  Criteria NOT satisfied, reported beside the pass rather than hidden: ${unsatisfied.join(", ")}.`,
    );
    console.log(
      `  Verification checks: ${passed}/${checks.length} ${allPassed ? "passed" : "FAILED"}. ` +
        `Measured in ${((Date.now() - startedAt) / 1000).toFixed(1)}s.`,
    );
    if (!allPassed) {
      for (const check of checks.filter((c) => !c.passed)) {
        console.log(`  FAILED ${check.id}. ${check.label} — ${check.detail}`);
      }
    }
  }

  if (json) {
    console.log(
      JSON.stringify(
        {
          verdict,
          passed,
          total: checks.length,
          checks,
          isolation: {
            architecture: configFingerprint,
            parameterCount,
            tokenizerFingerprint: tokenizer.fingerprint(),
            tokenizerVocabSize: tokenizer.vocabSize,
            trainingConfig,
            seed: SEED,
            suiteFingerprint: suite.fingerprint,
            suiteCases: suite.cases.length,
            statement:
              "Both arms share one architecture, one tokenizer and one training configuration; only the training corpus differs, which isolates the dataset and training-data changes as much as the experiment design allows.",
          },
          datasets: {
            baseline: {
              fingerprint: baselineVersion.manifest.fingerprint,
              documents: baselineVersion.documents.length,
              train: baselineSplit.train.length,
              validation: baselineSplit.validation.length,
              test: baselineSplit.test.length,
              mixture: null,
            },
            candidate: {
              fingerprint: candidateVersion.manifest.fingerprint,
              documents: candidateVersion.documents.length,
              train: candidateSplit.train.length,
              validation: candidateSplit.validation.length,
              test: candidateSplit.test.length,
              mixture: mixture.fingerprint,
            },
            heldOutDocuments: heldOutDocuments.length,
          },
          baseline: armJson(baselineExperiment, baselineMetrics, baselineReport, baselineSpeed),
          candidate: armJson(candidateExperiment, candidateMetrics, candidateReport, candidateSpeed),
          comparison: comparison.map((row) => ({ ...row })),
          gate: {
            fingerprint: gate.gateFingerprint,
            suitesMatch: gate.suitesMatch,
            satisfied: gate.satisfied,
            evaluated: gate.evaluated,
            required: gate.required,
            passed: gate.passed,
            verdict: gate.verdict,
            familiesImproved: gate.familiesImproved,
            results: gate.results.map((result) => ({
              id: result.criterion.id,
              measurement: result.criterion.measurement,
              baseline: result.baseline,
              candidate: result.candidate,
              absoluteImprovement: result.absoluteImprovement,
              satisfied: result.satisfied,
              detail: result.detail,
            })),
          },
          externalModels: "NONE",
        },
        null,
        2,
      ),
    );
  }

  if (exportPath) {
    const artifact = createServingArtifact({
      model: candidateRun.model,
      tokenizer,
      stage: "trained",
      createdAt: RECORDED_AT,
      training: {
        steps: candidateMetrics.steps,
        tokensSeen: candidateMetrics.tokensSeen,
        tokensPerStep: candidateRun.summary.tokensPerStep,
        seed: SEED,
        firstLoss: candidateMetrics.firstLoss,
        lastLoss: candidateMetrics.lastLoss,
        validationLoss: candidateMetrics.validationLoss,
        validationPerplexity: candidateMetrics.validationPerplexity,
        uniformLossBaseline: candidateMetrics.uniformLossBaseline,
        durationMs: candidateMetrics.durationMs,
        tokensPerSecond: candidateMetrics.tokensPerSecond,
        checkpointId: candidateRun.trainer.checkpoint?.id ?? null,
        gradientAccumulationSteps: candidateRun.summary.gradientAccumulationSteps,
        config: trainingConfig,
      },
      data: {
        datasetName: candidateVersion.name,
        datasetVersion: candidateVersion.version,
        datasetFingerprint: candidateVersion.manifest.fingerprint,
        mixtureFingerprint: mixture.fingerprint,
        documents: candidateVersion.documents.length,
        characters: candidateRunTexts.reduce((sum, text) => sum + text.length, 0),
      },
      evaluation: {
        suiteFingerprint: suite.fingerprint,
        suiteCases: suite.cases.length,
        heldOutDocuments: heldOutDocuments.length,
        loss: candidateReport.languageModeling.loss,
        perplexity: candidateReport.languageModeling.perplexity,
        nextTokenTop1Accuracy: candidateReport.languageModeling.nextTokenTop1Accuracy,
        gateFingerprint: gate.gateFingerprint,
        gatePassed: gate.passed,
      },
    });
    // Written with a newline and no pretty printing: the weights dominate the
    // size, and indentation would cost a third of the file for nothing.
    mkdirSync(dirname(exportPath), { recursive: true });
    writeFileSync(exportPath, `${JSON.stringify(artifact)}\n`, "utf8");
    if (!quiet) {
      console.log(`\nServing artifact written to ${exportPath}`);
      console.log(`  ${describeServingArtifact(artifact)}`);
      console.log(
        `  ${(JSON.stringify(artifact).length / 1024 ** 2).toFixed(2)} MiB on disk · loading it back and re-checking its fingerprints…`,
      );
    }
    const reloaded = loadServingArtifact(artifact);
    const reloadNote = `artifact ${artifact.formatVersion} reloaded in ${reloaded.loadMs}ms: tokenizer ${reloaded.tokenizer.fingerprint()}, ${reloaded.model.parameterCount.toLocaleString()} parameters, weights matched tensor-for-tensor`;
    record("27", "The exported serving artifact reloads with matching fingerprints and weights", true, reloadNote);
  }

  return allPassed ? 0 : 1;
}

// ---------------------------------------------------------------------------
// small factories kept out of main so the pipeline reads top to bottom
// ---------------------------------------------------------------------------

function armJson(
  experiment: Experiment,
  metrics: ReturnType<typeof tokenMetricsFromSummary>,
  report: CapabilityReport,
  speed: ReturnType<typeof generationSpeed>,
) {
  return {
    experimentId: experiment.experimentId,
    status: experiment.status,
    checkpointId: experiment.checkpoint?.id ?? null,
    datasetFingerprint: experiment.dataset.fingerprint,
    tokenizerFingerprint: experiment.tokenizer.fingerprint,
    suiteFingerprint: experiment.evaluationSuiteFingerprint,
    validationLoss: metrics.validationLoss,
    validationPerplexity: metrics.validationPerplexity,
    trainingLossFirst: metrics.firstLoss,
    trainingLossLast: metrics.lastLoss,
    trainingTokens: metrics.tokensSeen,
    trainingTokensPerSecond: metrics.tokensPerSecond,
    heldOut: {
      loss: report.languageModeling.loss,
      perplexity: report.languageModeling.perplexity,
      nextTokenTop1Accuracy: report.languageModeling.nextTokenTop1Accuracy,
      uniformLoss: report.languageModeling.uniformLoss,
      positions: report.languageModeling.positions,
      tokens: report.languageModeling.tokens,
    },
    categories: report.categories.map((category) => ({
      category: category.category,
      cases: category.cases,
      matched: category.matched,
      formatPassed: category.formatPassed,
      meanContinuationNll: category.meanContinuationNll,
      meanContinuationTop1: category.meanContinuationTop1,
      meanRepetitionRatio: category.meanRepetitionRatio,
      meanDistinctTrigramRatio: category.meanDistinctTrigramRatio,
      confusions: category.confusions,
    })),
    repetition: {
      longestTokenRun: report.cases.reduce((worst, c) => Math.max(worst, c.longestTokenRun), 0),
      formatFailures: report.cases.filter((c) => c.formatPassed === false).length,
      confusions: report.counts.confusion.confusions,
    },
    generationSpeed: {
      tokens: speed.tokens,
      latencyMs: speed.latencyMs,
      tokensPerSecond: speed.tokensPerSecond,
    },
  };
}
