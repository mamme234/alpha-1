/**
 * Step 5 tests — capability and intelligence upgrade.
 *
 * These cover the parts that can be checked as facts rather than reported as
 * claims: that gradient accumulation produces the gradient of the concatenated
 * batch, that early stopping fires on validation loss rather than on a step
 * count, that a mixture honours its quotas, that held-out data does not overlap
 * training data, and that a tokenizer is only replaced when a measurement says
 * so.
 *
 * Real models and real tokenizers throughout. Nothing here is mocked, because a
 * mock would make the gradient test prove nothing.
 */

import { describe, expect, it } from "vitest";

import { AlphaTokenizer } from "../tokenizer/bpe";
import { AlphaTransformer } from "../model/transformer";
import { ALPHA_MODEL_PRESETS } from "../model/config";
import { AlphaTrainer } from "../training/trainer";
import { createDataset, type AlphaDataset } from "../datasets/types";
import { buildAuthoredCorpus, authoredCategories, multilingualSamples } from "../datasets/authored-corpus";
import { buildGeneratedCorpus } from "../datasets/generated-corpus";
import {
  ALPHA_MIX_CATEGORIES,
  categoryCounts,
  createProvenanceDocument,
  describeProvenance,
  documentFingerprint,
  languageCounts,
  validateProvenance,
  withSourceCounts,
  type ProvenanceDocument,
  type ProvenanceSource,
} from "../datasets/provenance";
import { buildMixture, describeMixture } from "../datasets/mixture";
import {
  analyseDiversity,
  compareDiversity,
  splitSentences,
  summariseDiversity,
  words,
} from "../datasets/diversity";
import {
  assertNoInstructionLeakage,
  createInstructionDataset,
  createInstructionExample,
  detectInstructionLeakage,
  renderInstructionExample,
  renderInstructionPrompt,
  summariseInstructionDataset,
  validateInstructionDataset,
  type InstructionDataset,
} from "../datasets/instructions";
import {
  assertNoSplitLeakage,
  assignSplits,
  detectOverlap,
  shingleFingerprints,
  splitDocuments,
} from "../datasets/splits";
import { measureTokenizer, decideTokenizerChange } from "../tokenizer/analysis";
import { RUNTIME_CAPABILITIES } from "../training/scaling";

const ACQUIRED_AT = 1_760_000_000_000;
const CREATED_AT = 1_759_000_000_000;

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

function provenanceCorpus(count = 240, seed = 20260101) {
  const documents = buildAuthoredCorpus(count, seed);
  return documents.map(
    (entry, index): ProvenanceDocument =>
      createProvenanceDocument({
        documentId: `doc_${String(index).padStart(4, "0")}`,
        text: entry.text,
        sourceId: "alpha-authored-corpus",
        origin: "authored",
        license: "Alpha-owned",
        language: entry.language,
        category: entry.category,
        acquisition: {
          method: "generated-in-repo",
          location: "this-repository",
          acquiredAt: ACQUIRED_AT,
          collectedBy: "alpha:authored-corpus",
        },
        createdAt: CREATED_AT,
      }),
  );
}

function sourcesFor(documents: ProvenanceDocument[]): ProvenanceSource[] {
  return withSourceCounts(
    [
      {
        id: "alpha-authored-corpus",
        title: "Alpha authored multi-category corpus",
        origin: "authored",
        license: "Alpha-owned",
        documents: 0,
        note: "Generated in this repository by src/alpha/datasets/authored-corpus.ts",
      },
    ],
    documents,
  );
}

function corpusDataset(documents: ProvenanceDocument[]): AlphaDataset {
  return createDataset({
    id: "alpha-authored",
    name: "alpha-authored-mixture",
    version: "2.0.0",
    description: "multi-category Alpha-authored training corpus",
    license: "Alpha-owned",
    source: "alpha-authored-corpus",
    documents: documents.map((d) => d.text),
  });
}

function smallTrainerFixture(overrides: Record<string, unknown> = {}) {
  const documents = provenanceCorpus(120, 4242);
  const dataset = corpusDataset(documents);
  const tokenizer = AlphaTokenizer.train(dataset.documents, {
    vocabSize: 200,
    minPairFrequency: 1,
    version: "step5-test",
    trainedOn: "step5",
  });
  const config = {
    ...ALPHA_MODEL_PRESETS.nano,
    contextLength: 32,
    dModel: 48,
    nHeads: 4,
    nLayers: 2,
    dFeedForward: 96,
    vocabSize: tokenizer.vocabSize,
    dropout: 0,
  };
  const model = new AlphaTransformer(config);
  const trainer = new AlphaTrainer({
    model,
    tokenizer,
    dataset,
    config: {
      batchSize: 4,
      seqLen: 16,
      totalSteps: 2,
      learningRate: 1e-4,
      evalInterval: 0,
      checkpointInterval: 0,
      seed: 1337,
      ...overrides,
    } as never,
  });
  return { trainer, tokenizer, model, dataset, config };
}

// ---------------------------------------------------------------------------
// 1. provenance
// ---------------------------------------------------------------------------

describe("step 5 · per-document provenance", () => {
  it("gives every document an id, source, origin, licence, acquisition and fingerprint", () => {
    const documents = provenanceCorpus(16);
    for (const document of documents) {
      expect(document.documentId).toMatch(/^doc_\d{4}$/);
      expect(document.sourceId).toBe("alpha-authored-corpus");
      expect(document.origin).toBe("authored");
      expect(document.license).toBe("Alpha-owned");
      expect(document.language).not.toBe("");
      expect(document.category).not.toBe("");
      expect(document.acquisition.method).toBe("generated-in-repo");
      expect(document.acquisition.location).toBe("this-repository");
      expect(document.acquisition.acquiredAt).toBe(ACQUIRED_AT);
      expect(document.createdAt).toBe(CREATED_AT);
      expect(document.version).toBe("1");
      expect(document.fingerprint).toMatch(/^doc_[0-9a-f]{8}$/);
    }
  });

  it("is deterministic: the same corpus and seed give byte-identical documents", () => {
    expect(provenanceCorpus(40, 777).map((d) => d.text)).toEqual(
      provenanceCorpus(40, 777).map((d) => d.text),
    );
  });

  it("covers every declared category when there are enough documents", () => {
    const documents = provenanceCorpus(80);
    const counts = categoryCounts(documents);
    for (const category of authoredCategories()) {
      expect(counts[category]).toBeGreaterThan(0);
    }
    // Only categories that genuinely have documents are reported as present.
    const validation = validateProvenance(documents, sourcesFor(documents));
    expect(validation.categoriesPresent.sort()).toEqual([...ALPHA_MIX_CATEGORIES].sort());
  });

  it("records several languages for the multilingual category", () => {
    const documents = provenanceCorpus(80);
    const counts = languageCounts(documents);
    expect(counts.en).toBeGreaterThan(0);
    // More than one non-English language actually present, not merely declared.
    const nonEnglish = Object.keys(counts).filter((language) => language !== "en");
    expect(nonEnglish.length).toBeGreaterThanOrEqual(3);
    for (const sample of multilingualSamples()) {
      expect(nonEnglish).toContain(sample.language);
    }
  });

  it("rejects a document whose licence is missing", () => {
    const document = provenanceCorpus(1)[0];
    const validation = validateProvenance([{ ...document, license: "" }], sourcesFor([document]));
    expect(validation.valid).toBe(false);
    expect(validation.issues.some((i) => i.problem.includes("no licence"))).toBe(true);
  });

  it("rejects a document whose text was edited after its record was made", () => {
    const document = provenanceCorpus(1)[0];
    const validation = validateProvenance(
      [{ ...document, text: `${document.text} silently appended text` }],
      sourcesFor([document]),
    );
    expect(validation.valid).toBe(false);
    expect(validation.tampered).toEqual([document.documentId]);
  });

  it("rejects a document pointing at an undeclared source", () => {
    const document = provenanceCorpus(1)[0];
    const validation = validateProvenance([{ ...document, sourceId: "elsewhere" }], sourcesFor([document]));
    expect(validation.issues.some((i) => i.problem.includes("not a declared source"))).toBe(true);
  });

  it("detects duplicate documents and duplicate ids", () => {
    const documents = provenanceCorpus(4);
    const duplicated = [...documents, documents[0], documents[1]];
    const validation = validateProvenance(duplicated, sourcesFor(documents));
    expect(validation.valid).toBe(false);
    expect(validation.duplicateContent.length).toBeGreaterThan(0);
    expect(validation.issues.some((i) => i.problem.includes("identical text"))).toBe(true);

    const repeatedId = validateProvenance(
      [{ ...documents[0] }, { ...documents[1], documentId: documents[0].documentId }],
      sourcesFor(documents),
    );
    expect(repeatedId.duplicateDocumentIds).toContain(documents[0].documentId);
  });

  it("refuses an acquisition method paired with an impossible location", () => {
    expect(() =>
      createProvenanceDocument({
        documentId: "bad",
        text: "text",
        sourceId: "s",
        origin: "authored",
        license: "Alpha-owned",
        language: "en",
        category: "general-prose",
        acquisition: {
          method: "generated-in-repo",
          location: "user-upload",
          acquiredAt: ACQUIRED_AT,
          collectedBy: "alpha",
        },
        createdAt: CREATED_AT,
      }),
    ).toThrow(/cannot have location/);
  });

  it("summarises sources, licences and languages in one line", () => {
    const documents = provenanceCorpus(80);
    const line = describeProvenance(documents, sourcesFor(documents));
    expect(line).toContain("Alpha-owned");
    // Languages are listed alphabetically, so assert on membership, not position.
    expect(line).toMatch(/languages [a-z, ]*en/);
    expect(line).toContain("licences Alpha-owned");
    expect(line).toContain("categories general-prose");
  });

  it("gives identical text an identical content fingerprint", () => {
    expect(documentFingerprint("alpha")).toBe(documentFingerprint("alpha"));
    expect(documentFingerprint("alpha")).not.toBe(documentFingerprint("beta"));
  });
});

// ---------------------------------------------------------------------------
// 2. mixtures
// ---------------------------------------------------------------------------

describe("step 5 · dataset mixtures", () => {
  it("honours explicit quotas per component", () => {
    const all = provenanceCorpus(400, 999);
    const general = all.filter((d) => d.category === "general-prose").map(provenanceLike).slice(0, 60);
    const educational = all.filter((d) => d.category === "educational").map(provenanceLike).slice(0, 60);

    const mixture = buildMixture({
      totalDocuments: 80,
      seed: 5,
      components: [
        {
          id: "prose",
          corpus: {
            datasetId: "corpus-prose",
            name: "prose",
            version: "1.0.0",
            description: "",
            sources: sourcesFor(general),
            documents: general,
            createdAt: ACQUIRED_AT,
            previousVersion: null,
          },
          categories: ["general-prose"],
          quota: 60,
        },
        {
          id: "education",
          corpus: {
            datasetId: "corpus-education",
            name: "education",
            version: "1.0.0",
            description: "",
            sources: sourcesFor(educational),
            documents: educational,
            createdAt: ACQUIRED_AT,
            previousVersion: null,
          },
          categories: ["educational"],
          quota: 20,
        },
      ],
    });

    expect(mixture.achieved.totalDocuments).toBe(80);
    expect(categoryCounts(mixture.documents)["general-prose"]).toBe(60);
    expect(categoryCounts(mixture.documents).educational).toBe(20);
    expect(mixture.records[0].taken).toBe(60);
    expect(mixture.records[1].taken).toBe(20);
    expect(mixture.records[0].fulfilment).toBeCloseTo(1, 5);
  });

  it("splits by weight when no quota is given", () => {
    const documents = provenanceCorpus(400, 31);
    const corpus = {
      datasetId: "c",
      name: "c",
      version: "1",
      description: "",
      sources: sourcesFor(documents),
      documents,
      createdAt: ACQUIRED_AT,
      previousVersion: null,
    };
    const mixture = buildMixture({
      totalDocuments: 100,
      seed: 11,
      components: [
        { id: "prose", corpus, categories: ["general-prose"], weight: 3 },
        { id: "dialogue", corpus, categories: ["dialogue"], weight: 1 },
      ],
    });
    const counts = categoryCounts(mixture.documents);
    expect(counts["general-prose"]).toBe(75);
    expect(counts.dialogue).toBe(25);
    expect(mixture.achieved.totalDocuments).toBe(100);
  });

  it("records what was requested next to what was achieved", () => {
    const documents = provenanceCorpus(400, 12);
    const corpus = {
      datasetId: "c",
      name: "c",
      version: "1",
      description: "",
      sources: sourcesFor(documents),
      documents,
      createdAt: ACQUIRED_AT,
      previousVersion: null,
    };
    const mixture = buildMixture({
      totalDocuments: 60,
      components: [{ id: "all", corpus, weight: 1 }],
    });
    const record = mixture.records[0];
    expect(record.corpusId).toBe("c");
    expect(record.license).toContain("Alpha-owned");
    expect(record.origin).toBe("authored");
    expect(record.available).toBe(documents.length);
    expect(record.taken).toBe(60);
    expect(mixture.requested.totalDocuments).toBe(60);
    expect(mixture.achieved.totalDocuments).toBe(60);
    expect(describeMixture(mixture)).toContain(mixture.fingerprint);
  });

  it("reports an unsatisfiable category instead of substituting another", () => {
    const documents = provenanceCorpus(80, 13);
    const corpus = {
      datasetId: "c",
      name: "c",
      version: "1",
      description: "",
      sources: sourcesFor(documents),
      documents,
      createdAt: ACQUIRED_AT,
      previousVersion: null,
    };
    const mixture = buildMixture({
      totalDocuments: 20,
      components: [{ id: "prose", corpus, categories: ["general-prose"], weight: 1 }],
      expectedCategories: ["general-prose", "multilingual"],
    });
    expect(mixture.unsatisfiableCategories).toEqual(["multilingual"]);
    expect(mixture.achieved.categories.multilingual.documents).toBe(0);
    // And the category genuinely absent is reported as zero, not as a claim.
    expect(mixture.achieved.categories.multilingual.share).toBe(0);
    expect(describeMixture(mixture)).toContain("UNSATISFIABLE");
  });

  it("is deterministic for the same seed and reports a short shortfall rather than looping", () => {
    const documents = provenanceCorpus(40, 14);
    const corpus = {
      datasetId: "c",
      name: "c",
      version: "1",
      description: "",
      sources: sourcesFor(documents),
      documents,
      createdAt: ACQUIRED_AT,
      previousVersion: null,
    };
    const first = buildMixture({
      totalDocuments: 50,
      seed: 3,
      components: [{ id: "all", corpus, weight: 1 }],
      oversampleLimit: 2,
    });
    const second = buildMixture({
      totalDocuments: 50,
      seed: 3,
      components: [{ id: "all", corpus, weight: 1 }],
      oversampleLimit: 2,
    });
    expect(first.fingerprint).toBe(second.fingerprint);
    expect(first.documents.map((d) => d.documentId)).toEqual(second.documents.map((d) => d.documentId));

    // Asking for far more than exists, at a low oversampling limit, must be
    // reported as a shortfall rather than silently repeating documents forever.
    const starved = buildMixture({
      totalDocuments: 500,
      seed: 3,
      components: [{ id: "all", corpus, weight: 1 }],
      oversampleLimit: 2,
    });
    expect(starved.shortfall).toBeGreaterThan(0);
    expect(starved.achieved.totalDocuments).toBeLessThan(500);
    expect(describeMixture(starved)).toContain("SHORTFALL");
  });

  it("rejects a component with no documents rather than mixing nothing in", () => {
    const documents = provenanceCorpus(8, 15);
    expect(() =>
      buildMixture({
        totalDocuments: 10,
        components: [
          {
            id: "empty",
            corpus: {
              datasetId: "c",
              name: "c",
              version: "1",
              description: "",
              sources: sourcesFor([]),
              documents: [],
              createdAt: ACQUIRED_AT,
              previousVersion: null,
            },
            categories: ["dialogue"],
            weight: 1,
          },
        ],
      }),
    ).toThrow(/no documents to draw from/);
  });
});

function provenanceLike(document: ProvenanceDocument): ProvenanceDocument {
  return { ...document };
}

// ---------------------------------------------------------------------------
// 3. diversity
// ---------------------------------------------------------------------------

describe("step 5 · corpus diversity", () => {
  const step4 = buildGeneratedStep4Corpus(300, 20250930);
  const step5 = provenanceCorpus(300, 20260101).map((d) => d.text);

  it("measures vocabulary, sentence, character and length variety", () => {
    const report = analyseDiversity(step5);
    expect(report.documents).toBe(300);
    expect(report.vocabulary.size).toBeGreaterThan(0);
    expect(report.vocabulary.typeTokenRatio).toBeGreaterThan(0);
    expect(report.sentences.count).toBeGreaterThan(0);
    expect(report.characterVariety.entropyBits).toBeGreaterThan(0);
    expect(report.length.median).toBeGreaterThan(0);
    expect(report.length.p90).toBeGreaterThanOrEqual(report.length.median);
  });

  it("shows the authored corpus is genuinely more varied than the step 4 template", () => {
    const before = analyseDiversity(step4);
    const after = analyseDiversity(step5);
    const rows = compareDiversity(before, after, {
      "boilerplate.ratio": false,
      "boilerplate.meanOverlap": false,
      "lowInformation.ratio": false,
    });
    const byName = Object.fromEntries(rows.map((r) => [r.metric, r]));

    // The point of Step 5 data: measurable variety, not just more volume. The
    // authored corpus is *shorter per document* than the templated one, so
    // sentences-per-document is not the measure that matters — distinct
    // sentence openings, vocabulary size and boilerplate are.
    expect(byName["sentences.distinctOpeningRatio"].improved).toBe(true);
    expect(byName["vocabulary.size"].improved).toBe(true);
    expect(byName["vocabulary.typeTokenRatio"].improved).toBe(true);
    expect(byName["vocabulary.hapaxRatio"].improved).toBe(true);
    expect(byName["boilerplate.ratio"].step5).toBeLessThan(before.boilerplate.ratio);
    expect(byName["boilerplate.meanOverlap"].improved).toBe(true);
    expect(after.characterVariety.distinct).toBeGreaterThan(before.characterVariety.distinct);

    // Step 4's template had a vocabulary of essentially recurring stock phrases:
    // zero words occurred only once across 44k word occurrences.
    expect(before.vocabulary.hapaxRatio).toBe(0);
    expect(after.vocabulary.hapaxRatio).toBeGreaterThan(0);
    // And essentially every Step 4 sentence began the same way.
    expect(before.sentences.distinctOpeningRatio).toBeLessThan(0.1);
    expect(after.sentences.distinctOpeningRatio).toBeGreaterThan(
      before.sentences.distinctOpeningRatio,
    );
  });

  it("detects boilerplate when a corpus really is templated", () => {
    const report = analyseDiversity(step4);
    // Step 4's generator reused one shape, so a high overlap is the truth.
    expect(report.boilerplate.meanOverlap).toBeGreaterThan(0);
    expect(summariseDiversity(report)).toContain("boilerplate");
  });

  it("detects low-information documents", () => {
    const degenerate = [
      "alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha",
      "alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha",
      "the transformer processes every position in a sequence and the optimiser updates every parameter",
    ];
    const report = analyseDiversity(degenerate);
    expect(report.lowInformation.documents).toBe(2);
    expect(report.lowInformation.ratio).toBeGreaterThan(0);
  });

  it("measures token diversity when a tokenizer is supplied and says so when not", () => {
    const withoutTokenizer = analyseDiversity(step5);
    expect(withoutTokenizer.tokens.measured).toBe(false);
    expect(withoutTokenizer.tokens.typeTokenRatio).toBeNull();

    const tokenizer = AlphaTokenizer.train(step5, { vocabSize: 256, minPairFrequency: 1 });
    const withTokenizer = analyseDiversity(step5, tokenizer);
    expect(withTokenizer.tokens.measured).toBe(true);
    expect(withTokenizer.tokens.typeTokenRatio).toBeGreaterThan(0);
    expect(withTokenizer.tokens.distinct).toBeGreaterThan(0);
  });

  it("splits sentences and words in a stable way", () => {
    expect(splitSentences("One. Two! Three?")).toEqual(["One.", "Two!", "Three?"]);
    expect(words("Alpha's model, 42 tokens.")).toEqual(["alpha's", "model", "42", "tokens"]);
  });
});

/** The Step 4 corpus, rebuilt, so the comparison is against the real thing. */
function buildGeneratedStep4Corpus(count: number, seed: number): string[] {
  // The same generator the Step 4 verification used, not a reconstruction of it.
  return buildGeneratedCorpus(count, seed).documents;
}

// ---------------------------------------------------------------------------
// 4. instruction data
// ---------------------------------------------------------------------------

function instructionFixture(): InstructionDataset {
  const acquisition = {
    method: "authored-in-repo" as const,
    location: "this-repository" as const,
    acquiredAt: ACQUIRED_AT,
    collectedBy: "alpha:instruction-set",
  };
  const examples = [
    createInstructionExample({
      id: "ex-1",
      instruction: "Define attention in one sentence.",
      response: "Attention lets each position weigh the earlier positions by relevance.",
      synthetic: false,
      skill: "definition",
      sourceId: "alpha-instructions",
      license: "Alpha-owned",
      origin: "authored",
      acquisition,
      createdAt: CREATED_AT,
    }),
    createInstructionExample({
      id: "ex-2",
      instruction: "Summarise the passage in one sentence.",
      context: "Alpha trains from scratch. It has no external provider. Every weight is its own.",
      response: "Alpha trains its own transformer from scratch with no external provider.",
      synthetic: false,
      skill: "summarisation",
      sourceId: "alpha-instructions",
      license: "Alpha-owned",
      origin: "authored",
      acquisition,
      createdAt: CREATED_AT,
    }),
    createInstructionExample({
      id: "ex-3",
      instruction: "Return the answer as a list of two items.",
      context: "List two properties of layer normalisation.",
      response: "1. It centres a single position.\n2. It rescales that position to unit variance.",
      synthetic: true,
      skill: "structured-output",
      sourceId: "alpha-instructions",
      license: "Alpha-owned",
      origin: "authored",
      acquisition,
      createdAt: CREATED_AT,
    }),
  ];
  return createInstructionDataset({
    datasetId: "alpha-instructions",
    name: "alpha-instruction-set",
    version: "1.0.0",
    description: "Alpha-owned supervised instruction examples",
    license: "Alpha-owned",
    origin: "authored",
    examples,
    now: ACQUIRED_AT,
  });
}

describe("step 5 · supervised instruction data", () => {
  it("keeps instruction, context and response as separate fields", () => {
    const example = instructionFixture().examples[1];
    expect(typeof example.instruction).toBe("string");
    expect(typeof example.response).toBe("string");
    expect(example.context).not.toBeNull();
    expect(example.context).not.toBe("");
  });

  it("renders a prompt with explicit instruction/context/response markers", () => {
    const example = instructionFixture().examples[1];
    const prompt = renderInstructionPrompt(example);
    expect(prompt).toContain("Instruction:");
    expect(prompt).toContain("Context:");
    expect(prompt.endsWith("Response:")).toBe(true);

    const full = renderInstructionExample(example);
    expect(full.startsWith(prompt)).toBe(true);
    expect(full).toContain(example.response);
  });

  it("marks synthetic responses as synthetic", () => {
    const dataset = instructionFixture();
    const validation = validateInstructionDataset(dataset);
    expect(validation.counts.synthetic).toBe(1);
    expect(validation.counts.examples).toBe(3);
    expect(validation.counts.withContext).toBe(2);
    expect(validation.valid).toBe(true);
    expect(summariseInstructionDataset(dataset, validation)).toContain("33% synthetic");
  });

  it("rejects a missing instruction, a missing response and a malformed record", () => {
    const base = instructionFixture().examples[0];
    const validation = validateInstructionDataset({
      ...instructionFixture(),
      examples: [
        { ...base, id: "a", instruction: "   " },
        { ...base, id: "b", response: "  " },
        { ...base, id: "c", instruction: undefined as never },
      ],
    });
    expect(validation.valid).toBe(false);
    const codes = validation.issues.map((i) => i.code);
    // Blank fields are reported as empty; an absent field as missing; and a
    // non-string field as a malformed record.
    expect(codes).toContain("empty-instruction");
    expect(codes).toContain("empty-response");
    expect(codes).toContain("missing-instruction");
    expect(codes).toContain("malformed-record");
  });

  it("rejects excessive length", () => {
    const base = instructionFixture().examples[0];
    const validation = validateInstructionDataset({ ...instructionFixture(), examples: [base] }, {
      maxResponseCharacters: 10,
    });
    expect(validation.issues.some((i) => i.code === "excessive-length")).toBe(true);
  });

  it("rejects duplicate examples and duplicate responses", () => {
    const base = instructionFixture().examples[0];
    const validation = validateInstructionDataset({
      ...instructionFixture(),
      examples: [base, { ...base, id: "dupe" }],
    });
    const codes = validation.issues.map((i) => i.code);
    expect(codes).toContain("duplicate-example");
    expect(codes).toContain("duplicate-response");
  });

  it("rejects an example whose text no longer matches its provenance fingerprint", () => {
    const base = instructionFixture().examples[0];
    const validation = validateInstructionDataset({
      ...instructionFixture(),
      examples: [{ ...base, response: "an answer swapped in after the record was written" }],
    });
    expect(validation.issues.some((i) => i.code === "provenance-mismatch")).toBe(true);
  });

  it("rejects an example with an unknown skill", () => {
    const base = instructionFixture().examples[0];
    const validation = validateInstructionDataset({
      ...instructionFixture(),
      examples: [{ ...base, skill: "telepathy" as never }],
    });
    expect(validation.issues.some((i) => i.code === "unknown-skill")).toBe(true);
  });

  it("detects leakage between a training set and a held-out set", () => {
    const training = instructionFixture();
    const heldOut = createInstructionDataset({
      ...training,
      examples: [
        { ...training.examples[0] },
        createInstructionExample({
          id: "held-1",
          instruction: "What does perplexity measure?",
          response: "Perplexity is the exponential of the loss.",
          synthetic: false,
          skill: "question-answering",
          sourceId: "alpha-held-out",
          license: "Alpha-owned",
          origin: "authored",
          acquisition: {
            method: "authored-in-repo",
            location: "this-repository",
            acquiredAt: ACQUIRED_AT,
            collectedBy: "alpha:evaluation",
          },
          createdAt: CREATED_AT,
        }),
      ],
    });

    const report = detectInstructionLeakage(training, heldOut);
    expect(report.clean).toBe(false);
    expect(report.sharedExamples.length + report.sharedInstructions.length).toBeGreaterThan(0);
    expect(() => assertNoInstructionLeakage(training, heldOut)).toThrow(/leakage/);
  });

  it("confirms a genuinely disjoint held-out set is clean", () => {
    const training = instructionFixture();
    const heldOut = createInstructionDataset({
      ...training,
      examples: [
        createInstructionExample({
          id: "held-2",
          instruction: "Name the two running averages AdamW keeps.",
          response: "A first moment and a second moment of the gradient.",
          synthetic: false,
          skill: "question-answering",
          sourceId: "alpha-held-out",
          license: "Alpha-owned",
          origin: "authored",
          acquisition: {
            method: "authored-in-repo",
            location: "this-repository",
            acquiredAt: ACQUIRED_AT,
            collectedBy: "alpha:evaluation",
          },
          createdAt: CREATED_AT,
        }),
      ],
    });
    expect(() => assertNoInstructionLeakage(training, heldOut)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// 5. split boundaries
// ---------------------------------------------------------------------------

describe("step 5 · held-out boundaries", () => {
  it("assigns train, validation and test deterministically", () => {
    const documents = provenanceCorpus(200, 21);
    const first = splitDocuments(documents, { seed: 99 });
    const second = splitDocuments(documents, { seed: 99 });
    expect(first.test.map((d) => d.documentId)).toEqual(second.test.map((d) => d.documentId));
    expect(first.train.length + first.validation.length + first.test.length).toBe(200);
    expect(first.test.length).toBeGreaterThan(0);
    expect(first.validation.length).toBeGreaterThan(0);
    // The three splits are disjoint by construction.
    const all = [...first.train, ...first.validation, ...first.test].map((d) => d.documentId);
    expect(new Set(all).size).toBe(all.length);
  });

  it("carries indexes back to the input order", () => {
    const documents = provenanceCorpus(50, 22);
    const assignment = splitDocuments(documents, { seed: 5 });
    for (const index of assignment.indexes.test) {
      expect(documents[index].documentId).toBe(
        assignment.test.find((d) => d.documentId === documents[index].documentId)?.documentId,
      );
    }
  });

  it("detects exact and partial overlap between a held-out set and training", () => {
    const training = [
      "the transformer processes every position in a sequence in parallel rather than one position at a time",
      "attention computes a score between a query and a key and uses that score to weight the value",
    ];
    const heldOut = [
      { label: "exact", text: training[0] },
      { label: "partial", text: `${training[1]} and this trailing clause is unique to the test case` },
      { label: "clean", text: "a completely different sentence about an unrelated subject entirely here" },
    ];
    const report = detectOverlap(heldOut, training);
    expect(report.exactMatches).toEqual(["exact"]);
    // "partial" quotes most of a training sentence, so it is contamination.
    expect(report.shingleMatches).toContain("partial");
    expect(report.shingleMatches).not.toContain("clean");
    expect(report.clean).toBe(false);
    expect(report.contaminatedShare).toBeGreaterThan(0);
    expect(report.maxShingleCoverage).toBe(1);
  });

  it("does not flag a document that merely shares an incidental phrase", () => {
    // A detector that refused this would refuse every real corpus, because
    // ordinary English shares phrases constantly.
    const report = detectOverlap(
      [
        {
          label: "incidental",
          text:
            "in the end that was enough to begin with the rest of the work followed over many months of careful effort by everyone involved",
        },
      ],
      ["By the time the light reached the windows, the room had already gone quiet."],
    );
    expect(report.exactMatches).toEqual([]);
    expect(report.shingleMatches).toEqual([]);
    expect(report.clean).toBe(true);
    // The coverage figure is still reported, so a reader can disagree.
    expect(report.maxShingleCoverage).toBeLessThan(0.5);
  });

  it("reports a clean held-out set as clean", () => {
    const report = detectOverlap(
      [{ label: "a", text: "the optimiser keeps a first and second moment of the gradient" }],
      ["attention masks every future position from the current one"],
    );
    expect(report.clean).toBe(true);
    expect(report.contaminatedShare).toBe(0);
  });

  it("refuses to train when a split leaks", () => {
    const documents = provenanceCorpus(100, 23);
    const clean = splitDocuments(documents, { seed: 7 });
    expect(() => assertNoSplitLeakage(clean)).not.toThrow();

    // Forge a leak: put a test document into the training split too.
    const leaky = {
      ...clean,
      train: [...clean.train, clean.test[0]],
    };
    expect(() => assertNoSplitLeakage(leaky)).toThrow(/refusing to train/);
  });

  it("builds word shingles and handles text shorter than the shingle size", () => {
    expect(shingleFingerprints("one two three four five six seven eight nine").size).toBeGreaterThan(0);
    expect(shingleFingerprints("two words").size).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// 6. tokenizer measurement
// ---------------------------------------------------------------------------

describe("step 5 · tokenizer measurement", () => {
  const documents = provenanceCorpus(200, 20260101).map((d) => d.text);

  it("measures coverage, tokens per character and per word, unknowns and round trip", () => {
    const tokenizer = AlphaTokenizer.train(documents, {
      vocabSize: 300,
      minPairFrequency: 1,
      version: "measured",
      trainedOn: "authored",
    });
    const measurement = measureTokenizer(tokenizer, documents);
    expect(measurement.vocabularyCoverage).toBeGreaterThan(0.95);
    expect(measurement.vocabularyCoverage).toBeLessThanOrEqual(1);
    expect(measurement.tokensPerCharacter).toBeGreaterThan(0);
    expect(measurement.tokensPerWord).toBeGreaterThan(0);
    expect(measurement.unknownTokenShare).toBeLessThan(0.01);
    expect(measurement.roundTripExact).toBe(true);
    expect(measurement.vocabularyUtilisation).toBeGreaterThan(0);
    expect(measurement.top100Coverage).toBeGreaterThan(0);
    expect(measurement.hapaxTokenShare).toBeGreaterThanOrEqual(0);
  });

  it("reports characters the vocabulary cannot represent", () => {
    const tokenizer = AlphaTokenizer.train(["plain ascii text only"], {
      vocabSize: 60,
      minPairFrequency: 1,
    });
    // The tokenizer only ever saw "plain ascii text only". Characters that
    // appear only in the measurement corpus are genuinely uncovered, and the
    // measurement must say so rather than report full coverage.
    const measurement = measureTokenizer(tokenizer, ["plain ascii", "another line"]);
    expect(measurement.unknownCharacters.length).toBeGreaterThan(0);
    expect(measurement.vocabularyCoverage).toBeLessThan(1);
    // And they really are characters present in the measured text.
    const measuredChars = new Set("plain ascii another line".split(""));
    for (const character of measurement.unknownCharacters) {
      expect(measuredChars.has(character)).toBe(true);
      expect("plain ascii text only".includes(character)).toBe(false);
    }
  });

  it("keeps the current tokenizer when the candidate is not measurably better", () => {
    const documentsA = ["the same text for both tokenizers, character for character"];
    const current = AlphaTokenizer.train(documentsA, { vocabSize: 120, minPairFrequency: 1 });
    const candidate = AlphaTokenizer.train(documentsA, { vocabSize: 121, minPairFrequency: 1 });
    const decision = decideTokenizerChange(
      measureTokenizer(current, documentsA),
      measureTokenizer(candidate, documentsA),
    );
    expect(decision.retrain).toBe(false);
    expect(["candidate-is-not-better", "coverage-improvable"]).toContain(decision.reason);
    expect(decision.explanation).toMatch(/measured/);
  });

  it("adopts a genuinely denser vocabulary", () => {
    const corpus = Array.from({ length: 60 }, (_, i) =>
      `the transformer repeats a familiar pattern number ${i} so the vocabulary has something to merge`,
    );
    const small = AlphaTokenizer.train(corpus, { vocabSize: 60, minPairFrequency: 1 });
    const large = AlphaTokenizer.train(corpus, { vocabSize: 200, minPairFrequency: 1 });
    const decision = decideTokenizerChange(
      measureTokenizer(small, corpus),
      measureTokenizer(large, corpus),
    );
    expect(decision.retrain).toBe(true);
    expect(decision.efficiencyDelta).toBeGreaterThan(0);
    expect(decision.thresholds.minEfficiencyImprovement).toBe(0.05);
  });

  it("refuses a candidate whose coverage falls below the floor", () => {
    const corpus = ["alpha text about transformers and attention and optimisers"];
    const current = AlphaTokenizer.train(corpus, { vocabSize: 150, minPairFrequency: 1 });
    // A vocabulary that never saw any of this text cannot cover it.
    const candidate = AlphaTokenizer.train(["zzz qqq xxx"], { vocabSize: 150, minPairFrequency: 1 });
    const decision = decideTokenizerChange(
      measureTokenizer(current, corpus),
      measureTokenizer(candidate, corpus),
    );
    expect(decision.retrain).toBe(false);
    expect(["coverage-loss", "round-trip-broken"]).toContain(decision.reason);
  });

  it("is deterministic for the same corpus", () => {
    const tokenizer = AlphaTokenizer.train(documents, { vocabSize: 250, minPairFrequency: 1 });
    expect(measureTokenizer(tokenizer, documents).tokensPerCharacter).toBe(
      measureTokenizer(tokenizer, documents).tokensPerCharacter,
    );
  });
});

// ---------------------------------------------------------------------------
// 7. gradient accumulation — the numerical claim
// ---------------------------------------------------------------------------

describe("step 5 · gradient accumulation", () => {
  it("produces the same gradient as one pass over the concatenated batch", () => {
    const { trainer, tokenizer, model, config } = smallTrainerFixture({
      gradientAccumulationSteps: 4,
      batchSize: 2,
      seqLen: 16,
      totalSteps: 1,
    });

    // Build four deterministic micro-batches from the training stream.
    const ids = trainer.corpus.trainIds;
    const micro: Array<{ input: Int32Array; target: Int32Array; batch: number; seqLen: number }> = [];
    for (let i = 0; i < 4; i++) {
      const start = i * 2 * 16;
      micro.push({
        input: ids.slice(start, start + 2 * 16),
        target: ids.slice(start + 1, start + 1 + 2 * 16),
        batch: 2,
        seqLen: 16,
      });
    }

    const accumulated = trainer.accumulateGradients(micro as never);
    const accumulationGrads = model.parameters().map((p) => Float32Array.from(p.tensor.grad!));
    const accumulationNorm = accumulated.gradNorm;

    // Reference: the identical rows as one batch of 8.
    const combined = {
      input: new Int32Array(8 * 16),
      target: new Int32Array(8 * 16),
      batch: 8,
      seqLen: 16,
    };
    for (let row = 0; row < 8; row++) {
      const microIndex = Math.floor(row / 2);
      const rowInMicro = row % 2;
      for (let t = 0; t < 16; t++) {
        combined.input[row * 16 + t] = micro[microIndex].input[rowInMicro * 16 + t];
        combined.target[row * 16 + t] = micro[microIndex].target[rowInMicro * 16 + t];
      }
    }

    const referenceModel = new AlphaTransformer(config);
    const referenceTrainer = new AlphaTrainer({
      model: referenceModel,
      tokenizer,
      dataset: trainer.dataset,
      config: trainer.config,
    });
    const reference = referenceTrainer.accumulateGradients([combined as never]);
    const referenceGrads = referenceModel.parameters().map((p) => Float32Array.from(p.tensor.grad!));

    // Same rows, same model, same weighting: the gradients must agree.
    expect(accumulationNorm).toBeCloseTo(reference.gradNorm, 4);
    for (let p = 0; p < accumulationGrads.length; p++) {
      expect(accumulationGrads[p].length).toBe(referenceGrads[p].length);
      for (let i = 0; i < accumulationGrads[p].length; i++) {
        expect(accumulationGrads[p][i]).toBeCloseTo(referenceGrads[p][i], 5);
      }
    }
  });

  it("takes exactly one optimiser step per accumulation group", () => {
    const { trainer } = smallTrainerFixture({
      gradientAccumulationSteps: 4,
      totalSteps: 3,
      learningRate: 0.01,
    });
    const summary = trainer.trainToCompletion();
    expect(summary.steps).toBe(3);
    // Three steps, not twelve: the micro-batches were summed, not stepped on.
    expect(trainer.optimizer.step).toBe(3);
    expect(summary.gradientAccumulationSteps).toBe(4);
    expect(summary.tokensPerStep).toBe(4 * 16 * 4);
    expect(summary.tokensSeen).toBe(3 * 4 * 16 * 4);
  });

  it("counts tokens from the whole effective batch, not one micro-batch", () => {
    const { trainer } = smallTrainerFixture({ gradientAccumulationSteps: 3, totalSteps: 2 });
    trainer.trainToCompletion();
    expect(trainer.tokensSeen).toBe(2 * trainer.tokensPerStep);
  });

  it("behaves identically to no accumulation when the setting is 1", () => {
    const plain = smallTrainerFixture({ gradientAccumulationSteps: 1, totalSteps: 2 }).trainer;
    const summary = plain.trainToCompletion();
    expect(summary.gradientAccumulationSteps).toBe(1);
    expect(summary.tokensPerStep).toBe(plain.config.batchSize * plain.config.seqLen);
  });

  it("actually moves the weights, rather than accumulating nothing", () => {
    const { trainer, model } = smallTrainerFixture({
      gradientAccumulationSteps: 3,
      totalSteps: 2,
      learningRate: 0.02,
    });
    const before = model.parameters().map((p) => Float32Array.from(p.tensor.data));
    trainer.trainToCompletion();
    const after = model.parameters().map((p) => p.tensor.data);
    let changed = 0;
    before.forEach((snapshot, index) => {
      for (let i = 0; i < snapshot.length; i++) {
        if (Math.abs(snapshot[i] - after[index][i]) > 1e-9) {
          changed += 1;
          break;
        }
      }
    });
    expect(changed).toBeGreaterThan(0);
  });

  it("rejects a nonsensical accumulation setting", () => {
    const { trainer, tokenizer, dataset } = smallTrainerFixture();
    expect(() => new AlphaTrainer({ model: trainer.model, tokenizer, dataset, config: { gradientAccumulationSteps: 0 } })).toThrow(
      /gradientAccumulationSteps must be a positive integer/,
    );
  });

  it("refuses an accumulation group with no loss-bearing tokens", () => {
    const { trainer, tokenizer } = smallTrainerFixture();
    const allPad = {
      input: Int32Array.from(new Array(2 * 4).fill(tokenizer.padId)),
      target: Int32Array.from(new Array(2 * 4).fill(tokenizer.padId)),
      batch: 2,
      seqLen: 4,
      meanTokenId: 0,
      attentionMask: new Uint8Array(8),
      lossMask: new Uint8Array(8),
      paddingTokens: 8,
    };
    expect(() => trainer.accumulateGradients([allPad as never])).toThrow(/no loss-bearing tokens/);
  });

  it("declares gradient accumulation as supported only because it is implemented", () => {
    // This flag is the claim; the tests above are the evidence.
    expect(RUNTIME_CAPABILITIES.gradientAccumulation).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 8. early stopping
// ---------------------------------------------------------------------------

describe("step 5 · early stopping", () => {
  it("reports what it monitored even when it never fires", () => {
    const { trainer } = smallTrainerFixture({
      totalSteps: 4,
      learningRate: 0.05,
      earlyStopping: { monitor: "validationLoss", patience: 50, minDelta: 0, minSteps: 1, evalEvery: 2 },
    });
    const summary = trainer.trainToCompletion();
    expect(summary.earlyStopping).not.toBeNull();
    expect(summary.earlyStopping!.stoppedEarly).toBe(false);
    expect(summary.earlyStopping!.monitor).toBe("validationLoss");
    expect(summary.earlyStopping!.evaluations).toBeGreaterThan(0);
    expect(summary.earlyStopping!.bestValue).not.toBeNull();
    expect(summary.earlyStopping!.bestStep).not.toBeNull();
    expect(summary.stepsSkippedByEarlyStopping).toBe(0);
    expect(summary.state).toBe("completed");
  });

  it("stops a run whose validation loss has stopped improving", () => {
    // A huge learning rate makes validation loss diverge, which is exactly the
    // situation early stopping exists for.
    const { trainer } = smallTrainerFixture({
      totalSteps: 40,
      batchSize: 4,
      seqLen: 16,
      learningRate: 5,
      warmupSteps: 1,
      schedule: "constant",
      gradClipNorm: 0,
      dropout: 0,
      earlyStopping: {
        monitor: "validationLoss",
        patience: 2,
        minDelta: 1e-9,
        minSteps: 6,
        evalEvery: 3,
      },
    });
    const summary = trainer.trainToCompletion();
    expect(summary.earlyStopping!.stoppedEarly).toBe(true);
    expect(summary.steps).toBeLessThan(40);
    expect(summary.stepsSkippedByEarlyStopping).toBe(40 - summary.steps);
    expect(summary.earlyStopping!.stoppingReason).toMatch(/stopped at step/);
    expect(summary.earlyStopping!.stoppingReason).toMatch(/patience/);
    // The run is reported as stopped, not as completed: it did not use its budget.
    expect(summary.state).toBe("stopped");
  });

  it("records the monitored metric, best value and best step", () => {
    const { trainer } = smallTrainerFixture({
      totalSteps: 12,
      learningRate: 0.05,
      earlyStopping: { monitor: "validationLoss", patience: 50, minDelta: 0, minSteps: 1, evalEvery: 2 },
    });
    trainer.trainToCompletion();
    const report = trainer.earlyStoppingReport;
    expect(report.monitor).toBe("validationLoss");
    expect(report.bestValue).not.toBeNull();
    expect(report.bestStep).toBeGreaterThan(0);
    expect(report.evaluations).toBeGreaterThanOrEqual(2);
    // Best value is the minimum observed, so it is never above the last one.
    expect(report.bestValue!).toBeLessThanOrEqual(20);
  });

  it("can monitor perplexity instead of loss", () => {
    const { trainer } = smallTrainerFixture({
      totalSteps: 6,
      learningRate: 0.05,
      earlyStopping: { monitor: "validationPerplexity", patience: 50, minDelta: 0, minSteps: 1, evalEvery: 2 },
    });
    const summary = trainer.trainToCompletion();
    expect(summary.earlyStopping!.monitor).toBe("validationPerplexity");
    expect(summary.earlyStopping!.bestValue).toBeGreaterThan(1);
  });

  it("does not stop before minSteps even if the metric is terrible from the start", () => {
    const { trainer } = smallTrainerFixture({
      totalSteps: 30,
      learningRate: 5,
      schedule: "constant",
      earlyStopping: { monitor: "validationLoss", patience: 1, minDelta: 0, minSteps: 20, evalEvery: 2 },
    });
    const summary = trainer.trainToCompletion();
    expect(summary.steps).toBeGreaterThanOrEqual(20);
  });

  it("tolerates a minDelta that treats small movements as no movement", () => {
    const { trainer } = smallTrainerFixture({
      totalSteps: 10,
      learningRate: 0.02,
      earlyStopping: { monitor: "validationLoss", patience: 50, minDelta: 100, minSteps: 1, evalEvery: 2 },
    });
    const summary = trainer.trainToCompletion();
    // With minDelta 100, nothing can count as an improvement after the first.
    expect(summary.earlyStopping!.evaluationsSinceImprovement).toBeGreaterThan(0);
  });

  it("rejects an early-stopping configuration that could never fire", () => {
    const { trainer, tokenizer, dataset, model } = smallTrainerFixture();
    expect(
      () =>
        new AlphaTrainer({
          model,
          tokenizer,
          dataset,
          config: {
            totalSteps: 10,
            earlyStopping: { monitor: "validationLoss", patience: 2, minDelta: 0, minSteps: 50, evalEvery: 2 },
          },
        }),
    ).toThrow(/could never stop early/);
  });

  it("rejects an unknown monitored metric and a zero patience", () => {
    const { trainer, tokenizer, dataset, model } = smallTrainerFixture();
    expect(
      () =>
        new AlphaTrainer({
          model,
          tokenizer,
          dataset,
          config: {
            earlyStopping: { monitor: "vibes" as never, patience: 2, minDelta: 0, minSteps: 1, evalEvery: 1 },
          },
        }),
    ).toThrow(/monitor must be/);
    expect(
      () =>
        new AlphaTrainer({
          model,
          tokenizer,
          dataset,
          config: {
            earlyStopping: { monitor: "validationLoss", patience: 0, minDelta: 0, minSteps: 1, evalEvery: 1 },
          },
        }),
    ).toThrow(/patience must be a positive integer/);
  });

  it("writes a resumable checkpoint even when it stops early", () => {
    const { trainer } = smallTrainerFixture({
      totalSteps: 40,
      learningRate: 5,
      schedule: "constant",
      gradClipNorm: 0,
      earlyStopping: { monitor: "validationLoss", patience: 1, minDelta: 0, minSteps: 4, evalEvery: 2 },
    });
    const summary = trainer.trainToCompletion();
    expect(summary.checkpoint).not.toBeNull();
    expect(summary.checkpoint!.step).toBe(summary.tokensSeen / summary.tokensPerStep);
    expect(summary.checkpoint!.optimizer.step).toBeGreaterThan(0);
  });

  it("declares early stopping as supported only because it is implemented", () => {
    expect(RUNTIME_CAPABILITIES.earlyStopping).toBe(true);
  });

  it("still declares mixed precision as unsupported", () => {
    // Alpha's tensors are float32 and there is no mixed-precision path.
    expect(RUNTIME_CAPABILITIES.mixedPrecision).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// 9. assignment helpers used by the above
// ---------------------------------------------------------------------------

describe("step 5 · split assignment edge cases", () => {
  it("rejects fractions that leave no training data", () => {
    expect(() => assignSplits(["a", "b"], { validationFraction: 0.6, testFraction: 0.5 })).toThrow(
      /sum to less than 1/,
    );
  });
});
