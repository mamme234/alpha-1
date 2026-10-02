import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { buildGeneratedCorpus } from "../src/alpha/datasets/generated-corpus";
import {
  ALPHA_MIX_CATEGORIES,
  createProvenanceDocument,
  documentFingerprint,
  validateProvenance,
  withSourceCounts,
  type ProvenanceCorpus,
  type ProvenanceDocument,
  type ProvenanceSource,
} from "../src/alpha/datasets/provenance";
import { buildMixture, describeMixture } from "../src/alpha/datasets/mixture";
import {
  assertNoSplitLeakage,
  describeSplits,
  splitDocuments,
  shingleFingerprints,
} from "../src/alpha/datasets/splits";
import { createEvalSuite, auditSuiteLeakage } from "../src/alpha/evaluation/suite";
import {
  INSTRUCTION_SKILLS,
  createInstructionDataset,
  createInstructionExample,
  instructionDocuments,
  validateInstructionDataset,
} from "../src/alpha/datasets/instructions";

const ACQUIRED_AT = 1_760_000_000_000;
const CREATED_AT = 1_759_000_000_000;

function wrapAuthored(count: number, seed: number, prefix: string, sourceId: string) {
  const seen = new Set<string>();
  const out: ProvenanceDocument[] = [];
  let dupes = 0;
  buildAuthoredCorpus(count, seed).forEach((entry) => {
    const fp = documentFingerprint(entry.text);
    if (seen.has(fp)) {
      dupes++;
      return;
    }
    seen.add(fp);
    out.push(
      createProvenanceDocument({
        documentId: `${prefix}_${String(out.length).padStart(4, "0")}`,
        text: entry.text,
        sourceId,
        origin: "authored",
        license: "Alpha-owned",
        language: entry.language,
        category: entry.category,
        acquisition: { method: "generated-in-repo", location: "this-repository", acquiredAt: ACQUIRED_AT, collectedBy: `alpha:${prefix}` },
        createdAt: CREATED_AT,
      }),
    );
  });
  return { documents: out, duplicatesRemoved: dupes };
}

let t = Date.now();
const authored = wrapAuthored(480, 20260101, "a5", "alpha-authored-corpus");
console.log("authored", authored.documents.length, "dupes", authored.duplicatesRemoved);
const authoredSources: ProvenanceSource[] = withSourceCounts(
  [{ id: "alpha-authored-corpus", title: "Alpha authored multi-category corpus", origin: "authored", license: "Alpha-owned", documents: 0 }],
  authored.documents,
);
console.log("provenance valid:", validateProvenance(authored.documents, authoredSources).valid);

const mixture = buildMixture({
  components: ALPHA_MIX_CATEGORIES.map((category) => ({
    id: `authored-${category}`,
    corpus: {
      datasetId: "alpha-authored",
      name: "alpha-authored-mixture",
      version: "2.0.0",
      description: "step 5 multi-category authored corpus",
      sources: authoredSources,
      documents: authored.documents.filter((d) => d.category === category),
      createdAt: CREATED_AT,
      previousVersion: null,
    } as ProvenanceCorpus,
    weight: 1,
  })),
  totalDocuments: authored.documents.length,
  seed: 4242,
  expectedCategories: [...ALPHA_MIX_CATEGORIES],
});
console.log("mixture:", describeMixture(mixture));

const candSplit = splitProvenance(mixture.documents);
function splitProvenance(docs: ProvenanceDocument[]) {
  return splitDocuments(docs, { validationFraction: 0.12, testFraction: 0.12, seed: 20260101 });
}
const candAudit = assertNoSplitLeakage(candSplit);
console.log("candidate splits:", describeSplits(candSplit), "val clean", candAudit.validation.clean, "test clean", candAudit.test.clean);

const generated = buildGeneratedCorpus(300, 20250930);
const step4 = wrapGenerated(generated.documents);
function wrapGenerated(texts: string[]): ProvenanceDocument[] {
  return texts.map((text, i) =>
    createProvenanceDocument({
      documentId: `s4_${String(i).padStart(4, "0")}`,
      text,
      sourceId: "alpha-generated",
      origin: "authored",
      license: "Alpha-owned",
      language: "en",
      category: "factual-reference",
      acquisition: { method: "generated-in-repo", location: "this-repository", acquiredAt: ACQUIRED_AT, collectedBy: "alpha:generated-corpus" },
      createdAt: CREATED_AT,
    }),
  );
}
const s4Sources = withSourceCounts([{ id: "alpha-generated", title: "Alpha generated technical corpus", origin: "authored", license: "Alpha-owned", documents: 0 }], step4);
console.log("step4 provenance valid:", validateProvenance(step4, s4Sources).valid);
const baseSplit = splitDocuments(step4, { validationFraction: 0.12, testFraction: 0.12, seed: 20260101 });
const baseAudit = assertNoSplitLeakage(baseSplit);
console.log("baseline splits:", describeSplits(baseSplit), "val clean", baseAudit.validation.clean, "test clean", baseAudit.test.clean);

// instruction component from a separate seed, used only for supervision
const instructionSeed = wrapAuthored(48, 20260102, "i5", "alpha-authored-corpus");
const PHRASINGS = [
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
const examples: ReturnType<typeof createInstructionExample>[] = [];
instructionSeed.documents.forEach((doc, i) => {
  const first = (doc.text.split(/(?<=[.!?])\s+/)[0] ?? doc.text).trim();
  const key = first.toLowerCase();
  if (responseSeen.has(key)) return;
  responseSeen.add(key);
  examples.push(
    createInstructionExample({
      id: `ins-${String(examples.length).padStart(3, "0")}`,
      instruction: PHRASINGS[examples.length % PHRASINGS.length],
      context: doc.text,
      response: first,
      synthetic: true,
      skill: INSTRUCTION_SKILLS[examples.length % INSTRUCTION_SKILLS.length],
      sourceId: "alpha-authored-corpus",
      license: "Alpha-owned",
      origin: "authored",
      acquisition: { method: "generated-in-repo", location: "this-repository", acquiredAt: ACQUIRED_AT, collectedBy: "alpha:instructions" },
      createdAt: CREATED_AT,
    }),
  );
});
const instrDataset = createInstructionDataset({
  datasetId: "alpha-instructions",
  name: "alpha-instruction-set",
  version: "1.0.0",
  description: "synthetic supervised instruction examples authored in this repository",
  license: "Alpha-owned",
  origin: "authored",
  examples,
  now: CREATED_AT,
});
const instrValidation = validateInstructionDataset(instrDataset);
console.log("instruction dataset:", instrDataset.examples.length, "valid:", instrValidation.valid, instrValidation.issues.slice(0, 3));
const instructionDocs = instructionDocuments(instrDataset);

const candRun = [...candSplit.train.map((d) => d.text), ...candSplit.validation.map((d) => d.text), ...instructionDocs];
const baseRun = [...baseSplit.train.map((d) => d.text), ...baseSplit.validation.map((d) => d.text)];
console.log("candRun", candRun.length, "docs", candRun.reduce((s, d) => s + d.length, 0), "chars; baseRun", baseRun.length, "docs", baseRun.reduce((s, d) => s + d.length, 0), "chars");

const reference = [...candRun, ...baseRun];
const refShingles = new Set<string>();
for (const text of reference) for (const s of shingleFingerprints(text, 8)) refShingles.add(s);
const evalSource = buildAuthoredCorpus(80, 777001);
const cov = evalSource.map((d) => {
  const sh = shingleFingerprints(d.text, 8);
  if (sh.size === 0) return 0;
  let c = 0;
  for (const s of sh) if (refShingles.has(s)) c++;
  return c / sh.size;
});
const kept = evalSource.filter((_, i) => cov[i] < 0.4).map((d) => d.text);
console.log("eval docs kept:", kept.length, "of", evalSource.length);

const suite = createEvalSuite({ heldOutDocuments: kept });
console.log("suite:", suite.cases.length, suite.fingerprint);
const audit = auditSuiteLeakage(suite, reference);
console.log("audit clean:", audit.clean, "max:", audit.maxCoverage.toFixed(3), "contaminated:", audit.contaminated);
const auditAll = auditSuiteLeakage(suite, reference, { threshold: 0.0001 });
console.log(
  "all overlapping cases:",
  auditAll.contaminated
    .sort((a, b) => b.coverage - a.coverage)
    .map((c) => `${c.caseId}=${(c.coverage * 100).toFixed(0)}%`)
    .join(" "),
);
console.log("elapsed", ((Date.now() - t) / 1000).toFixed(1) + "s");
