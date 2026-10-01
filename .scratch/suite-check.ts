/**
 * Scratch check: does the suite build, freeze, and audit cleanly?
 * Run: bun .scratch/suite-check.ts
 */
import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { createProvenanceDocument } from "../src/alpha/datasets/provenance";
import { splitDocuments, describeSplits } from "../src/alpha/datasets/splits";
import {
  createEvalSuite,
  suiteFingerprint,
  assertSuiteFrozen,
  auditSuiteLeakage,
  describeEvalSuite,
} from "../src/alpha/evaluation/suite";

const raw = buildAuthoredCorpus(400, 20260101);
const docs = raw.map((d, i) =>
  createProvenanceDocument({
    documentId: `doc_${String(i).padStart(4, "0")}`,
    text: d.text,
    sourceId: "alpha-authored-corpus",
    origin: "authored",
    license: "Alpha-owned",
    language: d.language,
    category: d.category,
    acquisition: {
      method: "generated-in-repo",
      location: "this-repository",
      acquiredAt: 1_760_000_000_000,
      collectedBy: "alpha:authored-corpus",
    },
    createdAt: 1_759_000_000_000,
  }),
);

const split = splitDocuments(docs, { seed: 1337 });
console.log("splits:", describeSplits(split));
console.log("test docs:", split.test.length, "train docs:", split.train.length);

const testTexts = split.test.map((d) => d.text);
const trainTexts = split.train.map((d) => d.text);

const suite = createEvalSuite({ heldOutDocuments: testTexts, now: 1_760_000_000_000 });
console.log("\n", describeEvalSuite(suite));
console.log("counts:", JSON.stringify(suite.counts));

// Fingerprint stability
const rebuilt = suiteFingerprint(suite.cases);
console.log("fingerprint stable:", rebuilt === suite.fingerprint);

// Frozen check
try {
  assertSuiteFrozen(suite);
  console.log("frozen check: ok");
} catch (e) {
  console.log("frozen check FAILED:", (e as Error).message);
}

// Tamper check
const tampered = { ...suite, cases: [...suite.cases] };
tampered.cases[10] = { ...tampered.cases[10], prompt: "EDITED" };
try {
  assertSuiteFrozen(tampered);
  console.log("tamper check FAILED: edit was not detected");
} catch {
  console.log("tamper check: edit detected");
}

// Leakage audit against training
const report = auditSuiteLeakage(suite, trainTexts, { trainingFingerprint: "train" });
console.log("\nleakage:", report.summary);
console.log("clean:", report.clean);
console.log("max coverage:", (report.maxCoverage * 100).toFixed(1) + "%");
if (!report.clean) {
  for (const c of report.contaminated.slice(0, 12)) {
    console.log(`  - ${c.caseId}: ${(c.coverage * 100).toFixed(0)}%${c.exact ? " EXACT" : ""}`);
  }
}
