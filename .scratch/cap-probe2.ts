import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { buildGeneratedCorpus } from "../src/alpha/datasets/generated-corpus";
import { createProvenanceDocument, validateProvenance, ALPHA_MIX_CATEGORIES, withSourceCounts, documentFingerprint, type ProvenanceDocument, type ProvenanceSource } from "../src/alpha/datasets/provenance";
import { shingleFingerprints } from "../src/alpha/datasets/splits";
import { createEvalSuite, auditSuiteLeakage } from "../src/alpha/evaluation/suite";

const ACQUIRED_AT = 1_760_000_000_000;
const CREATED_AT = 1_759_000_000_000;

function wrap(entries: ReturnType<typeof buildAuthoredCorpus>, prefix: string, sourceId: string): ProvenanceDocument[] {
  return entries.map((entry, index) =>
    createProvenanceDocument({
      documentId: `${prefix}_${String(index).padStart(4, "0")}`,
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
}

const authored = wrap(buildAuthoredCorpus(480, 20260101), "a5", "alpha-authored-corpus");
const sources: ProvenanceSource[] = withSourceCounts(
  [{ id: "alpha-authored-corpus", title: "t", origin: "authored", license: "Alpha-owned", documents: 0 }],
  authored,
);
const v = validateProvenance(authored, sources);
console.log("provenance valid:", v.valid, "dupContent:", v.duplicateContent.length, "issues:", v.issues.slice(0, 3));

const gen = buildGeneratedCorpus(300, 20250930);
const step5Train = authored.map((d) => d.text);
const step4Train = gen.documents;
const reference = [...step5Train, ...step4Train];

const refShingles = new Set<string>();
const refExact = new Set<string>();
for (const text of reference) {
  refExact.add(documentFingerprint(text));
  for (const s of shingleFingerprints(text, 8)) refShingles.add(s);
}
function coverage(text: string): number {
  if (refExact.has(documentFingerprint(text))) return 1;
  const sh = shingleFingerprints(text, 8);
  if (sh.size === 0) return 0;
  let c = 0;
  for (const s of sh) if (refShingles.has(s)) c++;
  return c / sh.size;
}

const evalSource = buildAuthoredCorpus(80, 777001);
const covs = evalSource.map((d) => coverage(d.text));
console.log("eval source coverage distribution:", covs.map((c) => c.toFixed(2)).join(" "));
const kept = evalSource.filter((_, i) => covs[i] < 0.4).map((d) => d.text);
console.log("kept:", kept.length, "of", evalSource.length);

const suite = createEvalSuite({ heldOutDocuments: kept });
console.log("suite cases:", suite.cases.length, suite.fingerprint);
const audit = auditSuiteLeakage(suite, reference);
console.log("audit clean:", audit.clean, "max:", audit.maxCoverage.toFixed(3), "contaminated:", audit.contaminated.length, audit.contaminated.slice(0, 5));
