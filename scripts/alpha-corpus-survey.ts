/**
 * Alpha — corpus survey for the language training budget.
 *
 * Read-only: builds the proposed Alpha-owned corpus mixture and reports exact
 * sizes and diversity so the training budget is based on measured numbers.
 *
 * Sources (all in-repo, Alpha-owned, no external data):
 *   - authored corpus, two seeds
 *   - generated corpus
 *   - step-7 corpus
 *   - repository prose (README, CONTRIBUTING, docs/*.md)
 *
 * Token counts use the step-7 tokenizer snapshot (vocab 768) as a realistic
 * measuring stick; the actual run would train its own tokenizer on the
 * mixture.
 */

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { buildGeneratedCorpus } from "../src/alpha/datasets/generated-corpus";
import { buildStep7Corpus } from "../src/alpha/datasets/step7-corpus";

const ROOT = join(__dirname, "..");

const step7Artifact = JSON.parse(
  readFileSync(join(ROOT, "src/alpha/serving/step7-artifact.json"), "utf8"),
) as { tokenizer: { snapshot: Parameters<typeof AlphaTokenizer.fromJSON>[0] } };
const tokenizer = AlphaTokenizer.fromJSON(step7Artifact.tokenizer.snapshot);

const proseFiles = [
  join(ROOT, "README.md"),
  join(ROOT, "CONTRIBUTING.md"),
  ...readdirSync(join(ROOT, "docs"))
    .filter((f) => f.endsWith(".md"))
    .map((f) => join(ROOT, "docs", f)),
];
const prose = proseFiles.map((f) => readFileSync(f, "utf8")).join("\n\n");

const authoredA = buildAuthoredCorpus(2400, 20260101).map((d) => d.text);
const authoredB = buildAuthoredCorpus(1200, 20260202).map((d) => d.text);

const generated = buildGeneratedCorpus(600, 20250930) as unknown as {
  documents?: unknown[];
};
const generatedDocs = (generated.documents ?? []).map((d) =>
  typeof d === "string" ? d : (d as { text: string }).text,
);

const step7Docs = buildStep7Corpus().documents.map((d) => d.text);

const sources: [string, string[]][] = [
  ["authored-2400 (seed A)", authoredA],
  ["authored-1200 (seed B)", authoredB],
  ["generated-600", generatedDocs],
  ["step7-corpus", step7Docs],
  ["repo prose (README + docs)", [prose]],
];

let totalTokens = 0;
let totalChars = 0;
for (const [name, docs] of sources) {
  const text = docs.join("\n\n");
  const ids = tokenizer.encode(text);
  totalTokens += ids.length;
  totalChars += text.length;
  console.log(`${name}: ${docs.length} docs · ${text.length} chars · ${ids.length} tokens`);
}

const all = sources.flatMap(([, docs]) => docs).join("\n\n");
const allIds = tokenizer.encode(all);
const uniqueIds = new Set(allIds).size;
const sample = allIds.slice(0, 60000);
const grams = new Set<string>();
for (let i = 0; i + 8 <= sample.length; i++) grams.add(sample.slice(i, i + 8).join(","));
console.log(
  `\nTOTAL: ${sources.reduce((n, [, d]) => n + d.length, 0)} docs · ${totalChars} chars · ${totalTokens} tokens`,
);
console.log(
  `unique token ids used: ${uniqueIds}/${tokenizer.vocabSize} · unique 8-grams in first ${sample.length} tokens: ${(((grams.size / (sample.length - 7)) * 100)).toFixed(1)}%`,
);
