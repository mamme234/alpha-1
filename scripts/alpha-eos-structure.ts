/**
 * Alpha — WHY the model predicts EOS (read-only, no weights touched).
 *
 * Measures the structure of the training stream the checkpoint was fitted on:
 * how many tokens each document contributes, how often EOS occurs, and where
 * EOS falls inside a 256-token training window. This explains the Stage 4
 * result (EOS = argmax after a prompt ending in ".").
 *
 *   bun scripts/alpha-eos-structure.ts
 */

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

import { parseCheckpoint } from "../src/alpha/training/checkpoint";
import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { buildGeneratedCorpus } from "../src/alpha/datasets/generated-corpus";
import { buildStep7Corpus } from "../src/alpha/datasets/step7-corpus";
import type { AlphaDataset } from "../src/alpha/datasets/types";
import { encodeCorpus } from "../src/alpha/datasets/corpus";

const ROOT = join(__dirname, "..");
const CKPT = join(ROOT, "src/alpha/experiments/language/language-mv2-final.alpha-ckpt.json");

function pct(n: number, d: number): string {
  return `${((n / d) * 100).toFixed(3)}%`;
}

function main(): void {
  const ck = parseCheckpoint(readFileSync(CKPT, "utf8"));
  const tokenizer = AlphaTokenizer.fromJSON(ck.tokenizer.snapshot);
  const eos = tokenizer.eosId;
  const bos = tokenizer.bosId;

  // The exact mixture scripts/alpha-language.ts trained on.
  const authoredA = buildAuthoredCorpus(2400, 20260101).map((d) => d.text);
  const authoredB = buildAuthoredCorpus(1200, 20260202).map((d) => d.text);
  const generated = buildGeneratedCorpus(600, 20250930) as unknown as { documents?: unknown[] };
  const generatedDocs = (generated.documents ?? []).map((d) =>
    typeof d === "string" ? d : (d as { text: string }).text,
  );
  const step7Docs = buildStep7Corpus().documents.map((d) => d.text);
  const proseFiles = [
    join(ROOT, "README.md"),
    join(ROOT, "CONTRIBUTING.md"),
    ...readdirSync(join(ROOT, "docs"))
      .filter((f) => f.endsWith(".md"))
      .map((f) => join(ROOT, "docs", f)),
  ];
  const prose = proseFiles.map((f) => readFileSync(f, "utf8")).join("\n\n");
  const documents = [...authoredA, ...authoredB, ...generatedDocs, ...step7Docs, prose];

  const dataset: AlphaDataset = {
    id: "alpha-language-mixture",
    name: "alpha-language-mixture",
    version: "1.0.0",
    description: "Structural re-derivation of the trained mixture (read-only).",
    license: "Alpha-owned",
    source: "authored for Alpha",
    documents,
  };
  const corpus = encodeCorpus(dataset, tokenizer, { validationFraction: 0.1 });

  const trainDocs = corpus.trainDocuments.map((d) => Array.from(d));
  const lens = trainDocs.map((d) => d.length).sort((a, b) => a - b);
  const total = trainDocs.reduce((s, d) => s + d.length, 0);

  console.log("TRAINING STREAM STRUCTURE (re-derived, read-only)\n");
  console.log(`documents (train split) : ${trainDocs.length}`);
  console.log(`total train tokens      : ${total}`);
  console.log(`mean tokens per doc     : ${(total / trainDocs.length).toFixed(1)}`);
  console.log(`median                  : ${lens[Math.floor(lens.length / 2)]}`);
  console.log(`p10 / p90               : ${lens[Math.floor(lens.length * 0.1)]} / ${lens[Math.floor(lens.length * 0.9)]}`);
  console.log(`min / max               : ${lens[0]} / ${lens[lens.length - 1]}`);
  console.log(`model contextLength     : ${ck.config.contextLength}`);

  const ctx = ck.config.contextLength;
  const fits = trainDocs.filter((d) => d.length <= ctx).length;
  console.log(`\ndocs that FIT in one ${ctx}-token window: ${fits} / ${trainDocs.length} (${pct(fits, trainDocs.length)})`);
  const fitsHalf = trainDocs.filter((d) => d.length <= ctx / 2).length;
  console.log(`docs <= ${ctx / 2} tokens                    : ${fitsHalf} (${pct(fitsHalf, trainDocs.length)})`);

  // How often does a document end with a sentence-final period before EOS?
  let endsPeriod = 0;
  let endsOther = 0;
  for (const d of trainDocs) {
    if (d[d.length - 1] !== eos) continue;
    const prev = d[d.length - 2];
    if (prev === undefined) continue;
    const s = tokenizer.tokenForId(prev);
    if (s === ".") endsPeriod++;
    else endsOther++;
  }
  console.log(`\ndocs whose LAST content token is "." : ${endsPeriod} (${pct(endsPeriod, endsPeriod + endsOther)})`);
  console.log(`docs whose last content token is other: ${endsOther}`);

  // EOS density in the raw stream.
  let eosCount = 0;
  let bosCount = 0;
  for (const id of corpus.trainIds) {
    if (id === eos) eosCount++;
    if (id === bos) bosCount++;
  }
  console.log(`\nEOS occurrences in stream: ${eosCount} = ${pct(eosCount, corpus.trainIds.length)} (1 per ${(corpus.trainIds.length / eosCount).toFixed(0)} tokens)`);
  console.log(`BOS occurrences in stream: ${bosCount} = ${pct(bosCount, corpus.trainIds.length)}`);

  // Where does EOS land inside a random 256-token window?
  const positions: number[] = [];
  const stream = corpus.trainIds;
  let windows = 0;
  for (let start = 0; start + ctx <= stream.length && windows < 4000; start += ctx) {
    for (let t = 0; t < ctx; t++) if (stream[start + t] === eos) positions.push(t);
    windows++;
  }
  const hist = new Array(ctx).fill(0);
  for (const p of positions) hist[p]++;
  const top = hist
    .map((count, pos) => ({ pos, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);
  console.log(`\nEOS position inside a ${ctx}-token training window (${windows} sampled windows):`);
  for (const t of top) console.log(`   position ${String(t.pos).padStart(3)}: ${t.count} times`);
  const meanPos = positions.reduce((s, p) => s + p, 0) / positions.length;
  console.log(`mean EOS position: ${meanPos.toFixed(1)} / ${ctx - 1}`);

  // The decisive measurement: within a window, how predictable is EOS from
  // "the previous token was a period"?
  let afterPeriod = 0;
  let afterPeriodEos = 0;
  let otherPrev = 0;
  let otherPrevEos = 0;
  for (let start = 0; start + ctx <= stream.length && start < 40 * ctx; start += ctx) {
    for (let t = 1; t < ctx; t++) {
      const prevTok = tokenizer.tokenForId(stream[start + t - 1]);
      const cur = stream[start + t];
      if (prevTok === ".") {
        afterPeriod++;
        if (cur === eos) afterPeriodEos++;
      } else {
        otherPrev++;
        if (cur === eos) otherPrevEos++;
      }
    }
  }
  console.log(`\nP(EOS | previous token is "."):  ${pct(afterPeriodEos, afterPeriod)}  (n=${afterPeriod})`);
  console.log(`P(EOS | previous token other):  ${pct(otherPrevEos, otherPrev)}  (n=${otherPrev})`);
  console.log(
    `\nratio: EOS is ${(afterPeriodEos / Math.max(afterPeriod, 1) / Math.max(otherPrevEos / Math.max(otherPrev, 1), 1e-9)).toFixed(0)}x more likely after "." than otherwise`,
  );

  // Sentences per document: is each doc basically one sentence?
  let oneSentence = 0;
  for (const text of documents) {
    const stripped = text.trim();
    const stops = (stripped.match(/[.!?]/g) ?? []).length;
    if (stops <= 1) oneSentence++;
  }
  console.log(`\ndocuments with <= 1 sentence-final punctuation: ${oneSentence} / ${documents.length} (${pct(oneSentence, documents.length)})`);

  console.log(`\nNo weights were modified. Read-only.`);
}

main();