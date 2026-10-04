/**
 * Alpha — language quality diagnosis (read-only, deterministic, no training).
 *
 * Answers one question with measurements: is the corrupted generation a BUG or
 * insufficient training? It loads the exact served artifacts, runs the exact
 * tokenizer and transformer, and prints raw numbers:
 *
 *   1. tokenizer encode/decode round-trips (spacing, punctuation, specials)
 *   2. vocabulary id mapping sanity
 *   3. BOS/EOS/PAD handling
 *   5. causal masking (prefix invariance at runtime)
 *   7. embedding/output weight tying (dynamic check, restored afterwards)
 *   8. next-token distribution + generation collapse statistics
 *  10. training-data repetition statistics
 *  11. model NLL vs uniform / unigram / bigram baselines on held-out text
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { loadServingArtifact, parseServingArtifact } from "../src/alpha/serving/artifact";
import { AlphaInferenceEngine, SAMPLING_PRESETS, type GenerationResult } from "../src/alpha/inference/engine";
import { setGradEnabled } from "../src/alpha/core/tensor";
import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";

const ROOT = join(__dirname, "..");

function loadArtifact(path: string) {
  const artifact = parseServingArtifact(JSON.parse(readFileSync(path, "utf8")));
  const loaded = loadServingArtifact(artifact);
  const engine = new AlphaInferenceEngine({
    model: loaded.model,
    tokenizer: loaded.tokenizer,
    stage: artifact.stage,
  });
  return { artifact, engine, model: loaded.model, tokenizer: loaded.tokenizer };
}

const step5 = loadArtifact(join(ROOT, "src/alpha/serving/step5-artifact.json"));
const step7 = loadArtifact(join(ROOT, "src/alpha/serving/step7-artifact.json"));

function line(title: string) {
  console.log(`\n=== ${title} ===`);
}

/* 1. Tokenizer round-trips ------------------------------------------------- */

line("1. Tokenizer round-trips (step-5 served artifact)");
const t = step5.tokenizer;
console.log("specialTokenIds:", JSON.stringify(t.specialTokenIds));
const roundTrips = [
  "Hello, world.",
  "The cat sat on the mat. The cat sat.",
  "the to",
  "Hello,world.",
  "It isn't a nice day, is it?",
  "1234 + 5678 = 6912",
  "café naïve",
  "  leading and trailing  ",
  "New\nline and\ttab",
  "a",
  "",
];
let roundTripFails = 0;
for (const text of roundTrips) {
  const ids = t.encode(text);
  const back = t.decode(ids);
  const ok = back === text;
  if (!ok) roundTripFails++;
  console.log(
    `${ok ? "PASS" : "FAIL"}  ${JSON.stringify(text)} -> ${ids.length} ids -> ${JSON.stringify(back)}`,
  );
}
console.log(`round-trip failures: ${roundTripFails}/${roundTrips.length}`);

const detailed = t.encodeDetailed("Hello, world. The quick brown fox isn't here.");
console.log("pieces:", JSON.stringify(detailed.tokens), "unknown:", detailed.unknown);
const bosEos = t.encodeDetailed("Hello", { addBos: true, addEos: true });
console.log("with BOS/EOS:", JSON.stringify(bosEos.ids), JSON.stringify(bosEos.tokens));
console.log(
  "decode specials:",
  JSON.stringify(t.decode([t.specialTokenIds.bos, t.specialTokenIds.eos, t.specialTokenIds.pad])),
);

/* 2/3. vocabulary id mapping ---------------------------------------------- */

line("2/3. Vocabulary id mapping + specials");
console.log("tokenizer vocabSize:", t.vocabSize, "model vocabSize:", step5.model.config.vocabSize);
console.log("special ids:", JSON.stringify(t.specialTokenIds), "padId:", t.padId, "eosId:", t.eosId);
let idFails = 0;
for (let id = 0; id < Math.min(t.vocabSize, 64); id++) {
  const piece = t.tokenForId(id);
  if (typeof piece !== "string" || piece.length === 0) idFails++;
}
console.log(`single-id decode failures in first 64 ids: ${idFails}`);

/* 5. causal masking (runtime) --------------------------------------------- */

line("5. Causal masking: prefix invariance");
{
  const ids = t.encode("The quick brown fox jumps");
  setGradEnabled(false);
  try {
    const short = step5.model.forward(Int32Array.from(ids.slice(0, 3)), 1, 3, { training: false });
    const long = step5.model.forward(Int32Array.from(ids), 1, ids.length, { training: false });
    const V = step5.model.config.vocabSize;
    let maxDiff = 0;
    for (let pos = 0; pos < 3; pos++) {
      for (let i = 0; i < V; i++) {
        const d = Math.abs(short.logits.data[pos * V + i] - long.logits.data[pos * V + i]);
        if (d > maxDiff) maxDiff = d;
      }
    }
    console.log(`max |logit difference| on shared prefixes: ${maxDiff}`);
  } finally {
    setGradEnabled(true);
  }
}

/* 7. weight tying ---------------------------------------------------------- */

line("7. Weight tying");
{
  const map = step5.model.parameterMap();
  const names = [...map.keys()];
  console.log("tieEmbeddings:", step5.model.config.tieEmbeddings);
  console.log("parameter names:", names.join(", "));
  const emb = map.get("token_embedding");
  if (emb) {
    const x = t.encode(" zebra")[0];
    const promptIds = t.encode("Hello, world.");
    const dModel = step5.model.config.dModel;
    setGradEnabled(false);
    try {
      const logitFor = (): number => {
        const out = step5.model.forward(Int32Array.from(promptIds), 1, promptIds.length, {
          training: false,
        });
        return out.logits.data[(promptIds.length - 1) * step5.model.config.vocabSize + x];
      };
      const before = logitFor();
      const row = x * dModel;
      const original = emb.data.slice(row, row + dModel);
      for (let i = 0; i < dModel; i++) emb.data[row + i] *= 2;
      const after = logitFor();
      for (let i = 0; i < dModel; i++) emb.data[row + i] = original[i];
      const restored = logitFor();
      console.log(
        `scaling embedding row of token ${x} (${JSON.stringify(t.tokenForId(x))}) changed its output logit by ${Math.abs(after - before)} (restored: ${Math.abs(restored - before)})`,
      );
    } finally {
      setGradEnabled(true);
    }
  } else {
    console.log("no token_embedding parameter found");
  }
}

/* 8. next-token distribution + generation collapse ------------------------- */

line("8. Next-token distribution on the single token \"The\" (step-5)");
{
  const logits = step5.engine.scoreNextTokens(t.encode("The"));
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  let sum = 0;
  const probs = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    probs[i] = Math.exp(logits[i] - max);
    sum += probs[i];
  }
  for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  const top = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 8);
  console.log(
    "top-8:",
    top.map((id) => `${JSON.stringify(t.tokenForId(id))}(${(probs[id] * 100).toFixed(1)}%)`).join(" "),
  );
  let entropy = 0;
  for (const p of probs) if (p > 1e-12) entropy -= p * Math.log(p);
  console.log(`entropy: ${entropy.toFixed(3)} nats (uniform = ${Math.log(t.vocabSize).toFixed(3)} nats)`);
}

function generationStats(result: GenerationResult) {
  const ids = result.tokenIds;
  const unique = new Set(ids).size;
  let run = 1;
  let maxRun = 1;
  for (let i = 1; i < ids.length; i++) {
    run = ids[i] === ids[i - 1] ? run + 1 : 1;
    if (run > maxRun) maxRun = run;
  }
  const bigrams = new Set<string>();
  for (let i = 1; i < ids.length; i++) bigrams.add(`${ids[i - 1]}_${ids[i]}`);
  const repeatedBigramRatio = ids.length > 1 ? 1 - bigrams.size / (ids.length - 1) : 0;
  return {
    tokens: ids.length,
    uniqueRatio: Number((unique / ids.length).toFixed(3)),
    maxSameTokenRun: maxRun,
    repeatedBigramRatio: Number(repeatedBigramRatio.toFixed(3)),
    meanNll: Number(result.meanNll.toFixed(4)),
    stopReason: result.stopReason,
  };
}

function runGeneration(
  engine: AlphaInferenceEngine,
  label: string,
  prompt: string,
  sampling: Parameters<AlphaInferenceEngine["generate"]>[1],
) {
  const g = engine.generate(prompt, sampling);
  console.log(`${label}: ${JSON.stringify(g.text.slice(0, 90))} ${JSON.stringify(generationStats(g))}`);
}

line("8. Generation samples (served step-5 model)");
runGeneration(step5.engine, "greedy/The", "The", SAMPLING_PRESETS.greedy);
runGeneration(step5.engine, "greedy/Hello", "Hello,", SAMPLING_PRESETS.greedy);
runGeneration(step5.engine, "default/The", "The", { maxNewTokens: 40 });
runGeneration(step5.engine, "default/Hello", "Hello,", { maxNewTokens: 40, seed: 7 });

line("8. Generation sample (step-7 model, for comparison)");
runGeneration(step7.engine, "greedy/The", "The", SAMPLING_PRESETS.greedy);

/* 10/11. NLL vs n-gram baselines + data repetition ------------------------- */

line("11. Teacher-forced NLL: model vs uniform/unigram/bigram");
const corpusDocs = buildAuthoredCorpus(480, 20260101);
const V = t.vocabSize;
const uni = new Map<number, number>();
const bi = new Map<number, number>();
const prev = new Map<number, number>();
let corpusTokens = 0;
for (const doc of corpusDocs) {
  const ids = t.encode(doc.text);
  for (let i = 0; i < ids.length; i++) {
    uni.set(ids[i], (uni.get(ids[i]) ?? 0) + 1);
    corpusTokens++;
    if (i > 0) {
      const key = ids[i - 1] * V + ids[i];
      bi.set(key, (bi.get(key) ?? 0) + 1);
      prev.set(ids[i - 1], (prev.get(ids[i - 1]) ?? 0) + 1);
    }
  }
}
console.log(`counting corpus: ${corpusDocs.length} docs, ${corpusTokens} tokens`);
console.log(`uniform baseline: ${Math.log(V).toFixed(4)} nats per token (vocab ${V})`);

function teacherForcedNll(engine: AlphaInferenceEngine, text: string): { tokens: number; nll: number } | null {
  const ids = engine.tokenizer.encode(text).slice(0, engine.maxContextTokens);
  if (ids.length < 2) return null;
  setGradEnabled(false);
  try {
    const out = engine.model.forward(Int32Array.from(ids), 1, ids.length, { training: false });
    const vocab = engine.model.config.vocabSize;
    let nll = 0;
    for (let pos = 1; pos < ids.length; pos++) {
      const base = (pos - 1) * vocab;
      let max = -Infinity;
      for (let i = 0; i < vocab; i++) {
        const v = out.logits.data[base + i];
        if (v > max) max = v;
      }
      let sum = 0;
      for (let i = 0; i < vocab; i++) sum += Math.exp(out.logits.data[base + i] - max);
      nll += -(out.logits.data[base + ids[pos]] - max - Math.log(sum));
    }
    return { tokens: ids.length - 1, nll: nll / (ids.length - 1) };
  } finally {
    setGradEnabled(true);
  }
}

const evalDocs = buildAuthoredCorpus(20, 777777).slice(0, 3);
for (const doc of evalDocs) {
  const ids = t.encode(doc.text).slice(0, step5.engine.maxContextTokens);
  const measured = teacherForcedNll(step5.engine, doc.text);
  let uniNll = 0;
  let uniCount = 0;
  let biNll = 0;
  let biCount = 0;
  for (let i = 0; i < ids.length; i++) {
    if (i > 0) {
      const p = ((uni.get(ids[i]) ?? 0) + 1) / (corpusTokens + V);
      uniNll += -Math.log(p);
      uniCount++;
      const key = ids[i - 1] * V + ids[i];
      const pb = ((bi.get(key) ?? 0) + 1) / ((prev.get(ids[i - 1]) ?? 0) + V);
      biNll += -Math.log(pb);
      biCount++;
    }
  }
  console.log(
    `doc(${ids.length} tok): model ${measured?.nll.toFixed(4)} · unigram ${(uniNll / uniCount).toFixed(4)} · bigram ${(biNll / biCount).toFixed(4)}`,
  );
}

line("10. Training-data repetition (authored corpus, first 80 docs)");
{
  const sample = corpusDocs.slice(0, 80);
  const sentences = sample.flatMap((d) => d.text.split(/(?<=[.!?])\s+/));
  const counts = new Map<string, number>();
  for (const s of sentences) counts.set(s, (counts.get(s) ?? 0) + 1);
  const dupes = [...counts.entries()].filter(([, c]) => c > 1).sort((a, b) => b[1] - a[1]).slice(0, 3);
  console.log(
    `sentences: ${sentences.length}, repeated sentences: ${[...counts.values()].filter((c) => c > 1).length}`,
  );
  for (const [s, c] of dupes) console.log(`  x${c}: ${JSON.stringify(s.slice(0, 70))}`);

  const grams = new Map<string, number>();
  let consecutiveDupes = 0;
  let total = 0;
  for (const doc of sample.slice(0, 40)) {
    const ids = t.encode(doc.text);
    for (let i = 0; i + 4 <= ids.length; i++) {
      const key = ids.slice(i, i + 4).join(",");
      grams.set(key, (grams.get(key) ?? 0) + 1);
    }
    for (let i = 1; i < ids.length; i++) {
      if (ids[i] === ids[i - 1]) consecutiveDupes++;
      total++;
    }
  }
  const top = [...grams.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
  console.log(`consecutive duplicate-token rate in corpus: ${(consecutiveDupes / total).toFixed(4)}`);
  for (const [key, c] of top) {
    const ids = key.split(",").map(Number);
    console.log(`  4-gram x${c}: ${JSON.stringify(t.decode(ids))}`);
  }
  const repeatedGramKinds = [...grams.values()].filter((c) => c >= 5).length;
  console.log(`4-grams occurring >= 5 times: ${repeatedGramKinds} of ${grams.size}`);
}

line("11. Training budget (both models)");
for (const [label, m] of [
  ["step-5 (served)", step5],
  ["step-7", step7],
] as const) {
  const a = m.artifact;
  console.log(
    `${label}: ${a.model.parameterCount} params · ${a.training.steps} steps · ${a.training.tokensSeen} tokens · ${(a.training.tokensSeen / a.model.parameterCount).toFixed(5)} tokens/param · recorded loss ${a.evaluation.loss}`,
  );
}
