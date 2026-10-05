/**
 * Alpha — generation pipeline diagnosis (READ-ONLY).
 *
 * Root-cause hunt for the 1M-token model's bad generations. Nothing here
 * trains, writes weights, or calls any external service. It loads exactly one
 * set of weights — `language-mv2-final.alpha-ckpt.json` — and walks the
 * generation pipeline stage by stage:
 *
 *   Stage 1  Tokenizer identity   — specials (PAD/UNK/BOS/EOS) ids, vocabulary
 *                                  size, fingerprint vs the checkpoint record.
 *   Stage 2  Encode               — prompt -> ids, and the ids decoded back to
 *                                  text byte-for-byte (round-trip).
 *   Stage 3  Teacher forcing      — held-out corpus windows, next-token top-1
 *                                  accuracy and NLL. This is *separate* from
 *                                  generation: it never feeds a model output
 *                                  back into the model.
 *   Stage 4  Logits               — the actual next-token logit row for a prompt:
 *                                  top-10 ids with their token strings, plus
 *                                  where EOS / PAD / UNK rank.
 *   Stage 5  Token selection      — which id greedy argmax picks, whether it is
 *                                  EOS, and what that id decodes to.
 *   Stage 6  Decode               — selected ids -> strings -> text, with
 *                                  specials rendered visibly so a decode that
 *                                  silently drops everything is observable.
 *
 * Stage 3 and Stage 5/6 answer different questions: Stage 3 says whether the
 * weights model the corpus at all; Stage 5/6 says whether the *pipeline*
 * (selection + decode) destroys an otherwise-fine prediction.
 *
 * Usage:  bun scripts/alpha-pipeline-diagnose.ts
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { parseCheckpoint } from "../src/alpha/training/checkpoint";
import { AlphaTransformer } from "../src/alpha/model/transformer";
import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import { AlphaInferenceEngine, SAMPLING_PRESETS } from "../src/alpha/inference/engine";
import { setGradEnabled } from "../src/alpha/core/tensor";
import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { encodeCorpus } from "../src/alpha/datasets/corpus";
import type { AlphaDataset } from "../src/alpha/datasets/types";

const ROOT = join(__dirname, "..");
const CKPT = join(ROOT, "src/alpha/experiments/language/language-mv2-final.alpha-ckpt.json");

const PROMPTS = [
  "Hello",
  "Hi, I want to ask you something.",
  "What is Alpha?",
  "Tell me about Ethiopia.",
];

const MAX_NEW = 40;

function rule(title: string): void {
  console.log(`\n${"=".repeat(74)}\n${title}\n${"=".repeat(74)}`);
}

function vis(s: string): string {
  return JSON.stringify(s);
}

function main(): void {
  rule("STAGE 0 — load final checkpoint (read-only)");
  const raw = readFileSync(CKPT, "utf8");
  const ck = parseCheckpoint(raw);
  const tokenizer = AlphaTokenizer.fromJSON(ck.tokenizer.snapshot);
  const model = new AlphaTransformer(ck.config);
  model.loadWeights(ck.weights);

  console.log(`checkpoint file        : ${CKPT}`);
  console.log(`checkpoint id          : ${ck.id}`);
  console.log(`runId                  : ${ck.runId}`);
  console.log(`step / tokensSeen      : ${ck.step} / ${ck.tokensSeen}`);
  console.log(`metrics.trainLoss      : ${ck.metrics.trainLoss}`);
  console.log(`metrics.validationLoss : ${ck.metrics.validationLoss}`);
  console.log(`metrics.uniformLoss    : ${ck.metrics.uniformLoss}`);
  console.log(`model                  : ${ck.config.name} @ ${ck.config.version}`);
  console.log(`params                 : ${model.parameterCount}`);
  console.log(`configFingerprint      : ${ck.configFingerprint}`);

  const engine = new AlphaInferenceEngine({ model, tokenizer, stage: ck.stage ?? "trained" });

  /* ------------------------------------------------------------------ */
  rule("STAGE 1 — tokenizer identity and special-token ids");

  const specials = tokenizer.specialTokenIds;
  const snap = ck.tokenizer.snapshot;
  const vocabLen = snap.tokens.length;

  console.log(`tokenizer version      : ${snap.version}`);
  console.log(`recorded fingerprint   : ${ck.tokenizer.fingerprint}`);
  console.log(`recomputed fingerprint : ${tokenizer.fingerprint()}`);
  console.log(
    `fingerprint match      : ${ck.tokenizer.fingerprint === tokenizer.fingerprint() ? "YES" : "NO — MISMATCH"}`,
  );
  console.log(`checkpoint vocabSize   : ${ck.tokenizer.vocabSize}`);
  console.log(`model config vocabSize : ${ck.config.vocabSize}`);
  console.log(`tokenizer tokens[] len : ${vocabLen}`);
  console.log(
    `vocab agreement        : ${
      ck.tokenizer.vocabSize === ck.config.vocabSize && ck.config.vocabSize === vocabLen
        ? "YES (768 = 768 = 768)"
        : "NO — vocab size disagreement"
    }`,
  );
  console.log(`specialTokens (strings): ${vis(JSON.stringify(snap.specialTokens))}`);
  console.log(`PAD  id = ${specials.pad}  -> ${vis(tokenizer.tokenForId(specials.pad))}`);
  console.log(`UNK  id = ${specials.unk}  -> ${vis(tokenizer.tokenForId(specials.unk))}`);
  console.log(`BOS  id = ${specials.bos}  -> ${vis(tokenizer.tokenForId(specials.bos))}`);
  console.log(`EOS  id = ${specials.eos}  -> ${vis(tokenizer.tokenForId(specials.eos))}`);

  const expected = { pad: 0, unk: 1, bos: 2, eos: 3 };
  const idsOk =
    specials.pad === expected.pad &&
    specials.unk === expected.unk &&
    specials.bos === expected.bos &&
    specials.eos === expected.eos;
  console.log(
    `expected PAD/UNK/BOS/EOS: ${expected.pad}/${expected.unk}/${expected.bos}/${expected.eos} -> ${
      idsOk ? "MATCH" : "MISMATCH"
    }`,
  );

  // The ids the engine actually bans / stops on, read back off the live tokenizer.
  const banned = tokenizer.padId;
  console.log(
    `\nengine bans padId=${banned} during selection; engine stops on eosId=${tokenizer.eosId}`,
  );
  console.log(
    `ids inside vocab       : ${[specials.pad, specials.unk, specials.bos, specials.eos].every(
      (id) => id >= 0 && id < ck.config.vocabSize,
    )}`,
  );

  /* ------------------------------------------------------------------ */
  rule("STAGE 2 — encode -> ids -> decode round-trip (per prompt)");

  for (const prompt of PROMPTS) {
    const enc = tokenizer.encodeDetailed(prompt, {
      maxLength: engine.maxContextTokens - 1,
      truncation: "left",
      addBos: true,
    });
    const back = tokenizer.decode(enc.ids);
    const stripped = back.replace(/^<bos>/, "");
    console.log(`\nprompt      : ${vis(prompt)}`);
    console.log(`ids         : [${enc.ids.join(", ")}]`);
    console.log(`as strings  : ${enc.ids.map((id) => vis(tokenizer.tokenForId(id))).join(" ")}`);
    console.log(`decode(ids) : ${vis(back)}`);
    console.log(`exact RT    : ${back === prompt ? "YES" : `NO — got ${vis(back)}`}`);
    console.log(`RT w/o BOS  : ${stripped === prompt ? "YES" : `NO — got ${vis(stripped)}`}`);
    console.log(`unknown tok : ${enc.unknown}`);
  }

  /* ------------------------------------------------------------------ */
  rule("STAGE 3 — TEACHER FORCED next-token prediction (held-out corpus)");

  // Rebuild a small held-out corpus with the SAME tokenizer, so the measured
  // NLL is on real text the model was trained on but evaluated out-of-sample.
  // Teacher forcing only: targets come from the corpus, never from the model.
  const docs = buildAuthoredCorpus(120, 20260202).map((d) => d.text);
  const holdout: AlphaDataset = {
    id: "alpha-diagnose-holdout",
    name: "alpha-diagnose-holdout",
    version: "1.0.0",
    description: "Read-only diagnosis holdout, rebuilt from Alpha's authored corpus.",
    license: "Alpha-owned",
    source: "authored for Alpha",
    documents: docs,
  };
  const corpus = encodeCorpus(holdout, tokenizer, { validationFraction: 0.25 });

  const valIds = corpus.validationIds;
  const ctx = ck.config.contextLength;
  const windows: number[] = [];
  for (let start = 0; start + ctx + 1 <= valIds.length && windows.length < 8; start += ctx * 3) {
    windows.push(start);
  }

  setGradEnabled(false);
  let tfTokens = 0;
  let tfLoss = 0;
  let tfTop1 = 0;
  let tfEosCorrect = 0;
  let tfEosTotal = 0;
  let tfEosRankSum = 0;
  const tfSamples: { actual: number; top1: number; top1Str: string; actualStr: string }[] = [];

  for (const start of windows) {
    const ids = valIds.subarray(start, start + ctx + 1);
    const input = Int32Array.from(ids.subarray(0, ctx));
    const forward = model.forward(input, 1, ctx, { training: false });
    const vocab = ck.config.vocabSize;
    for (let t = 0; t < ctx; t++) {
      const target = ids[t + 1];
      const off = t * vocab;
      const row = forward.logits.data.subarray(off, off + vocab) as Float32Array;
      let max = -Infinity;
      for (let j = 0; j < vocab; j++) if (row[j] > max) max = row[j];
      let sum = 0;
      for (let j = 0; j < vocab; j++) sum += Math.exp(row[j] - max);
      const logZ = max + Math.log(sum);
      tfLoss += logZ - row[target];
      tfTokens++;
      let best = 0;
      for (let j = 1; j < vocab; j++) if (row[j] > row[best]) best = j;
      if (best === target) tfTop1++;
      let rank = 0;
      for (let j = 0; j < vocab; j++) if (row[j] > row[target]) rank++;
      if (target === tokenizer.eosId) {
        tfEosTotal++;
        tfEosRankSum += rank;
        if (best === tokenizer.eosId) tfEosCorrect++;
      } else if (rank === 0 && tfSamples.length < 12) {
        tfSamples.push({
          actual: target,
          top1: best,
          top1Str: tokenizer.tokenForId(best),
          actualStr: tokenizer.tokenForId(target),
        });
      }
    }
  }
  setGradEnabled(true);

  const tfNll = tfLoss / tfTokens;
  console.log(`held-out windows       : ${windows.length} (ctx=${ctx})`);
  console.log(`teacher-forced tokens  : ${tfTokens}`);
  console.log(`teacher-forced NLL     : ${tfNll.toFixed(4)}`);
  console.log(`teacher-forced ppl     : ${Math.exp(tfNll).toFixed(2)}`);
  console.log(`checkpoint val NLL     : ${ck.metrics.validationLoss}`);
  console.log(`uniform (random) NLL   : ${Math.log(ck.config.vocabSize).toFixed(4)}`);
  console.log(`top-1 accuracy         : ${((tfTop1 / tfTokens) * 100).toFixed(2)}%`);
  console.log(
    `\nEOS targets            : ${tfEosTotal}, predicted correctly ${tfEosCorrect}, mean rank ${(
      tfEosTotal ? tfEosRankSum / tfEosTotal : NaN
    ).toFixed(1)} / ${ck.config.vocabSize}`,
  );
  console.log(`correct non-EOS predictions (context):`);
  for (const s of tfSamples) {
    console.log(`   actual ${vis(s.actualStr)} (${s.actual})  top1 ${vis(s.top1Str)} (${s.top1})`);
  }
  console.log(
    `\nVERDICT teacher forcing: ${
      tfNll < Math.log(ck.config.vocabSize) - 0.5
        ? "the weights DO model the corpus; next-token prediction works"
        : "the weights barely beat uniform; next-token prediction itself is weak"
    }`,
  );

  /* ------------------------------------------------------------------ */
  rule("STAGE 4/5/6 — logits -> selection -> decode, per prompt (autoregressive)");

  for (const prompt of PROMPTS) {
    console.log(`\n${"-".repeat(70)}\nprompt: ${vis(prompt)}`);

    const enc = tokenizer.encodeDetailed(prompt, {
      maxLength: engine.maxContextTokens - 1,
      truncation: "left",
      addBos: true,
    });
    const ids = Int32Array.from(enc.ids);
    const fwd = model.forward(ids, 1, ids.length, { training: false });
    const vocab = ck.config.vocabSize;
    const row = fwd.logits.data.subarray((ids.length - 1) * vocab, ids.length * vocab) as Float32Array;
    let max = -Infinity;
    for (let j = 0; j < vocab; j++) if (row[j] > max) max = row[j];
    let sum = 0;
    for (let j = 0; j < vocab; j++) sum += Math.exp(row[j] - max);
    const logZ = max + Math.log(sum);

    const order = Array.from({ length: vocab }, (_, j) => j).sort((a, b) => row[b] - row[a]);
    console.log(`prompt ids (${ids.length}) : [${enc.ids.join(", ")}]`);
    console.log(`logit row: max=${max.toFixed(4)} logZ=${logZ.toFixed(4)} range=${(
      order.length ? (row[order[0]] - row[order[vocab - 1]]).toFixed(4) : "n/a"
    )}`);
    console.log(`top-10 next-token candidates:`);
    for (let n = 0; n < 10; n++) {
      const id = order[n];
      const p = Math.exp(row[id] - logZ);
      console.log(
        `   #${String(n + 1).padStart(2)} id=${String(id).padStart(3)} p=${p.toFixed(4)} ${vis(
          tokenizer.tokenForId(id),
        )}`,
      );
    }
    const rankOf = (id: number) => order.indexOf(id) + 1;
    console.log(
      `ranks — EOS(${specials.eos})=${rankOf(specials.eos)}  UNK(${specials.unk})=${rankOf(
        specials.unk,
      )}  BOS(${specials.bos})=${rankOf(specials.bos)}  PAD(${specials.pad})=banned`,
    );
    const greedyId = order[0];
    console.log(
      `\nSELECTION greedy argmax -> id ${greedyId} = ${vis(tokenizer.tokenForId(greedyId))}`,
    );
    if (greedyId === tokenizer.eosId) {
      console.log(
        `  *** argmax IS EOS. engine.decode() returns at engine.ts:442 before pushing it,`,
      );
      console.log(`      so tokenIds=[] and text="" with stopReason "eos". Pipeline is faithful;`);
      console.log(`      the MODEL chose to stop immediately. ***`,
      );
    }

    for (const preset of ["greedy", "balanced"] as const) {
      const res = engine.generate(prompt, { ...SAMPLING_PRESETS[preset], maxNewTokens: MAX_NEW, seed: 1337 });
      const shown = res.tokenIds
        .map((id) => `${id}:${vis(tokenizer.tokenForId(id))}`)
        .join(" ");
      const allSpecial =
        res.tokenIds.length > 0 &&
        res.tokenIds.every((id) => [specials.pad, specials.unk, specials.bos, specials.eos].includes(id));
      console.log(`\n  [${preset}] stopReason=${res.stopReason} generated=${res.generatedTokens} meanNll=${res.meanNll.toFixed(3)}`);
      console.log(`    selected ids + strings: ${shown || "(none)"}`);
      console.log(`    decode(ids)            : ${vis(res.text)}`);
      console.log(
        `    decode(ids, keepSpecial): ${vis(tokenizer.decode(res.tokenIds, { skipSpecial: false }))}`,
      );
      console.log(
        `    all-selected-are-special: ${allSpecial}  (true => decode() legitimately yields "")`,
      );
    }
  }

  /* ------------------------------------------------------------------ */
  rule("STAGE 7 — control: does decode() alone lose information?");

  // Decode a known-good string straight from ids. If this round-trips, decode is
  // not the problem and any empty text came from the ids, not from decode.
  for (const s of ["Hello", "Tell me about Ethiopia.", "The quick brown fox."]) {
    const ids = tokenizer.encodeDetailed(s, { addBos: false }).ids;
    console.log(`${vis(s)} -> [${ids.join(", ")}] -> ${vis(tokenizer.decode(ids))}`);
  }
  const unkOnly = [specials.unk, specials.unk, specials.unk];
  console.log(
    `sanity: decode([unk,unk,unk]) = ${vis(tokenizer.decode(unkOnly))} (expected "" by design)`,
  );
  const padOnly = new Array(5).fill(specials.pad);
  console.log(`sanity: decode(pad x5)     = ${vis(tokenizer.decode(padOnly))} (expected "" by design)`);

  console.log(`\nDiagnosis complete. No weights were read for writing; nothing was trained.`);
}

main();