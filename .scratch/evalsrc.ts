import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { buildGeneratedCorpus } from "../src/alpha/datasets/generated-corpus";
import { detectOverlap } from "../src/alpha/datasets/splits";

const s5train = buildAuthoredCorpus(400, 20260101).slice(0,352).map(d=>d.text);
const s4all = buildGeneratedCorpus(300, 20250930).documents;
const s4train = s4all.slice(0,258);
const ref = [...s5train, ...s4train];
for (const seed of [777001, 424242, 999983]) {
  const cands = buildAuthoredCorpus(80, seed);
  const kept = cands.filter(d => detectOverlap([{label:"x",text:d.text}], ref, {contaminationThreshold:0.4}).clean);
  const maxCov = Math.max(...cands.map(d => detectOverlap([{label:"x",text:d.text}], ref, {contaminationThreshold:0}).maxShingleCoverage));
  console.log(`seed ${seed}: kept ${kept.length}/80, max coverage ${(maxCov*100).toFixed(1)}%`);
}
