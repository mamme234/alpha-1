import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
const d = buildAuthoredCorpus(300, 20260101);
for (const cat of ["educational","explanations","dialogue","instructions","factual-reference"]) {
  const s = d.find(x=>x.category===cat)!;
  console.log(`\n=== ${cat} ===\n${s.text}`);
}
