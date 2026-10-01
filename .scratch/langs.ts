import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
const d = buildAuthoredCorpus(80, 20260101);
const c: Record<string,number>={};
for(const x of d) c[x.language]=(c[x.language]??0)+1;
console.log(c);
