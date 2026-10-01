import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
const d=buildAuthoredCorpus(300,20260101);
for(const c of ["general-prose","structured","multilingual"]){const s=d.find(x=>x.category===c)!;console.log(`\n--${c}--\n${s.text}`);}
