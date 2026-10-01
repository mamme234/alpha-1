import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
const d=buildAuthoredCorpus(400,20260101);
for(const c of ["general-prose","dialogue"]){
  for(const s of d.filter(x=>x.category===c).slice(0,2)) console.log(`\n--${c}--\n${s.text}`);
}
