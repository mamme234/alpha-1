import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { words } from "../src/alpha/datasets/diversity";
const docs = buildAuthoredCorpus(400,20260101);
const bodyKey=(t:string)=>words(t).join(" ").replace(/^[^:]*:\s*/,"").slice(0,14);
for(const cat of ["general-prose","educational","factual-reference","dialogue","instructions","explanations","structured","multilingual"]){
  const m=new Map<string,number>();
  for(const d of docs.filter(x=>x.category===cat)){const k=bodyKey(d.text);m.set(k,(m.get(k)??0)+1);}
  let dup=0; for(const c of m.values()) if(c>1) dup+=c-1;
  console.log(cat, "near-dupes:", dup, "/", docs.filter(x=>x.category===cat).length);
}
