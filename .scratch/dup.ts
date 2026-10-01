import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { documentFingerprint } from "../src/alpha/datasets/provenance";
import { words } from "../src/alpha/datasets/diversity";
function norm(t:string){return t.replace(/^[^.]*[:.]\s*/,"").trim();}
function key(t:string){return words(t).slice(0,12).join(" ");}
const docs = buildAuthoredCorpus(400,20260101);
const m=new Map<string,number>();
for(const d of docs){const k=key(d.text); m.set(k,(m.get(k)??0)+1);}
let dup=0; for(const c of m.values()) if(c>1) dup+=c;
console.log("docs with same 12-word opening (approx dupes):", dup, "of", docs.length);
// exact body dup ignoring the opener
const b=new Map<string,number>();
for(const d of docs){const k=key(norm(d.text)); b.set(k,(b.get(k)??0)+1);}
let d2=0; for(const c of b.values()) if(c>1) d2+=c;
console.log("docs with same body after the frame:", d2);
const ex = docs.filter(d=>d.category==="multilingual");
const byLang=new Map<string,number>();
for(const d of ex){const k=key(norm(d.text)); byLang.set(k,(byLang.get(k)??0)+1);}
console.log("multilingual same-body groups:", [...byLang.values()].filter(v=>v>1));
