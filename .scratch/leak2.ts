import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { words } from "../src/alpha/datasets/diversity";
const docs = buildAuthoredCorpus(100, 23).map((d,i)=>({id:`doc_${String(i).padStart(4,"0")}`,cat:d.category,text:d.text}));
function sh(t:string,n=8){const w=words(t);const s=new Set<string>();if(w.length<n){if(w.length>0)s.add(w.join(" "));return s;}for(let i=0;i+n<=w.length;i++)s.add(w.slice(i,i+n).join(" "));return s;}
const train=docs.filter(d=>!["doc_0084","doc_0092","doc_0055","doc_0079"].includes(d.id));
const ref=new Set<string>(); for(const d of train) for(const s of sh(d.text)) ref.add(s);
for(const id of ["doc_0084","doc_0092","doc_0055","doc_0079"]){
  const d=docs.find(x=>x.id===id)!; const a=sh(d.text); let c=0; for(const s of a) if(ref.has(s)) c++;
  console.log(`\n${id} (${d.cat}) coverage ${(c/a.size).toFixed(3)}`);
  console.log(d.text.slice(0,220));
}
