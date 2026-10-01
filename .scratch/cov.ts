import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { words } from "../src/alpha/datasets/diversity";
function sh(t:string,n=8){const w=words(t);const s=new Set<string>();if(w.length<n){if(w.length>0)s.add(w.join(" "));return s;}for(let i=0;i+n<=w.length;i++)s.add(w.slice(i,i+n).join(" "));return s;}
const docs = buildAuthoredCorpus(300,20260101);
let over=0, tot=0, maxc=0;
for(let i=0;i<docs.length;i++)for(let j=i+1;j<docs.length;j++){
  const a=sh(docs[i].text), b=sh(docs[j].text);
  let c=0; for(const s of a) if(b.has(s)) c++;
  const cov=c/a.size; tot++; if(cov>=0.5){over++; if(cov>maxc)maxc=cov;}
}
console.log(`pairs ${tot}, >=50% coverage: ${over} (${(over/tot*100).toFixed(1)}%), max ${maxc.toFixed(2)}`);
// Step4 for comparison
import { buildGeneratedCorpus } from "../src/alpha/datasets/generated-corpus";
const s4 = buildGeneratedCorpus(300,20250930).documents;
let o4=0,t4=0,m4=0;
for(let i=0;i<s4.length;i++)for(let j=i+1;j<s4.length;j++){
  const a=sh(s4[i]),b=sh(s4[j]); let c=0; for(const s of a) if(b.has(s)) c++;
  const cov=c/a.size; t4++; if(cov>=0.5){o4++; if(cov>m4)m4=cov;}
}
console.log(`STEP4 pairs ${t4}, >=50% coverage: ${o4} (${(o4/t4*100).toFixed(1)}%), max ${m4.toFixed(2)}`);
