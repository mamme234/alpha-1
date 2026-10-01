import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { words } from "../src/alpha/datasets/diversity";

const docs = buildAuthoredCorpus(100, 23).map((d,i)=>({id:`doc_${String(i).padStart(4,"0")}`, cat:d.category, text:d.text}));
function shingles(t:string,n=8){const w=words(t);const s=new Set<string>();if(w.length<n){if(w.length>0)s.add(w.join(" "));return s;}for(let i=0;i+n<=w.length;i++)s.add(w.slice(i,i+n).join(" "));return s;}
const byCat=new Map<string,any[]>();
for(const d of docs){ if(!byCat.has(d.cat))byCat.set(d.cat,[]); byCat.get(d.cat)!.push(d); }
// within-category pairwise coverage
for(const [cat,ds] of byCat){
  let tot=0,n=0;
  for(let i=0;i<ds.length;i++)for(let j=i+1;j<ds.length;j++){
    const a=shingles(ds[i].text), b=shingles(ds[j].text);
    let sh=0; for(const x of a) if(b.has(x)) sh++;
    tot += sh/a.size; n++;
  }
  console.log(cat, "docs",ds.length, "mean pairwise shingle coverage", (tot/n).toFixed(3));
}
console.log("\nsample edu doc:\n", byCat.get("educational")![0].text.slice(0,400));
console.log("\nsample edu doc2:\n", byCat.get("educational")![1].text.slice(0,400));
