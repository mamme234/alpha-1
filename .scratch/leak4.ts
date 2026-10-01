import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { createProvenanceDocument } from "../src/alpha/datasets/provenance";
import { splitDocuments, auditSplitOverlap } from "../src/alpha/datasets/splits";
const mk=(n:number,seed:number)=>buildAuthoredCorpus(n,seed).map((d,i)=>createProvenanceDocument({documentId:`doc_${String(i).padStart(4,"0")}`,text:d.text,sourceId:"s",origin:"authored",license:"Alpha-owned",language:d.language,category:d.category,acquisition:{method:"generated-in-repo",location:"this-repository",acquiredAt:1,collectedBy:"a"},createdAt:1}));
for(const [n,seed] of [[100,23],[300,7],[240,20260101],[400,99]] as [number,number][]){
  const a=splitDocuments(mk(n,seed),{seed:7});
  const au=auditSplitOverlap(a);
  console.log(`n=${n} seed=${seed}: val leaks ${au.validation.shingleMatches.length}, test leaks ${au.test.shingleMatches.length}`);
}
for(const [n,seed] of [[100,23],[300,7],[240,20260101],[400,99]] as [number,number][]){
  const a=splitDocuments(mk(n,seed),{seed:7});
  console.log(`n=${n}: train ${a.train.length}, val ${a.validation.length}, test ${a.test.length}`);
}
