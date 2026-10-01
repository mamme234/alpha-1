import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { createProvenanceDocument } from "../src/alpha/datasets/provenance";
import { splitDocuments } from "../src/alpha/datasets/splits";
import { createEvalSuite } from "../src/alpha/evaluation/suite";
import { detectOverlap } from "../src/alpha/datasets/splits";

const raw = buildAuthoredCorpus(400, 20260101);
const docs = raw.map((d,i)=>createProvenanceDocument({documentId:`d${i}`,text:d.text,sourceId:"s",origin:"authored",license:"Alpha-owned",language:d.language,category:d.category,acquisition:{method:"generated-in-repo",location:"this-repository",acquiredAt:1,collectedBy:"a"},createdAt:1}));
const split = splitDocuments(docs,{seed:1337});
const train = split.train.map(d=>d.text);
const suite = createEvalSuite({heldOutDocuments: split.test.map(d=>d.text), now:1});
const rows:{id:string;cov:number}[]=[];
for(const c of suite.cases){
  if(c.category==="language-modeling") continue;
  const r = detectOverlap([{label:c.id,text:c.prompt}], train, {contaminationThreshold:0});
  const r2 = c.continuation? detectOverlap([{label:c.id,text:c.continuation}], train, {contaminationThreshold:0}) : null;
  rows.push({id:c.id, cov: Math.max(r.maxShingleCoverage, r2?r2.maxShingleCoverage:0)});
}
rows.sort((a,b)=>b.cov-a.cov);
for(const r of rows.slice(0,8)) console.log(`${r.id}: ${(r.cov*100).toFixed(1)}%`);
