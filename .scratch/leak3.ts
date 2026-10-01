import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { createProvenanceDocument } from "../src/alpha/datasets/provenance";
import { splitDocuments, auditSplitOverlap, nearDuplicateGroupKey } from "../src/alpha/datasets/splits";
const docs = buildAuthoredCorpus(100,23).map((d,i)=>createProvenanceDocument({documentId:`doc_${String(i).padStart(4,"0")}`,text:d.text,sourceId:"s",origin:"authored",license:"Alpha-owned",language:d.language,category:d.category,acquisition:{method:"generated-in-repo",location:"this-repository",acquiredAt:1,collectedBy:"a"},createdAt:1}));
const a = splitDocuments(docs,{seed:7});
const audit = auditSplitOverlap(a);
console.log("val:", audit.validation.shingleMatches, audit.validation.maxShingleCoverage);
console.log("test:", audit.test.shingleMatches, audit.test.maxShingleCoverage);
const byId=new Map(docs.map(d=>[d.documentId,d]));
for(const id of [...audit.validation.shingleMatches,...audit.test.shingleMatches].slice(0,4)){
  const d=byId.get(id)!;
  console.log(`\n--- ${id} (${d.category}) key=${nearDuplicateGroupKey(d.text).slice(0,60)}`);
  console.log(d.text.slice(0,180));
}
