import { buildGeneratedCorpus } from "../src/alpha/datasets/generated-corpus";
import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { analyseDiversity } from "../src/alpha/datasets/diversity";

const s4 = buildGeneratedCorpus(300, 20250930).documents;
const s5 = buildAuthoredCorpus(300, 20260101).map(d => d.text);
const a = analyseDiversity(s4);
const b = analyseDiversity(s5);
const pick = (r:any)=>({sent:r.sentences, vocab:r.vocabulary, char:r.characterVariety, boiler:r.boilerplate, len:r.length, lowInfo:r.lowInformation});
console.log("STEP4", JSON.stringify(pick(a), null, 1));
console.log("STEP5", JSON.stringify(pick(b), null, 1));
