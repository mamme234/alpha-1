import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import { ALPHA_MODEL_PRESETS } from "../src/alpha/model/config";
import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { buildGeneratedCorpus } from "../src/alpha/datasets/generated-corpus";
import { measureTokenizer, decideTokenizerChange, summariseTokenizerDecision } from "../src/alpha/tokenizer/analysis";

const authored = buildAuthoredCorpus(480, 20260101).map((d) => d.text);
const generated = buildGeneratedCorpus(300, 20250930).documents;
const union = [...authored, ...generated];

const step4Tok = AlphaTokenizer.train(generated, { vocabSize: ALPHA_MODEL_PRESETS.micro.vocabSize, version: "1.0.0", trainedOn: "step4" });
const sharedTok = AlphaTokenizer.train(union, { vocabSize: ALPHA_MODEL_PRESETS.micro.vocabSize, version: "1.0.0", trainedOn: "union" });

const current = measureTokenizer(step4Tok, union);
const candidate = measureTokenizer(sharedTok, union);
const decision = decideTokenizerChange(current, candidate);
console.log(summariseTokenizerDecision(decision));
console.log("current unknown share:", current.unknownTokenShare.toFixed(5), "coverage:", current.vocabularyCoverage.toFixed(4), "unknown chars:", current.unknownCharacters.join(""));
console.log("candidate unknown share:", candidate.unknownTokenShare.toFixed(5), "coverage:", candidate.vocabularyCoverage.toFixed(4));
console.log("adopt:", decision.retrain ? "shared" : "step4", "| step4 fp:", step4Tok.fingerprint(), "| shared fp:", sharedTok.fingerprint(), "| vocab", step4Tok.vocabSize, sharedTok.vocabSize);
