import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import { AlphaTransformer } from "../src/alpha/model/transformer";
import { ALPHA_MODEL_PRESETS, createModelConfig } from "../src/alpha/model/config";
import { AlphaTrainer } from "../src/alpha/training/trainer";
import { createDataset } from "../src/alpha/datasets/types";
import { buildAuthoredCorpus } from "../src/alpha/datasets/authored-corpus";
import { buildGeneratedCorpus } from "../src/alpha/datasets/generated-corpus";
import { createEvalSuite, auditSuiteLeakage } from "../src/alpha/evaluation/suite";
import { runEvalSuite } from "../src/alpha/evaluation/runner";

const t = (label: string, start: number) =>
  console.log(`${label}: ${((Date.now() - start) / 1000).toFixed(1)}s`);

let s = Date.now();
const authored = buildAuthoredCorpus(480, 20260101);
const gen = buildGeneratedCorpus(300, 20250930);
t("corpora", s);

s = Date.now();
const union = [...authored.map((d) => d.text), ...gen.documents];
const tokenizer = AlphaTokenizer.train(union, {
  vocabSize: ALPHA_MODEL_PRESETS.micro.vocabSize,
  version: "probe",
  trainedOn: "probe",
});
t(`tokenizer (${tokenizer.vocabSize} tokens)`, s);

s = Date.now();
const evalSource = buildAuthoredCorpus(80, 777001).map((d) => d.text);
const heldOut = evalSource.slice(0, 41);
const suite = createEvalSuite({ heldOutDocuments: heldOut });
t(`suite (${suite.cases.length} cases)`, s);

s = Date.now();
const report = auditSuiteLeakage(suite, union);
t(`auditSuiteLeakage (clean=${report.clean}, max=${report.maxCoverage.toFixed(3)})`, s);

s = Date.now();
const config = createModelConfig({ preset: "micro", vocabSize: Math.max(768, tokenizer.vocabSize) });
const model = new AlphaTransformer(config);
const dataset = createDataset({
  name: "probe",
  version: "1",
  description: "p",
  license: "Alpha-owned",
  source: "probe",
  documents: gen.documents,
});
const trainer = new AlphaTrainer({
  model,
  tokenizer,
  dataset,
  config: {
    batchSize: 8,
    seqLen: 48,
    totalSteps: 30,
    gradientAccumulationSteps: 2,
    learningRate: 0.002,
    warmupSteps: 4,
    evalInterval: 10,
    evalBatches: 4,
    checkpointInterval: 30,
    seed: 1337,
    earlyStopping: { monitor: "validationLoss", patience: 3, minDelta: 0.01, minSteps: 15, evalEvery: 10 },
  },
});
const summary = trainer.trainToCompletion();
t(`training 30 steps (loss ${summary.firstLoss?.toFixed(3)} -> ${summary.lastLoss?.toFixed(3)}, val ${summary.validationLoss?.toFixed(3)}, ${summary.throughputTokensPerSecond} tok/s)`, s);

s = Date.now();
const cap = runEvalSuite(model, tokenizer, suite, { heldOutDocuments: heldOut });
t(`runEvalSuite (lm loss ${cap.languageModeling.loss?.toFixed(3)})`, s);
