import { microV2Config } from "../model/micro-v2-config";
import { buildStep7Corpus, STEP7_CORPUS_VERSION, STEP7_SOURCE_ID } from "../datasets/step7-corpus";
import { validateDataset } from "../datasets/types";
import { AlphaTokenizer } from "../tokenizer/bpe";
import { AlphaTransformer } from "../model/transformer";
import { AlphaTrainer, createTrainingConfig } from "../training/trainer";
import { performance } from "node:perf_hooks";

function rssMb() {
  if (process.memoryUsage?.rss) return (process.memoryUsage().rss / 1024 / 1024).toFixed(1);
  return "n/a";
}

const t0 = performance.now();
const c = microV2Config;
const built = buildStep7Corpus();
const ds = {
  id: "dataset_step7",
  name: "alpha-step7",
  version: STEP7_CORPUS_VERSION,
  description: "",
  license: "Alpha-owned",
  source: STEP7_SOURCE_ID,
  documents: built.documents.map((d) => d.text),
};
const tokenizer = AlphaTokenizer.train(ds.documents, {
  vocabSize: c.vocabSize,
  version: "1.0.0",
  trainedOn: "alpha-step7@1.0.0",
});
const modelConfig = { ...c, vocabSize: Math.max(c.vocabSize, tokenizer.vocabSize) };
const model = new AlphaTransformer(modelConfig);
const tc = createTrainingConfig({
  batchSize: 8,
  seqLen: 256,
  totalSteps: 64,
  learningRate: 0.002,
  warmupSteps: 5,
  evalInterval: 16,
  evalBatches: 2,
  checkpointInterval: 64,
  seed: 1337,
  validationFraction: 0.12,
  gradientAccumulationSteps: 1,
});
const trainer = new AlphaTrainer({ model, tokenizer, dataset: ds, config: tc, runId: "step7-micro-v2" });
console.log(`setup ms ${performance.now() - t0} rss ${rssMb()}`);
const t1 = performance.now();
const lr = c().learningRateAt(trainer.stepCount + 1, trainer.config.schedule);
const acc: import("../alpha/datasets/corpus").TrainingBatch[] = [];
for (let k = 0; k < trainer.config.gradientAccumulationSteps; k++) acc.push(trainer.sampler.next());
const accumulated = trainer.accumulateGradients(acc);
const report = trainer.optimizer.stepWithSchedule(lr);
trainer.optimizer.zeroGrad();
trainer.stepCount = report.step;
console.log(`step1 ms ${performance.now() - t1} loss ${accumulated.meanLoss} rss ${rssMb()} gradNorm ${report.gradNorm}`);
