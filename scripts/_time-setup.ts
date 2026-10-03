import { performance } from "node:perf_hooks";
import {
  ALPHA_MODEL_PRESETS_MICRO_V2,
  microV2Config,
} from "../src/alpha/model/micro-v2-config";
import {
  STEP7_CORPUS_VERSION,
  STEP7_SOURCE_ID,
  buildStep7Corpus,
} from "../src/alpha/datasets/step7-corpus";
import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import { AlphaTransformer } from "../src/alpha/model/transformer";
import { AlphaTrainer, createTrainingConfig } from "../src/alpha/training/trainer";

function rssMb(): string {
  if (process.memoryUsage?.rss) return (process.memoryUsage().rss / 1024 / 1024).toFixed(1);
  return "n/a";
}

const t0 = performance.now();
const built = buildStep7Corpus();
console.log("build corpus ms", performance.now() - t0, "rss", rssMb());

const t1 = performance.now();
const tokenizer = AlphaTokenizer.train(built.documents.map((d) => d.text), {
  vocabSize: 768,
  version: "1.0.0",
  trainedOn: "alpha-step7@1.0.0",
});
console.log("train tokenizer ms", performance.now() - t1, "rss", rssMb());

const t2 = performance.now();
const model = new AlphaTransformer(microV2Config());
console.log("new model ms", performance.now() - t2, "rss", rssMb());

const t3 = performance.now();
const ds = {
  id: "dataset_step7",
  name: "alpha-step7",
  version: STEP7_CORPUS_VERSION,
  description: "",
  license: "Alpha-owned",
  source: STEP7_SOURCE_ID,
  documents: built.documents.map((d) => d.text),
};
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
console.log("new trainer ms", performance.now() - t3, "rss", rssMb());

const t4 = performance.now();
const batch = trainer.sampler.next();
console.log("sampler.next() ms", performance.now() - t4, "rss", rssMb());

const t5 = performance.now();
const acc = trainer.accumulateGradients([batch]);
console.log("accumulateGradients ms", performance.now() - t5, "rss", rssMb());
