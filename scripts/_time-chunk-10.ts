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
  vocabSize: 768,
  version: "1.0.0",
  trainedOn: "alpha-step7@1.0.0",
});
const model = new AlphaTransformer(microV2Config());
const tc = createTrainingConfig({
  batchSize: 8,
  seqLen: 256,
  totalSteps: 10,
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
const summary = trainer.trainToCompletion();
const ms = performance.now() - t1;
console.log(`chunk10 ms ${ms} rss ${rssMb()}`);
console.log(`per-step ms ${ms / 10}`);
console.log(`last loss ${summary.lastLoss}`);
console.log(`last checkpoint id ${summary.checkpoint?.id} step ${summary.checkpoint?.step} size ${summary.checkpoint?.sizeBytes}`);
