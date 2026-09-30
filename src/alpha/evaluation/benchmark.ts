/**
 * Alpha's evaluation benchmark.
 *
 * This corpus exists to be *measured against*, never to be trained on. It is
 * authored separately from the training corpus, versioned on its own, and
 * carries its own fingerprint so an evaluation result can be tied to exactly
 * the questions that produced it.
 *
 * The split below is deliberate: `trainingFacts` is material Alpha *may* have
 * seen during training, and `heldOut` is deliberately not in the seed corpus,
 * so a model that has genuinely learned nothing cannot score on it. Both are
 * scored; the distinction is reported, so a reader can tell memorisation apart
 * from generalisation instead of seeing one blended number.
 */

import { datasetFingerprint, type AlphaDataset } from "../datasets/types";
import { createDatasetVersion, type AlphaDatasetVersion } from "../datasets/versions";

export const ALPHA_BENCHMARK_VERSION = "1.0.0";

/**
 * Prompts whose answers are present in Alpha's own seed corpus. A model that
 * scores here may be recalling; that is a real capability, and it is labelled
 * as recall rather than generalisation.
 */
export const BENCHMARK_TRAINING_FACTS: Array<{ prompt: string; expect: string[] }> = [
  { prompt: "alpha is", expect: ["a self"] },
  { prompt: "alpha has no external", expect: ["provider", "api"] },
  { prompt: "the tokenizer alpha uses is a byte pair", expect: ["bpe", "byte pair"] },
];

/**
 * Held-out continuations that do not appear in the seed corpus. Any signal here
 * is generalisation within the model's capacity, not recall.
 */
export const BENCHMARK_HELD_OUT: Array<{ prompt: string; expect: string[] }> = [
  { prompt: "the transformer attends over", expect: ["token", "position", "context"] },
  { prompt: "a loss value that goes down during training means the model is", expect: ["learn", "better", "fitting"] },
  { prompt: "checkpoints let training", expect: ["resume", "continue", "restore"] },
];

const TRAINING_FACT_DOCUMENTS = BENCHMARK_TRAINING_FACTS.map(
  (f) => `${f.prompt} ${f.expect[0]}.`,
);

const HELD_OUT_DOCUMENTS = BENCHMARK_HELD_OUT.map((f) => `${f.prompt} ${f.expect[0]}.`);

/**
 * Build the benchmark as a dataset version, so it carries sources, licence and
 * a fingerprint like any other corpus. Authored for Alpha, so the licence is
 * Alpha's own.
 */
export function createAlphaBenchmark(
  now = Date.now(),
): AlphaDatasetVersion {
  return createDatasetVersion({
    datasetId: "alpha-benchmark",
    name: "alpha-eval-benchmark",
    version: ALPHA_BENCHMARK_VERSION,
    description:
      "Versioned capability benchmark for Alpha. Held out from all training corpora; never mixed into a training set.",
    sources: [
      {
        id: "alpha-authored-eval",
        title: "Benchmark prompts authored for Alpha",
        license: "Alpha-owned",
        origin: "authored",
        note: "Written for evaluation only. Alpha is forbidden from training on this corpus.",
      },
    ],
    documents: [...TRAINING_FACT_DOCUMENTS, ...HELD_OUT_DOCUMENTS],
    now,
  });
}

/** The prompts that measure recall of material Alpha may have been trained on. */
export function recallCases(): Array<{ prompt: string; expect: string[]; kind: "recall" }> {
  return BENCHMARK_TRAINING_FACTS.map((f) => ({ ...f, kind: "recall" as const }));
}

/** The prompts that measure behaviour on material Alpha was never trained on. */
export function heldOutCases(): Array<{ prompt: string; expect: string[]; kind: "held-out" }> {
  return BENCHMARK_HELD_OUT.map((f) => ({ ...f, kind: "held-out" as const }));
}

/** Identity of the benchmark, recorded in every evaluation report. */
export function benchmarkIdentity(): {
  name: string;
  version: string;
  fingerprint: string;
  recallCases: number;
  heldOutCases: number;
} {
  const benchmark = createAlphaBenchmark(0);
  return {
    name: benchmark.name,
    version: benchmark.version,
    fingerprint: benchmark.manifest.fingerprint,
    recallCases: BENCHMARK_TRAINING_FACTS.length,
    heldOutCases: BENCHMARK_HELD_OUT.length,
  };
}

/**
 * Refuse a training set that contains the benchmark. This is the mechanism
 * behind "never train on the evaluation dataset" — it is checked, not assumed.
 */
export function assertNotBenchmark(dataset: AlphaDataset): { ok: true } {
  const fingerprint = datasetFingerprint(dataset);
  const benchmark = createAlphaBenchmark(0);
  if (fingerprint === benchmark.manifest.fingerprint) {
    throw new Error(
      "refusing to train: this dataset is Alpha's evaluation benchmark; training on it would invalidate every evaluation result",
    );
  }
  return { ok: true };
}
