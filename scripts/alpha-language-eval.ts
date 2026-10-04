/**
 * Alpha — language generation evaluation (read-only).
 *
 * Compares the freshly trained 1.04 M-token checkpoint against the Step 5
 * baseline on the four required prompts, using Alpha's own inference engine
 * and tokenizer — no external model, no external service. It writes no files
 * and trains nothing; it only generates text and measures it.
 *
 *   Prompts: "Hello", "Hi, I want to ask you something.",
 *             "What is Alpha?", "Tell me about Ethiopia."
 *
 * Metrics per (model, prompt, sampling preset):
 *   - the generated text itself
 *   - consecutive-duplicate-token rate   (repetition)
 *   - unique-token ratio                 (diversity)
 *   - mean per-token NLL                 (how surprised the model was)
 *   - stopReason / generatedTokens
 *
 * Comparison is greedy (deterministic, shows what the model believes) plus
 * balanced sampling (shows what a user would actually see).
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { parseCheckpoint } from "../src/alpha/training/checkpoint";
import type { AlphaModelConfig } from "../src/alpha/model/config";
import type { AlphaModelStage } from "../src/alpha/core/types";
import { AlphaTransformer } from "../src/alpha/model/transformer";
import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import {
  AlphaInferenceEngine,
  SAMPLING_PRESETS,
} from "../src/alpha/inference/engine";

const ROOT = join(__dirname, "..");

const PROMPTS = [
  "Hello",
  "Hi, I want to ask you something.",
  "What is Alpha?",
  "Tell me about Ethiopia.",
];

const MAX_NEW = 40;

type ModelSpec = {
  label: string;
  stage: AlphaModelStage;
  checkpointPath?: string;
  /** Step 5's serving artifact holds weights + tokenizer together. */
  artifactPath?: string;
};

const MODELS: ModelSpec[] = [
  {
    label: "Step 5 baseline (24,576 tokens)",
    stage: "trained",
    artifactPath: join(ROOT, "src/alpha/serving/step5-artifact.json"),
  },
  {
    label: "New run (1,044,480 tokens)",
    stage: "trained",
    checkpointPath: join(ROOT, "src/alpha/experiments/language/language-mv2-final.alpha-ckpt.json"),
  },
];

type EngineBundle = {
  engine: AlphaInferenceEngine;
  params: number;
  tokensSeen: number;
  valLoss: number | null;
};

function loadModel(spec: ModelSpec): EngineBundle {
  if (spec.artifactPath) {
    // A serving artifact nests the architecture under `model.config` and stores
    // weights as { config, tensors, shapes } — the same shape loadWeights takes.
    const raw = JSON.parse(readFileSync(spec.artifactPath, "utf8")) as {
      stage: AlphaModelStage;
      model: { config: AlphaModelConfig; parameterCount: number };
      weights: Parameters<AlphaTransformer["loadWeights"]>[0];
      tokenizer: { snapshot: Parameters<typeof AlphaTokenizer.fromJSON>[0] };
      training: {
        steps: number;
        tokensSeen: number;
        validationLoss: number | null;
      };
    };
    const tokenizer = AlphaTokenizer.fromJSON(raw.tokenizer.snapshot);
    const model = new AlphaTransformer(raw.model.config);
    model.loadWeights(raw.weights);
    return {
      // The artifact records its own provenance stage; use it rather than
      // guessing one from the model name.
      engine: new AlphaInferenceEngine({
        model,
        tokenizer,
        stage: raw.stage ?? spec.stage,
      }),
      params: model.parameterCount,
      tokensSeen: raw.training.tokensSeen,
      valLoss: raw.training.validationLoss,
    };
  }

  const ck = parseCheckpoint(readFileSync(spec.checkpointPath!, "utf8"));
  const tokenizer = AlphaTokenizer.fromJSON(ck.tokenizer.snapshot);
  const model = new AlphaTransformer(ck.config);
  model.loadWeights(ck.weights);
  return {
    engine: new AlphaInferenceEngine({ model, tokenizer, stage: spec.stage }),
    params: model.parameterCount,
    tokensSeen: ck.tokensSeen,
    valLoss: ck.metrics.validationLoss,
  };
}

/** Fraction of adjacent token pairs that are identical — a repetition signal. */
function dupRate(ids: number[]): number {
  if (ids.length < 2) return 0;
  let dups = 0;
  for (let i = 1; i < ids.length; i++) if (ids[i] === ids[i - 1]) dups++;
  return dups / (ids.length - 1);
}

function uniqueRatio(ids: number[]): number {
  if (ids.length === 0) return 0;
  return new Set(ids).size / ids.length;
}

const row = (n: number, width: number) => String(n).padStart(width);

for (const spec of MODELS) {
  let bundle: EngineBundle;
  try {
    bundle = loadModel(spec);
  } catch (error) {
    console.log(`\n### ${spec.label}\n  COULD NOT LOAD: ${(error as Error).message}`);
    continue;
  }

  console.log("\n" + "=".repeat(78));
  console.log(`### ${spec.label}`);
  console.log(
    `  params ${bundle.params} | trained tokens ${bundle.tokensSeen}` +
      (bundle.valLoss !== null ? ` | val loss ${bundle.valLoss.toFixed(4)}` : ""),
  );
  console.log("=".repeat(78));

  for (const presetName of ["greedy", "balanced"] as const) {
    const preset = SAMPLING_PRESETS[presetName];
    console.log(`\n--- sampling: ${presetName} (temp ${preset.temperature}, maxNewTokens ${MAX_NEW}) ---`);
    for (const prompt of PROMPTS) {
      const result = bundle.engine.generate(prompt, {
        ...preset,
        maxNewTokens: MAX_NEW,
        seed: 1337,
      });
      const text = result.text.replace(/\s+/g, " ").trim();
      const textLine = text.length > 110 ? `${text.slice(0, 110)}…` : text;
      console.log(`\n  PROMPT: ${JSON.stringify(prompt)}`);
      console.log(`  OUTPUT: ${textLine.length > 0 ? textLine : "(empty)"}`);
      console.log(
        `  tokens ${row(result.generatedTokens, 3)} | dup-rate ${dupRate(result.tokenIds).toFixed(3)}` +
          ` | unique ${uniqueRatio(result.tokenIds).toFixed(3)}` +
          ` | meanNLL ${result.meanNll.toFixed(3)}` +
          ` | stop ${result.stopReason}`,
      );
    }
  }
}

console.log("\n" + "=".repeat(78));
console.log("Reading the numbers: dup-rate near 1.0 means the model is looping on");
console.log("one token; unique near 1.0 means varied output. Compare the two blocks.");
console.log("=".repeat(78));