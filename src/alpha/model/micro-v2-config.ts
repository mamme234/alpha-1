/**
 * Alpha model — Step 7 micro-v2 configuration.
 *
 * Frozen preset for the Step 7 experiment. Do not edit after the training run
 * starts: it is the architecture the model being verified was built with.
 *
 * Preset (micro-v2): vocab 768, contextLength 256, dModel 160, nHeads 4,
 * nLayers 4, dFeedForward 640, dropout 0.05, learned positional, tied
 * embeddings, initStd 0.02 — ~1,401,280 parameters.
 */

import { ALPHA_MODEL_PRESETS, AlphaModelConfig, createModelConfig } from "./config";

export const ALPHA_MODEL_PRESETS_MICRO_V2: AlphaModelConfig = {
  name: "alpha-micro-v2",
  version: "0.1.0",
  vocabSize: 768,
  contextLength: 256,
  dModel: 160,
  nHeads: 4,
  nLayers: 4,
  dFeedForward: 640,
  dropout: 0.05,
  normEps: 1e-5,
  positionalEncoding: "learned",
  tieEmbeddings: true,
  initStd: 0.02,
};

/**
 * The Step 7 micro-v2 preset exactly as frozen, at its own 768-token
 * vocabulary. This is the configuration a reader should load for the
 * capability gate.
 */
export function microV2Config(): AlphaModelConfig {
  return { ...ALPHA_MODEL_PRESETS_MICRO_V2 };
}