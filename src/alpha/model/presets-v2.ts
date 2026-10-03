/**
 * Step 7 architecture presets — new, versioned, and additive.
 *
 * The historical presets in `./config.ts` (nano / micro / small) describe the
 * models Steps 1–6 trained and verified. They are not modified here, because
 * checkpoints and served artifacts fingerprint those exact configurations and a
 * historical shape must never move under them.
 *
 * What this file adds is one new architecture for Step 7:
 *
 *   alpha-micro-v2   4 layers, dModel 160, a 256-token context window, and the
 *                    same 768-token vocabulary as Step 5's tokenizer, so the
 *                    frozen evaluation suite stays comparable token for token.
 *
 * `validateStep7Presets()` checks every preset the way `./config` checks the
 * historical ones (divisibility, limits, parameter ceiling), and reports the
 * parameter count computed from the configuration. The Step 7 verification run
 * separately counts the same model's parameters from its tensors, so the number
 * published is verified two independent ways.
 */

import {
  countParameters,
  modelConfigFingerprint,
  validateModelConfig,
  type AlphaModelConfig,
} from "./config";

export type AlphaStep7Preset = "micro-v2";

export const ALPHA_STEP7_PRESETS: Record<AlphaStep7Preset, AlphaModelConfig> = {
  "micro-v2": {
    name: "alpha-micro-v2",
    version: "0.2.0",
    // Step 5's tokenizer, reused deliberately: the frozen suite and the Step 5
    // baseline are measured in this vocabulary, and perplexities from different
    // vocabularies are not comparable.
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
  },
};

/** Validation pass over every Step 7 preset. Throws with the reason on failure. */
export function validateStep7Presets(): {
  preset: AlphaStep7Preset;
  parameterCount: number;
  fingerprint: string;
}[] {
  return (Object.keys(ALPHA_STEP7_PRESETS) as AlphaStep7Preset[]).map((preset) => {
    const config = ALPHA_STEP7_PRESETS[preset];
    validateModelConfig(config);
    return {
      preset,
      parameterCount: countParameters(config),
      fingerprint: modelConfigFingerprint(config),
    };
  });
}
