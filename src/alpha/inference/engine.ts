/**
 * Alpha Inference Engine — local generation.
 *
 * Prompt in, tokens out, sampled from Alpha's own weights. There is no fallback
 * path: if Alpha is untrained, the engine still runs (the architecture is real)
 * but every result carries `modelStage: "untrained"` and a warning saying so.
 * That is deliberately impossible to miss — Alpha must never be presented as a
 * finished model because a caller forgot to check.
 *
 * Note on performance: generation re-runs the full prefix each step (no KV
 * cache yet). KV caching is listed as in-development in the architecture docs.
 */

import { AlphaRng } from "../core/rng";
import { AlphaValidationError } from "../core/errors";
import { setGradEnabled } from "../core/tensor";
import type { AlphaModelStage } from "../core/types";
import type { AlphaTransformer } from "../model/transformer";
import type { AlphaTokenizer } from "../tokenizer/bpe";

export type SamplingConfig = {
  temperature: number;
  topK: number;
  topP: number;
  maxNewTokens: number;
  repetitionPenalty: number;
  stopSequences: string[];
  seed: number;
};

export const DEFAULT_SAMPLING: SamplingConfig = {
  temperature: 0.8,
  topK: 40,
  topP: 0.95,
  maxNewTokens: 64,
  repetitionPenalty: 1.1,
  stopSequences: [],
  seed: 2026,
};

export type StopReason = "eos" | "stop-sequence" | "max-tokens" | "context-limit";

export type GenerationResult = {
  text: string;
  promptTokens: number;
  generatedTokens: number;
  stopReason: StopReason;
  latencyMs: number;
  tokensPerSecond: number;
  /** Honest provenance of the weights that produced this text. */
  modelStage: AlphaModelStage;
  modelName: string;
  warning: string | null;
  /** Per-token negative log-likelihood of what was actually generated. */
  tokenLogProbs: number[];
  sampling: SamplingConfig;
};

export type GenerationStreamChunk = {
  token: string;
  text: string;
  index: number;
  done: boolean;
  stopReason?: StopReason;
};

export type InferenceEngineOptions = {
  model: AlphaTransformer;
  tokenizer: AlphaTokenizer;
  stage?: AlphaModelStage;
  /** Hard cap on prompt + completion length; defaults to the model context. */
  maxContextTokens?: number;
};

function untrainedWarning(stage: AlphaModelStage): string | null {
  switch (stage) {
    case "untrained":
    case "architecture":
      return "These weights are random initialisation. Alpha has not been trained yet, so this output is not meaningful language — it is the raw behaviour of an untrained transformer.";
    case "trained":
      return "Alpha was trained from scratch on the project corpus. Expect the style and vocabulary of that corpus only.";
    case "fine-tuned":
      return "Alpha was trained from scratch and then fine-tuned on a narrower corpus.";
    case "production":
      return null;
  }
}

export class AlphaInferenceEngine {
  readonly model: AlphaTransformer;
  readonly tokenizer: AlphaTokenizer;
  readonly stage: AlphaModelStage;
  readonly maxContextTokens: number;

  constructor(options: InferenceEngineOptions) {
    this.model = options.model;
    this.tokenizer = options.tokenizer;
    this.stage = options.stage ?? "untrained";
    this.maxContextTokens = Math.min(
      options.maxContextTokens ?? options.model.config.contextLength,
      options.model.config.contextLength,
    );
  }

  private resolveSampling(partial: Partial<SamplingConfig> = {}): SamplingConfig {
    const config = { ...DEFAULT_SAMPLING, ...partial };
    if (config.temperature < 0) {
      throw new AlphaValidationError("inference", "temperature must be >= 0");
    }
    if (config.topP <= 0 || config.topP > 1) {
      throw new AlphaValidationError("inference", "topP must be in (0, 1]");
    }
    return config;
  }

  /**
   * Turn one row of logits into a token id.
   * temperature -> repetition penalty -> top-k -> top-p -> sample.
   */
  private sampleNextToken(
    logits: Float32Array,
    generated: number[],
    config: SamplingConfig,
    rng: AlphaRng,
  ): { id: number; logProb: number } {
    const vocab = logits.length;
    const scores = new Float32Array(vocab);
    for (let i = 0; i < vocab; i++) scores[i] = logits[i];

    if (config.repetitionPenalty > 0 && config.repetitionPenalty !== 1) {
      for (const id of generated) {
        if (id < 0 || id >= vocab) continue;
        scores[id] = scores[id] > 0 ? scores[id] / config.repetitionPenalty : scores[id] * config.repetitionPenalty;
      }
    }

    // Disallow structural special tokens in free generation.
    const banned = new Set<number>([this.tokenizer.padId]);
    for (const id of banned) if (id >= 0 && id < vocab) scores[id] = -Infinity;

    const temperature = config.temperature > 0 ? config.temperature : 1;
    const indices = Array.from({ length: vocab }, (_, i) => i);

    // Softmax with temperature.
    let max = -Infinity;
    for (let i = 0; i < vocab; i++) {
      scores[i] /= temperature;
      if (scores[i] > max) max = scores[i];
    }
    let sum = 0;
    const probs = new Float32Array(vocab);
    for (let i = 0; i < vocab; i++) {
      const e = Math.exp(scores[i] - max);
      probs[i] = e;
      sum += e;
    }
    for (let i = 0; i < vocab; i++) probs[i] /= sum;

    // Top-k.
    let candidates = indices;
    if (config.topK > 0 && config.topK < vocab) {
      candidates = [...indices].sort((a, b) => probs[b] - probs[a]).slice(0, config.topK);
    }

    // Top-p (nucleus) over the remaining candidates.
    if (config.topP < 1) {
      const sorted = [...candidates].sort((a, b) => probs[b] - probs[a]);
      const kept: number[] = [];
      let cumulative = 0;
      for (const id of sorted) {
        kept.push(id);
        cumulative += probs[id];
        if (cumulative >= config.topP) break;
      }
      candidates = kept;
    }

    let mass = 0;
    for (const id of candidates) mass += probs[id];
    if (mass <= 0) {
      const fallback = candidates[0] ?? 0;
      return { id: fallback, logProb: Math.log(probs[fallback] + 1e-12) };
    }
    const r = rng.next() * mass;
    let acc = 0;
    let chosen = candidates[candidates.length - 1];
    for (const id of candidates) {
      acc += probs[id];
      if (r <= acc) {
        chosen = id;
        break;
      }
    }
    return { id: chosen, logProb: Math.log(probs[chosen] + 1e-12) };
  }

  /** Full generation, no streaming. */
  generate(prompt: string, sampling: Partial<SamplingConfig> = {}): GenerationResult {
    const config = this.resolveSampling(sampling);
    const started = Date.now();
    const rng = new AlphaRng(config.seed);
    const promptEncoding = this.tokenizer.encodeDetailed(prompt, {
      maxLength: Math.max(1, this.maxContextTokens - 1),
      truncation: "left",
      addBos: true,
    });
    const context = [...promptEncoding.ids];
    const generated: number[] = [];
    const tokenLogProbs: number[] = [];
    let stopReason: StopReason = "max-tokens";
    let text = "";

    setGradEnabled(false);
    try {
      while (generated.length < config.maxNewTokens) {
        if (context.length >= this.maxContextTokens) {
          stopReason = "context-limit";
          break;
        }
        const logits = this.forwardLastRow(context);
        const { id, logProb } = this.sampleNextToken(logits, generated, config, rng);
        if (id === this.tokenizer.eosId) {
          stopReason = "eos";
          break;
        }
        generated.push(id);
        tokenLogProbs.push(logProb);
        context.push(id);
        text = this.tokenizer.decode(generated);
        const matched = config.stopSequences.find((seq) => seq.length > 0 && text.endsWith(seq));
        if (matched) {
          text = text.slice(0, text.length - matched.length);
          stopReason = "stop-sequence";
          break;
        }
      }
    } finally {
      setGradEnabled(true);
    }

    const latencyMs = Date.now() - started;
    return {
      text,
      promptTokens: context.length - generated.length,
      generatedTokens: generated.length,
      stopReason,
      latencyMs,
      tokensPerSecond:
        latencyMs > 0 ? Number(((generated.length / latencyMs) * 1000).toFixed(2)) : 0,
      modelStage: this.stage,
      modelName: this.model.config.name,
      warning: untrainedWarning(this.stage),
      tokenLogProbs,
      sampling: config,
    };
  }

  /** Incremental generation for a chat-style UI. */
  async *generateStream(
    prompt: string,
    sampling: Partial<SamplingConfig> = {},
  ): AsyncGenerator<GenerationStreamChunk, GenerationResult, void> {
    const config = this.resolveSampling(sampling);
    const started = Date.now();
    const rng = new AlphaRng(config.seed);
    const promptEncoding = this.tokenizer.encodeDetailed(prompt, {
      maxLength: Math.max(1, this.maxContextTokens - 1),
      truncation: "left",
      addBos: true,
    });
    const context = [...promptEncoding.ids];
    const generated: number[] = [];
    const tokenLogProbs: number[] = [];
    let stopReason: StopReason = "max-tokens";
    let text = "";

    setGradEnabled(false);
    try {
      let index = 0;
      while (generated.length < config.maxNewTokens) {
        if (context.length >= this.maxContextTokens) {
          stopReason = "context-limit";
          break;
        }
        const logits = this.forwardLastRow(context);
        const { id, logProb } = this.sampleNextToken(logits, generated, config, rng);
        if (id === this.tokenizer.eosId) {
          stopReason = "eos";
          break;
        }
        generated.push(id);
        tokenLogProbs.push(logProb);
        context.push(id);
        text = this.tokenizer.decode(generated);
        const matched = config.stopSequences.find((seq) => seq.length > 0 && text.endsWith(seq));
        const tokenText = this.tokenizer.decode([id]);
        if (matched) {
          stopReason = "stop-sequence";
          const trimmed = text.slice(0, text.length - matched.length);
          yield {
            token: tokenText,
            text: trimmed,
            index: index++,
            done: true,
            stopReason,
          };
          text = trimmed;
          break;
        }
        yield { token: tokenText, text, index: index++, done: false };
      }
    } finally {
      setGradEnabled(true);
    }

    const latencyMs = Date.now() - started;
    return {
      text,
      promptTokens: context.length - generated.length,
      generatedTokens: generated.length,
      stopReason,
      latencyMs,
      tokensPerSecond:
        latencyMs > 0 ? Number(((generated.length / latencyMs) * 1000).toFixed(2)) : 0,
      modelStage: this.stage,
      modelName: this.model.config.name,
      warning: untrainedWarning(this.stage),
      tokenLogProbs,
      sampling: config,
    };
  }

  /** Next-token scores for a prompt — used by diagnostics in the workspace. */
  scoreNextTokens(context: number[]): Float32Array {
    setGradEnabled(false);
    try {
      return this.forwardLastRow(context);
    } finally {
      setGradEnabled(true);
    }
  }

  private forwardLastRow(context: number[]): Float32Array {
    const tokens = context.slice(-this.maxContextTokens);
    const ids = Int32Array.from(tokens);
    const result = this.model.forward(ids, 1, ids.length, { training: false });
    const vocab = this.model.config.vocabSize;
    const offset = (ids.length - 1) * vocab;
    return result.logits.data.subarray(offset, offset + vocab) as Float32Array;
  }
}

/** Exposed for tests and diagnostics: greedy argmax over a logits vector. */
export function argmax(logits: Float32Array | number[]): number {
  let best = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > bestValue) {
      bestValue = logits[i];
      best = i;
    }
  }
  return best;
}

/** Mean negative log-likelihood over the given token log-probabilities. */
export function meanNll(logProbs: number[]): number {
  if (logProbs.length === 0) return 0;
  return -logProbs.reduce((sum, value) => sum + value, 0) / logProbs.length;
}
