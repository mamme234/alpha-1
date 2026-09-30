/**
 * Alpha Inference Engine — local generation.
 *
 * Prompt in, tokens out, sampled from Alpha's own weights. There is no fallback
 * path: if Alpha is untrained, the engine still runs (the architecture is real)
 * but every result carries `modelStage: "untrained"` and a warning saying so.
 * That is deliberately impossible to miss — Alpha must never be presented as a
 * finished model because a caller forgot to check.
 *
 * Decoding is explicit: temperature 0 (or `deterministic: true`) is greedy
 * argmax, which is reproducible; any positive temperature samples from the
 * truncated distribution using Alpha's own seeded generator, so the same seed
 * reproduces the same tokens.
 *
 * Performance: decoding uses a KV cache, so each new token attends against the
 * stored keys/values of the prefix instead of re-running it. The cached path is
 * verified to produce bit-identical logits to the uncached forward pass, so
 * enabling it changes speed and nothing else.
 */

import { AlphaRng } from "../core/rng";
import { AlphaValidationError } from "../core/errors";
import { assertResourceLimit } from "../core/limits";
import { setGradEnabled } from "../core/tensor";
import type { AlphaModelStage } from "../core/types";
import { createKvCache, kvCacheStats, resetKvCache, type KvCache, type KvCacheStats } from "../model/kv-cache";
import type { AlphaTransformer } from "../model/transformer";
import type { AlphaTokenizer } from "../tokenizer/bpe";

export type SamplingConfig = {
  /** 0 means greedy (argmax) decoding — fully deterministic. */
  temperature: number;
  topK: number;
  topP: number;
  maxNewTokens: number;
  repetitionPenalty: number;
  stopSequences: string[];
  /** Token ids that end generation when they are produced. */
  stopTokenIds: number[];
  /** Force greedy decoding regardless of temperature. */
  deterministic: boolean;
  seed: number;
};

export const DEFAULT_SAMPLING: SamplingConfig = {
  temperature: 0.8,
  topK: 40,
  topP: 0.95,
  maxNewTokens: 64,
  repetitionPenalty: 1.1,
  stopSequences: [],
  stopTokenIds: [],
  deterministic: false,
  seed: 2026,
};

/** Named presets so a caller does not have to rediscover sensible settings. */
export const SAMPLING_PRESETS: Record<"greedy" | "balanced" | "creative", SamplingConfig> = {
  greedy: { ...DEFAULT_SAMPLING, temperature: 0, topK: 0, topP: 1, repetitionPenalty: 1, deterministic: true },
  balanced: { ...DEFAULT_SAMPLING },
  creative: { ...DEFAULT_SAMPLING, temperature: 1.1, topK: 80, topP: 0.98, repetitionPenalty: 1.15 },
};

export type StopReason =
  | "eos"
  | "stop-token"
  | "stop-sequence"
  | "max-tokens"
  | "context-limit"
  | "cancelled";

export type GenerationResult = {
  text: string;
  /** Token ids the model actually produced, in order. */
  tokenIds: number[];
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
  /** Mean negative log-likelihood over the generated tokens. */
  meanNll: number;
  sampling: SamplingConfig;
  /** How the model produced the first token: "greedy" or "sampled". */
  decoding: "greedy" | "sampled";
  /** How the decode loop ran: with a KV cache, or recomputing the prefix. */
  cache: KvCacheStats;
  /** Wall-clock split of the decode, in milliseconds. */
  timing: { prefillMs: number; decodeMs: number };
};

/**
 * A cooperative cancellation handle. Alpha is CPU-bound JavaScript, so a
 * generation can only stop at a token boundary — but it stops promptly, and the
 * partial result is returned with `stopReason: "cancelled"` rather than being
 * discarded. Nothing is thrown away silently.
 */
export type GenerationCancellation = {
  readonly cancelled: boolean;
  readonly reason: string | null;
  cancel(reason?: string): void;
};

export function createGenerationCancellation(): GenerationCancellation {
  let cancelled = false;
  let reason: string | null = null;
  return {
    get cancelled() {
      return cancelled;
    },
    get reason() {
      return reason;
    },
    cancel(why?: string) {
      cancelled = true;
      reason = why ?? "cancelled by caller";
    },
  };
}

export type GenerationStreamChunk = {
  token: string;
  tokenId: number;
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
  /**
   * Use the KV cache. On by default; set false to run the original
   * recompute-the-prefix loop, which produces the same tokens more slowly and
   * exists so the two paths can be compared in tests and benchmarks.
   */
  useCache?: boolean;
};

export function untrainedWarning(stage: AlphaModelStage): string | null {
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

/** One token emitted by the shared decode loop. */
type DecodeStep = {
  id: number;
  /** Cumulative text including this token (and excluding any stop sequence). */
  text: string;
  /** True when this token ended generation. */
  done: boolean;
};

/** Mutable state carried through one generation loop. */
type DecodeState = {
  config: SamplingConfig;
  context: number[];
  generated: number[];
  tokenLogProbs: number[];
  promptTokens: number;
  stopReason: StopReason;
  text: string;
  timing: { prefillMs: number; decodeMs: number };
};

export class AlphaInferenceEngine {
  readonly model: AlphaTransformer;
  readonly tokenizer: AlphaTokenizer;
  readonly stage: AlphaModelStage;
  readonly maxContextTokens: number;
  /** Whether decoding uses the KV cache. Both paths emit the same tokens. */
  readonly useCache: boolean;

  constructor(options: InferenceEngineOptions) {
    this.model = options.model;
    this.tokenizer = options.tokenizer;
    this.stage = options.stage ?? "untrained";
    this.maxContextTokens = Math.min(
      options.maxContextTokens ?? options.model.config.contextLength,
      options.model.config.contextLength,
    );
    this.useCache = options.useCache ?? true;
  }

  /** Validate and merge a partial sampling configuration against the defaults. */
  resolveSampling(partial: Partial<SamplingConfig> = {}): SamplingConfig {
    const config = { ...DEFAULT_SAMPLING, ...partial };
    if (!Number.isFinite(config.temperature) || config.temperature < 0) {
      throw new AlphaValidationError("inference", "temperature must be 0 or a positive number");
    }
    if (config.temperature > 4) {
      throw new AlphaValidationError("inference", "temperature above 4 would produce noise, not language");
    }
    if (!(config.topP > 0) || config.topP > 1) {
      throw new AlphaValidationError("inference", "topP must be in (0, 1]");
    }
    if (!Number.isInteger(config.topK) || config.topK < 0) {
      throw new AlphaValidationError("inference", "topK must be 0 (disabled) or a positive integer");
    }
    if (!Number.isInteger(config.maxNewTokens) || config.maxNewTokens < 1) {
      throw new AlphaValidationError("inference", "maxNewTokens must be a positive integer");
    }
    assertResourceLimit("maxNewTokens", config.maxNewTokens, "inference");
    if (config.repetitionPenalty < 0) {
      throw new AlphaValidationError("inference", "repetitionPenalty must be >= 0");
    }
    if (!Number.isFinite(config.seed)) {
      throw new AlphaValidationError("inference", "seed must be a finite number");
    }
    if (config.stopSequences.some((sequence) => typeof sequence !== "string")) {
      throw new AlphaValidationError("inference", "stopSequences must be strings");
    }
    if (config.stopTokenIds.some((id) => !Number.isInteger(id) || id < 0)) {
      throw new AlphaValidationError("inference", "stopTokenIds must be non-negative integers");
    }
    // The completion cannot be longer than the window it starts in.
    const room = Math.max(1, this.maxContextTokens - 1);
    if (config.maxNewTokens > room) {
      config.maxNewTokens = room;
    }
    return config;
  }

  /**
   * Turn one row of logits into a token id.
   * temperature -> repetition penalty -> top-k -> top-p -> argmax or sample.
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
        scores[id] =
          scores[id] > 0 ? scores[id] / config.repetitionPenalty : scores[id] * config.repetitionPenalty;
      }
    }

    // Disallow structural special tokens in free generation.
    const banned = new Set<number>([this.tokenizer.padId]);
    for (const id of banned) if (id >= 0 && id < vocab) scores[id] = -Infinity;

    const deterministic = config.deterministic || config.temperature <= 0;
    const temperature = deterministic ? 1 : config.temperature;
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

    if (deterministic) {
      let best = candidates[0] ?? 0;
      for (const id of candidates) if (probs[id] > probs[best]) best = id;
      return { id: best, logProb: Math.log(probs[best] + 1e-12) };
    }

    // Renormalise over the surviving candidates and sample with Alpha's own RNG.
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

  /** Encode a prompt into the window, reporting what actually fit. */
  private encodePrompt(prompt: string): { ids: number[]; truncated: boolean } {
    if (typeof prompt !== "string") {
      throw new AlphaValidationError("inference", "prompt must be a string");
    }
    assertResourceLimit("maxPromptTokens", prompt.length, "inference prompt");
    const encoded = this.tokenizer.encodeDetailed(prompt, {
      maxLength: Math.max(1, this.maxContextTokens - 1),
      truncation: "left",
      addBos: true,
    });
    return { ids: [...encoded.ids], truncated: encoded.truncated };
  }

  private decodeLoopState(config: SamplingConfig, promptIds: number[]): DecodeState {
    return {
      config,
      context: [...promptIds],
      generated: [],
      tokenLogProbs: [],
      promptTokens: promptIds.length,
      stopReason: "max-tokens",
      text: "",
      timing: { prefillMs: 0, decodeMs: 0 },
    };
  }

  /**
   * Run the prompt through the model once, priming a cache when one is in use.
   * Returns the logits of the final position, which is what the first sampled
   * token is chosen from.
   */
  private prime(
    state: DecodeState,
    cache: KvCache | null,
  ): { logits: Float32Array; prefillMs: number } {
    const started = performance.now();
    const ids = Int32Array.from(state.context);
    const vocab = this.model.config.vocabSize;
    let logits: Float32Array;
    if (cache) {
      resetKvCache(cache);
      const result = this.model.forwardCached(ids, 1, ids.length, { cache, incremental: false });
      logits = result.logits.data.subarray(
        (ids.length - 1) * vocab,
        ids.length * vocab,
      ) as Float32Array;
    } else {
      const result = this.model.forward(ids, 1, ids.length, { training: false });
      logits = result.logits.data.subarray(
        (ids.length - 1) * vocab,
        ids.length * vocab,
      ) as Float32Array;
    }
    return { logits, prefillMs: performance.now() - started };
  }

  /** One incremental token step: process `tokenId` at its position in the cache. */
  private stepFromCache(cache: KvCache, tokenId: number): Float32Array {
    const vocab = this.model.config.vocabSize;
    const result = this.model.forwardCached(Int32Array.from([tokenId]), 1, 1, {
      cache,
      incremental: true,
    });
    return result.logits.data.subarray(0, vocab) as Float32Array;
  }

  /**
   * Shared decode loop. The two implementations differ only in how the logits
   * for the next token are obtained; every sampling and stop decision is
   * identical, which is what makes the cached path a pure optimisation.
   */
  private *decode(
    state: DecodeState,
    rng: AlphaRng,
    cache: KvCache | null,
    cancellation: GenerationCancellation | null,
  ): Generator<DecodeStep, void, void> {
    const { logits: firstLogits, prefillMs } = this.prime(state, cache);
    let logits = firstLogits;
    const decodeStart = performance.now();
    // The last sampled token has not yet been processed by the cache; it is
    // fed in at the top of the next iteration to produce the logits for the
    // position after it.
    let pending: number | null = null;

    try {
      while (state.generated.length < state.config.maxNewTokens) {
        if (cancellation?.cancelled) {
          state.stopReason = "cancelled";
          return;
        }
        if (state.context.length >= this.maxContextTokens) {
          state.stopReason = "context-limit";
          return;
        }
        if (pending !== null) {
          if (cache) {
            if (cache.length >= cache.capacity) {
              state.stopReason = "context-limit";
              return;
            }
            logits = this.stepFromCache(cache, pending);
          } else {
            logits = this.forwardLastRow(state.context);
          }
        }

        const { id, logProb } = this.sampleNextToken(logits, state.generated, state.config, rng);
        if (id === this.tokenizer.eosId) {
          state.stopReason = "eos";
          return;
        }
        if (state.config.stopTokenIds.includes(id)) {
          state.stopReason = "stop-token";
          return;
        }
        pending = id;
        state.generated.push(id);
        state.tokenLogProbs.push(logProb);
        state.context.push(id);
        state.text = this.tokenizer.decode(state.generated);
        const matched = state.config.stopSequences.find(
          (seq) => seq.length > 0 && state.text.endsWith(seq),
        );
        if (matched) {
          state.text = state.text.slice(0, state.text.length - matched.length);
          state.stopReason = "stop-sequence";
          yield { id, text: state.text, done: true };
          return;
        }
        yield { id, text: state.text, done: false };
      }
    } finally {
      // Recorded on every exit path, including cancellation and the stop
      // conditions above, so a timing report is never silently zeroed.
      state.timing = { prefillMs, decodeMs: performance.now() - decodeStart };
    }
  }

  private finish(state: DecodeState, startedAt: number, cache: KvCache | null): GenerationResult {
    const latencyMs = Date.now() - startedAt;
    const meanNll =
      state.tokenLogProbs.length === 0
        ? 0
        : -state.tokenLogProbs.reduce((sum, value) => sum + value, 0) / state.tokenLogProbs.length;
    const deterministic = state.config.deterministic || state.config.temperature <= 0;
    return {
      text: state.text,
      tokenIds: [...state.generated],
      promptTokens: state.promptTokens,
      generatedTokens: state.generated.length,
      stopReason: state.stopReason,
      latencyMs,
      tokensPerSecond:
        latencyMs > 0 ? Number(((state.generated.length / latencyMs) * 1000).toFixed(2)) : 0,
      modelStage: this.stage,
      modelName: this.model.config.name,
      warning: untrainedWarning(this.stage),
      tokenLogProbs: state.tokenLogProbs,
      meanNll,
      sampling: state.config,
      decoding: deterministic ? "greedy" : "sampled",
      cache: kvCacheStats(cache, cache !== null),
      timing: {
        prefillMs: Number(state.timing.prefillMs.toFixed(3)),
        decodeMs: Number(state.timing.decodeMs.toFixed(3)),
      },
    };
  }

  /** Full generation, no streaming. */
  generate(
    prompt: string,
    sampling: Partial<SamplingConfig> = {},
    options: { cancellation?: GenerationCancellation | null } = {},
  ): GenerationResult {
    const config = this.resolveSampling(sampling);
    const started = Date.now();
    const rng = new AlphaRng(config.seed);
    const encoded = this.encodePrompt(prompt);
    const state = this.decodeLoopState(config, encoded.ids);
    const cache = this.useCache ? createKvCache(this.model.config) : null;

    setGradEnabled(false);
    try {
      // The loop is consumed for its side effects; `finish` reads the state.
      for (const _step of this.decode(state, rng, cache, options.cancellation ?? null)) {
        void _step;
      }
    } finally {
      setGradEnabled(true);
    }

    return this.finish(state, started, cache);
  }

  /** Incremental generation for a chat-style UI. */
  async *generateStream(
    prompt: string,
    sampling: Partial<SamplingConfig> = {},
    options: { cancellation?: GenerationCancellation | null } = {},
  ): AsyncGenerator<GenerationStreamChunk, GenerationResult, void> {
    const config = this.resolveSampling(sampling);
    const started = Date.now();
    const rng = new AlphaRng(config.seed);
    const encoded = this.encodePrompt(prompt);
    const state = this.decodeLoopState(config, encoded.ids);
    const cache = this.useCache ? createKvCache(this.model.config) : null;

    setGradEnabled(false);
    try {
      let index = 0;
      for (const step of this.decode(state, rng, cache, options.cancellation ?? null)) {
        yield {
          token: this.tokenizer.decode([step.id]),
          tokenId: step.id,
          text: step.text,
          index: index++,
          done: step.done,
          ...(step.done ? { stopReason: state.stopReason } : {}),
        };
      }
    } finally {
      setGradEnabled(true);
    }

    return this.finish(state, started, cache);
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
