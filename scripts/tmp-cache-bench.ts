/**
 * Verifies the KV-cache decode path and measures it honestly (with warmup).
 * This is a scratch verification run, not part of the test suite.
 */
import { AlphaInferenceEngine, SAMPLING_PRESETS } from "../src/alpha/inference/engine";
import { AlphaTransformer } from "../src/alpha/model/transformer";
import { createModelConfig } from "../src/alpha/model/config";
import { AlphaTokenizer } from "../src/alpha/tokenizer/bpe";
import { ALPHA_SEED_CORPUS } from "../src/alpha/datasets/seed-corpus";

const config = createModelConfig({ preset: "nano" });
const tokenizer = AlphaTokenizer.train(ALPHA_SEED_CORPUS.documents, {
  vocabSize: config.vocabSize,
});
config.vocabSize = tokenizer.vocabSize;
const model = new AlphaTransformer(config, 1337);
model.disableGrad();

const prompt = "Alpha is a self owned system that trains its own weights and";
const cached = new AlphaInferenceEngine({ model, tokenizer, useCache: true });
const uncached = new AlphaInferenceEngine({ model, tokenizer, useCache: false });

// Warmup so the JIT is not credited to one path.
for (let i = 0; i < 3; i++) {
  cached.generate(prompt, { ...SAMPLING_PRESETS.greedy, maxNewTokens: 24 });
  uncached.generate(prompt, { ...SAMPLING_PRESETS.greedy, maxNewTokens: 24 });
}

for (const tokens of [16, 32, 60]) {
  const sampling = { ...SAMPLING_PRESETS.greedy, maxNewTokens: tokens };
  const a = cached.generate(prompt, sampling);
  const b = uncached.generate(prompt, sampling);
  const same = a.tokenIds.join() === b.tokenIds.join();

  // Repeat each a few times and take the median.
  const timeOf = (engine: AlphaInferenceEngine): number => {
    const runs: number[] = [];
    for (let i = 0; i < 5; i++) {
      const t0 = performance.now();
      engine.generate(prompt, sampling);
      runs.push(performance.now() - t0);
    }
    runs.sort((x, y) => x - y);
    return runs[Math.floor(runs.length / 2)];
  };
  const cachedMs = timeOf(cached);
  const uncachedMs = timeOf(uncached);

  console.log(
    `tokens=${String(tokens).padStart(2)} identical=${same} ` +
      `cached=${cachedMs.toFixed(0)}ms uncached=${uncachedMs.toFixed(0)}ms ` +
      `speedup=${(uncachedMs / cachedMs).toFixed(2)}x ` +
      `prefill=${a.timing.prefillMs.toFixed(1)}ms decode=${a.timing.decodeMs.toFixed(1)}ms ` +
      `cachePos=${a.cache.positions}/${a.cache.capacity} bytes=${a.cache.bytes}`,
  );
}
