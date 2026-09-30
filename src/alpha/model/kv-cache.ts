/**
 * Alpha KV cache — incremental decoding for Alpha's own transformer.
 *
 * Autoregressive generation without a cache re-runs the entire prefix for every
 * new token, so the cost of a `T`-token completion is O(T^2) attention work.
 * This cache stores the per-layer keys and values of the positions already
 * processed, so each new token only computes its own query against the stored
 * keys. The result must be *the same tokens*, not merely similar ones: the
 * cached path is verified against the uncached path in the test suite.
 *
 * This is inference-only by construction. It stores raw `Float32Array`s and
 * records no autograd graph, so it cannot be used to train — which is the point.
 * Training always goes through `AlphaTransformer.forward`, which builds the real
 * graph.
 *
 * Layout, per layer:
 *   keys   [contextLength, dModel] — one row per cached position
 *   values [contextLength, dModel]
 *
 * A separate head-major scratch buffer is not needed: the attention score
 * computation reads the cache directly, keeping the arithmetic identical to the
 * batched path.
 */

import { AlphaValidationError } from "../core/errors";
import type { AlphaModelConfig } from "./config";

/** One layer's cached keys and values. */
export type KvLayerCache = {
  keys: Float32Array;
  values: Float32Array;
};

export type KvCache = {
  /** Number of positions currently held. */
  length: number;
  /** Maximum positions this cache can hold (the model's context length). */
  capacity: number;
  layers: KvLayerCache[];
  /** Total float32 cells allocated — the memory cost of the cache. */
  bytes: number;
  /** Increments once per appended token, for reporting cache behaviour. */
  writes: number;
  /** Increments once per token served from cache rather than recomputed. */
  hits: number;
};

/** Allocates a cache sized for one batch position per model context window. */
export function createKvCache(config: AlphaModelConfig): KvCache {
  if (config.nLayers < 1) {
    throw new AlphaValidationError("model", "a KV cache needs at least one layer");
  }
  const capacity = config.contextLength;
  const cells = capacity * config.dModel;
  const layers: KvLayerCache[] = [];
  for (let i = 0; i < config.nLayers; i++) {
    layers.push({ keys: new Float32Array(cells), values: new Float32Array(cells) });
  }
  return {
    length: 0,
    capacity,
    layers,
    bytes: layers.length * cells * 4 * 2,
    writes: 0,
    hits: 0,
  };
}

/** Empties a cache in place, keeping the allocation so the loop can be reused. */
export function resetKvCache(cache: KvCache): KvCache {
  cache.length = 0;
  cache.writes = 0;
  cache.hits = 0;
  return cache;
}

/** True when the cache can accept another position. */
export function kvCacheHasRoom(cache: KvCache): boolean {
  return cache.length < cache.capacity;
}

/**
 * Writes one position's keys and values into a single layer's row, at the
 * given absolute position.
 *
 * The cache length is owned by the caller: a forward pass walks its layers
 * first and then advances the length once, so every layer's row for a position
 * exists before anything attends over it.
 */
export function kvCacheWriteLayer(
  cache: KvCache,
  layerIndex: number,
  position: number,
  keysAt: Float32Array,
  valuesAt: Float32Array,
): void {
  if (position < 0 || position >= cache.capacity) {
    throw new AlphaValidationError(
      "model",
      `kvCacheWriteLayer: position ${position} is outside the cache capacity ${cache.capacity}`,
    );
  }
  const layer = cache.layers[layerIndex];
  if (!layer) {
    throw new AlphaValidationError("model", `kvCacheWriteLayer: no layer ${layerIndex}`);
  }
  const dModel = keysAt.length;
  const offset = position * dModel;
  layer.keys.set(keysAt, offset);
  layer.values.set(valuesAt, offset);
}

/** Records `count` newly cached positions and any reused ones. */
export function kvCacheCommit(cache: KvCache, count: number, reused: number): KvCache {
  cache.length += count;
  cache.writes += count;
  cache.hits += reused;
  return cache;
}

/**
 * Number of positions that must be dropped to make room, dropping from the
 * left. Alpha's window is a hard limit, so the caller decides whether to trim
 * or to stop; this only reports the cost.
 */
export function kvCacheOverflow(cache: KvCache, extra: number): number {
  return Math.max(0, cache.length + extra - cache.capacity);
}

/** Drops the oldest `count` positions by shifting the stored rows left. */
export function kvCacheDropOldest(cache: KvCache, count: number): KvCache {
  if (count <= 0) return cache;
  const dModel = cache.layers[0].keys.length / cache.capacity;
  const drop = Math.min(count, cache.length);
  for (const layer of cache.layers) {
    layer.keys.copyWithin(0, drop * dModel, cache.length * dModel);
    layer.values.copyWithin(0, drop * dModel, cache.length * dModel);
  }
  cache.length -= drop;
  return cache;
}

/** Cache occupancy report, safe to attach to a generation result. */
export type KvCacheStats = {
  used: boolean;
  positions: number;
  capacity: number;
  bytes: number;
  writes: number;
  hits: number;
  /** 0..1 — how full the cache is. */
  occupancy: number;
};

export function kvCacheStats(cache: KvCache | null, used: boolean): KvCacheStats {
  if (!cache) {
    return { used: false, positions: 0, capacity: 0, bytes: 0, writes: 0, hits: 0, occupancy: 0 };
  }
  return {
    used,
    positions: cache.length,
    capacity: cache.capacity,
    bytes: cache.bytes,
    writes: cache.writes,
    hits: cache.hits,
    occupancy: cache.capacity > 0 ? Number((cache.length / cache.capacity).toFixed(4)) : 0,
  };
}
