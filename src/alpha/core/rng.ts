/**
 * Alpha Core — deterministic pseudo-random number generation.
 *
 * Weight initialisation, dropout and token sampling all draw from this
 * generator, so a run can be reproduced exactly from a seed. Training
 * checkpoints persist the generator state, which is what makes resumed runs
 * deterministic rather than merely "similar".
 */

export type RngState = {
  seed: number;
  calls: number;
};

/** mulberry32 — small, fast, adequate for init and sampling. */
export class AlphaRng {
  readonly seed: number;
  private state: number;
  private calls = 0;

  constructor(seed = 1337) {
    this.seed = seed >>> 0;
    this.state = this.seed;
  }

  /** Uniform in [0, 1). */
  next(): number {
    this.calls++;
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform in [min, max). */
  range(min: number, max: number): number {
    return min + this.next() * (max - min);
  }

  /** Integer in [0, max). */
  int(max: number): number {
    return Math.floor(this.next() * max);
  }

  /** Standard normal via Box-Muller (no cached spare, keeps state simple). */
  normal(): number {
    let u = 0;
    while (u === 0) u = this.next();
    const v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Fisher-Yates shuffle in place. */
  shuffle<T>(items: T[]): T[] {
    for (let i = items.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const tmp = items[i];
      items[i] = items[j];
      items[j] = tmp;
    }
    return items;
  }

  /** Sample an index from a probability distribution. */
  sampleFrom(probabilities: number[] | Float32Array): number {
    const r = this.next();
    let acc = 0;
    for (let i = 0; i < probabilities.length; i++) {
      acc += probabilities[i];
      if (r < acc) return i;
    }
    return probabilities.length - 1;
  }

  fork(salt: number): AlphaRng {
    return new AlphaRng((this.seed ^ Math.imul(salt + 1, 0x9e3779b9)) >>> 0);
  }

  /**
   * Export state so a training checkpoint can resume this exact stream.
   * Replaying `calls` draws restores the position exactly.
   */
  saveState(): RngState {
    return { seed: this.seed, calls: this.calls };
  }

  static fromState(state: RngState): AlphaRng {
    const rng = new AlphaRng(state.seed);
    for (let i = 0; i < state.calls; i++) rng.next();
    return rng;
  }
}
