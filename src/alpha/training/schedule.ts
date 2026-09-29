/**
 * Alpha Training Engine — learning-rate schedules.
 *
 * Warmup followed by decay is what keeps a tiny transformer from diverging in
 * the first few dozen steps, so this is part of the engine rather than a
 * hard-coded constant.
 */

export type ScheduleKind = "constant" | "linear-decay" | "cosine" | "inverse-sqrt";

export type ScheduleConfig = {
  kind: ScheduleKind;
  peakLearningRate: number;
  warmupSteps: number;
  totalSteps: number;
  /** Final fraction of the peak rate (cosine/linear-decay). */
  minFactor: number;
};

export function defaultSchedule(totalSteps: number, peakLearningRate: number): ScheduleConfig {
  return {
    kind: "cosine",
    peakLearningRate,
    warmupSteps: Math.max(1, Math.round(totalSteps * 0.1)),
    totalSteps,
    minFactor: 0.1,
  };
}

/**
 * Learning rate at a given step (1-indexed step number).
 * Deterministic and pure: resuming from step N reproduces the same schedule.
 */
export function learningRateAt(step: number, config: ScheduleConfig): number {
  if (step <= 0) return 0;
  const peak = config.peakLearningRate;
  if (config.warmupSteps > 0 && step <= config.warmupSteps) {
    return (peak * step) / config.warmupSteps;
  }
  const progress = Math.min(
    1,
    Math.max(0, (step - config.warmupSteps) / Math.max(1, config.totalSteps - config.warmupSteps)),
  );
  switch (config.kind) {
    case "constant":
      return peak;
    case "linear-decay":
      return peak * (1 - progress * (1 - config.minFactor));
    case "cosine": {
      const cosine = 0.5 * (1 + Math.cos(Math.PI * progress));
      return peak * (config.minFactor + (1 - config.minFactor) * cosine);
    }
    case "inverse-sqrt":
      return peak / Math.sqrt(Math.max(1, step - config.warmupSteps + 1));
  }
}

export function describeSchedule(config: ScheduleConfig): string {
  const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
  return `${config.kind}, peak ${config.peakLearningRate}, warmup ${config.warmupSteps} steps, floor ${percent(
    config.minFactor,
  )}`;
}
