/**
 * Alpha Observability — metrics.
 *
 * Counters, gauges and histograms with a bounded sample window. Percentiles are
 * computed from the actual samples in the window, not estimated from a
 * distribution, so a reported p95 is a p95 of what happened.
 */

export type MetricKind = "counter" | "gauge" | "histogram";

export type HistogramSummary = {
  count: number;
  sum: number;
  min: number;
  max: number;
  average: number;
  p50: number;
  p90: number;
  p95: number;
  p99: number;
};

export type MetricSnapshot = {
  kind: MetricKind;
  name: string;
  value: number;
  unit: string;
  samples: number;
  histogram: HistogramSummary | null;
};

type Series = {
  kind: MetricKind;
  name: string;
  unit: string;
  value: number;
  samples: number[];
  windowSize: number;
};

/** Metric names Alpha emits. Kept as constants so dashboards cannot drift. */
export const ALPHA_METRICS = {
  inferenceRequests: "alpha.inference.requests",
  inferenceLatencyMs: "alpha.inference.latency_ms",
  inferenceTokensGenerated: "alpha.inference.tokens_generated",
  inferenceTokensPrompt: "alpha.inference.tokens_prompt",
  inferenceStopReason: "alpha.inference.stop_reason",
  embeddingRequests: "alpha.embedding.requests",
  trainingSteps: "alpha.training.steps",
  trainingLoss: "alpha.training.loss",
  trainingTokens: "alpha.training.tokens",
  ragIngestions: "alpha.rag.ingestions",
  ragRetrievals: "alpha.rag.retrievals",
  toolExecutions: "alpha.tool.executions",
  toolFailures: "alpha.tool.failures",
  agentRuns: "alpha.agent.runs",
  agentSteps: "alpha.agent.steps",
  workflowExecutions: "alpha.workflow.executions",
  workflowFailures: "alpha.workflow.failures",
  memoryWrites: "alpha.memory.writes",
  policyDenials: "alpha.security.policy_denials",
  rateLimitHits: "alpha.security.rate_limit_hits",
  errors: "alpha.errors",
} as const;

export class AlphaMetricsRegistry {
  private series = new Map<string, Series>();
  private readonly defaultWindow: number;

  constructor(options: { sampleWindow?: number } = {}) {
    this.defaultWindow = options.sampleWindow ?? 500;
  }

  private seriesFor(name: string, kind: MetricKind, unit: string, windowSize?: number): Series {
    const existing = this.series.get(name);
    if (existing) return existing;
    const series: Series = {
      kind,
      name,
      unit,
      value: 0,
      samples: [],
      windowSize: windowSize ?? this.defaultWindow,
    };
    this.series.set(name, series);
    return series;
  }

  increment(name: string, value = 1, unit = "count"): number {
    const series = this.seriesFor(name, "counter", unit);
    series.value += value;
    return series.value;
  }

  setGauge(name: string, value: number, unit = "value"): void {
    const series = this.seriesFor(name, "gauge", unit);
    series.value = value;
  }

  observe(name: string, value: number, unit = "ms"): HistogramSummary {
    const series = this.seriesFor(name, "histogram", unit);
    series.samples.push(value);
    if (series.samples.length > series.windowSize) {
      series.samples = series.samples.slice(-series.windowSize);
    }
    series.value = this.histogramSummary(series.samples).average;
    return this.histogramSummary(series.samples);
  }

  private histogramSummary(samples: number[]): HistogramSummary {
    if (samples.length === 0) {
      return { count: 0, sum: 0, min: 0, max: 0, average: 0, p50: 0, p90: 0, p95: 0, p99: 0 };
    }
    const sorted = [...samples].sort((a, b) => a - b);
    const percentile = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))];
    const sum = samples.reduce((total, value) => total + value, 0);
    return {
      count: samples.length,
      sum: Number(sum.toFixed(4)),
      min: sorted[0],
      max: sorted[sorted.length - 1],
      average: Number((sum / samples.length).toFixed(4)),
      p50: percentile(0.5),
      p90: percentile(0.9),
      p95: percentile(0.95),
      p99: percentile(0.99),
    };
  }

  value(name: string): number {
    return this.series.get(name)?.value ?? 0;
  }

  histogram(name: string): HistogramSummary | null {
    const series = this.series.get(name);
    if (!series || series.kind !== "histogram") return null;
    return this.histogramSummary(series.samples);
  }

  snapshot(): MetricSnapshot[] {
    return [...this.series.values()]
      .map((series) => ({
        kind: series.kind,
        name: series.name,
        value: Number(series.value.toFixed(4)),
        unit: series.unit,
        samples: series.samples.length,
        histogram: series.kind === "histogram" ? this.histogramSummary(series.samples) : null,
      }))
      .sort((a, b) => (a.name < b.name ? -1 : 1));
  }

  reset(): void {
    this.series.clear();
  }
}
