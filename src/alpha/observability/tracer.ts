/**
 * Alpha Observability — traces and spans.
 *
 * Everything Alpha does that is worth measuring opens a span: inference, a RAG
 * retrieval, a tool call, an agent step, a workflow job, a training step. Spans
 * carry a trace id so a request can be reconstructed end to end, and they are
 * plain objects so they can be persisted (the workspace writes them to Alpha's
 * own tables) or printed.
 */

import { newSpanId, newTraceId } from "../core/types";

export type SpanKind = "internal" | "inference" | "retrieval" | "tool" | "agent" | "workflow" | "training" | "embedding";

export type SpanStatus = "ok" | "error" | "cancelled";

export type Span = {
  id: string;
  traceId: string;
  parentId: string | null;
  name: string;
  kind: SpanKind;
  module: string;
  startMs: number;
  endMs: number | null;
  durationMs: number | null;
  status: SpanStatus;
  attributes: Record<string, unknown>;
  error: string | null;
};

export type TraceSummary = {
  traceId: string;
  rootSpan: string;
  spans: number;
  durationMs: number;
  status: SpanStatus;
  startedAt: number;
  modules: string[];
};

export type TraceEvent = {
  at: number;
  span: Span;
};

export class AlphaTracer {
  private spans: Span[] = [];
  private open = new Map<string, Span>();
  private readonly maxSpans: number;
  private readonly listeners = new Set<(event: TraceEvent) => void>();

  constructor(options: { maxSpans?: number } = {}) {
    this.maxSpans = options.maxSpans ?? 4000;
  }

  /** Begin a span. `traceId` links it to an existing trace when provided. */
  startSpan(input: {
    name: string;
    kind: SpanKind;
    module: string;
    traceId?: string;
    parentId?: string | null;
    attributes?: Record<string, unknown>;
  }): Span {
    const span: Span = {
      id: newSpanId(),
      traceId: input.traceId ?? newTraceId(),
      parentId: input.parentId ?? null,
      name: input.name,
      kind: input.kind,
      module: input.module,
      startMs: Date.now(),
      endMs: null,
      durationMs: null,
      status: "ok",
      attributes: input.attributes ?? {},
      error: null,
    };
    this.spans.push(span);
    this.open.set(span.id, span);
    if (this.spans.length > this.maxSpans) {
      this.spans = this.spans.slice(-Math.floor(this.maxSpans / 2));
    }
    return span;
  }

  endSpan(
    span: Span,
    outcome: { status?: SpanStatus; error?: string | null; attributes?: Record<string, unknown> } = {},
  ): Span {
    span.endMs = Date.now();
    span.durationMs = span.endMs - span.startMs;
    span.status = outcome.status ?? (outcome.error ? "error" : "ok");
    span.error = outcome.error ?? null;
    if (outcome.attributes) span.attributes = { ...span.attributes, ...outcome.attributes };
    this.open.delete(span.id);
    const event: TraceEvent = { at: span.endMs, span };
    for (const listener of this.listeners) listener(event);
    return span;
  }

  /**
   * Run a synchronous function inside a span, recording errors without
   * swallowing them.
   */
  withSpan<T>(
    input: { name: string; kind: SpanKind; module: string; traceId?: string; parentId?: string | null; attributes?: Record<string, unknown> },
    fn: (span: Span) => T,
  ): T {
    const span = this.startSpan(input);
    try {
      const result = fn(span);
      if (result instanceof Promise) {
        return result
          .then((value) => {
            this.endSpan(span);
            return value;
          })
          .catch((error: unknown) => {
            this.endSpan(span, { status: "error", error: error instanceof Error ? error.message : String(error) });
            throw error;
          }) as T;
      }
      this.endSpan(span);
      return result;
    } catch (error) {
      this.endSpan(span, { status: "error", error: error instanceof Error ? error.message : String(error) });
      throw error;
    }
  }

  /** Asynchronous variant, used by tools, agents and workflows. */
  async withSpanAsync<T>(
    input: { name: string; kind: SpanKind; module: string; traceId?: string; parentId?: string | null; attributes?: Record<string, unknown> },
    fn: (span: Span) => Promise<T>,
  ): Promise<T> {
    const span = this.startSpan(input);
    try {
      const value = await fn(span);
      this.endSpan(span);
      return value;
    } catch (error) {
      this.endSpan(span, { status: "error", error: error instanceof Error ? error.message : String(error) });
      throw error;
    }
  }

  onSpanEnd(listener: (event: TraceEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  list(limit = 100): Span[] {
    return this.spans.slice(-limit).reverse();
  }

  forTrace(traceId: string): Span[] {
    return this.spans.filter((span) => span.traceId === traceId);
  }

  summaries(limit = 20): TraceSummary[] {
    const grouped = new Map<string, Span[]>();
    for (const span of this.spans) {
      const bucket = grouped.get(span.traceId) ?? [];
      bucket.push(span);
      grouped.set(span.traceId, bucket);
    }
    return [...grouped.entries()]
      .map(([traceId, spans]) => {
        const sorted = [...spans].sort((a, b) => a.startMs - b.startMs);
        const root = sorted[0];
        const finished = spans.filter((span) => span.endMs !== null);
        const lastEnd = finished.length ? Math.max(...finished.map((span) => span.endMs!)) : root.startMs;
        return {
          traceId,
          rootSpan: root.name,
          spans: spans.length,
          durationMs: lastEnd - root.startMs,
          status: spans.some((span) => span.status === "error") ? "error" : "ok",
          startedAt: root.startMs,
          modules: [...new Set(spans.map((span) => span.module))],
        } satisfies TraceSummary;
      })
      .sort((a, b) => b.startedAt - a.startedAt)
      .slice(0, limit);
  }

  get openSpanCount(): number {
    return this.open.size;
  }

  get spanCount(): number {
    return this.spans.length;
  }

  clear(): void {
    this.spans = [];
    this.open.clear();
  }
}
