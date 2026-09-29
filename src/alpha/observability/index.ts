/**
 * Alpha Observability — the facade the rest of the stack records through.
 *
 * Every subsystem calls one of the `record*` methods below, which update
 * metrics, write a structured log line and close a span. Because the recording
 * is centralised, the workspace can show real numbers instead of a mock
 * dashboard: what appears there is what the modules actually did.
 */

import type { GenerationResult } from "../inference/engine";
import type { TrainingMetricPoint } from "../training/trainer";
import type { ToolExecutionRecord } from "../tools/registry";
import type { AlphaModelStage } from "../core/types";
import type { AgentRunResult } from "../agents/types";
import type { JobExecution } from "../automation/engine";
import type { RagAnswer } from "../rag/pipeline";
import type { MemoryRetrieval } from "../memory/store";
import { ALPHA_METRICS, AlphaMetricsRegistry, type MetricSnapshot } from "./metrics";
import { AlphaLogger, type LogEntry, type LogLevel } from "./logger";
import { AlphaTracer, type Span, type SpanKind, type TraceSummary } from "./tracer";

export type ObservabilitySnapshot = {
  metrics: MetricSnapshot[];
  traces: TraceSummary[];
  logs: LogEntry[];
  spans: number;
  openSpans: number;
  logCounts: Record<LogLevel, number>;
  errorCount: number;
  takenAt: number;
};

export class AlphaObservability {
  readonly tracer: AlphaTracer;
  readonly metrics: AlphaMetricsRegistry;
  readonly logger: AlphaLogger;
  private modelStage: AlphaModelStage = "untrained";

  constructor(options: { logLevel?: LogLevel; sampleWindow?: number; maxSpans?: number } = {}) {
    this.tracer = new AlphaTracer({ maxSpans: options.maxSpans });
    this.metrics = new AlphaMetricsRegistry({ sampleWindow: options.sampleWindow });
    this.logger = new AlphaLogger({ minLevel: options.logLevel });
  }

  /** Model provenance is a first-class metric, not a comment in the UI copy. */
  setModelStage(stage: AlphaModelStage, attributes: Record<string, unknown> = {}): void {
    this.modelStage = stage;
    this.metrics.setGauge("alpha.model.stage_ordinal", stageOrdinal(stage), "stage");
    this.logger.log("info", "model", `model stage is now ${stage}`, attributes);
  }

  get stage(): AlphaModelStage {
    return this.modelStage;
  }

  startSpan(input: {
    name: string;
    kind: SpanKind;
    module: string;
    traceId?: string;
    parentId?: string | null;
    attributes?: Record<string, unknown>;
  }): Span {
    return this.tracer.startSpan({
      ...input,
      attributes: { modelStage: this.modelStage, ...(input.attributes ?? {}) },
    });
  }

  endSpan(span: Span, outcome: Parameters<AlphaTracer["endSpan"]>[1] = {}): Span {
    return this.tracer.endSpan(span, outcome);
  }

  span<T>(
    input: Parameters<AlphaTracer["withSpan"]>[0],
    fn: (span: Span) => T,
  ): T {
    return this.tracer.withSpan(
      { ...input, attributes: { modelStage: this.modelStage, ...(input.attributes ?? {}) } },
      fn,
    );
  }

  spanAsync<T>(
    input: Parameters<AlphaTracer["withSpanAsync"]>[0],
    fn: (span: Span) => Promise<T>,
  ): Promise<T> {
    return this.tracer.withSpanAsync(
      { ...input, attributes: { modelStage: this.modelStage, ...(input.attributes ?? {}) } },
      fn,
    );
  }

  recordInference(result: GenerationResult, traceId: string): void {
    this.metrics.increment(ALPHA_METRICS.inferenceRequests);
    this.metrics.observe(ALPHA_METRICS.inferenceLatencyMs, result.latencyMs);
    this.metrics.increment(ALPHA_METRICS.inferenceTokensGenerated, result.generatedTokens);
    this.metrics.increment(ALPHA_METRICS.inferenceTokensPrompt, result.promptTokens);
    this.metrics.increment(`${ALPHA_METRICS.inferenceStopReason}.${result.stopReason}`);
    this.logger.log(
      "info",
      "inference",
      `generated ${result.generatedTokens} token(s) in ${result.latencyMs}ms (${result.stopReason})`,
      {
        modelStage: result.modelStage,
        tokensPerSecond: result.tokensPerSecond,
        promptTokens: result.promptTokens,
        warning: result.warning,
      },
      traceId,
    );
  }

  recordTrainingStep(point: TrainingMetricPoint, traceId?: string): void {
    this.metrics.increment(ALPHA_METRICS.trainingSteps);
    this.metrics.observe(ALPHA_METRICS.trainingLoss, point.loss, "nats");
    this.metrics.increment(ALPHA_METRICS.trainingTokens, point.tokensSeen);
    if (point.step % 10 === 0) {
      this.logger.log(
        "debug",
        "training",
        `step ${point.step}: loss ${point.loss.toFixed(4)}, lr ${point.learningRate.toExponential(2)}`,
        { gradNorm: point.gradNorm, perplexity: point.perplexity },
        traceId ?? null,
      );
    }
  }

  recordToolExecution(record: ToolExecutionRecord): void {
    this.metrics.increment(ALPHA_METRICS.toolExecutions);
    this.metrics.observe("alpha.tool.duration_ms", record.durationMs);
    if (!record.ok) {
      this.metrics.increment(ALPHA_METRICS.toolFailures);
      this.metrics.increment(ALPHA_METRICS.errors);
    }
    this.logger.log(
      record.ok ? "info" : "warn",
      "tools",
      `${record.tool} ${record.ok ? "succeeded" : "failed"} in ${record.durationMs}ms`,
      { actorId: record.actorId, error: record.error },
      record.traceId,
    );
  }

  recordRagIngestion(document: { id: string; title: string; chunks: number; tokens: number }): void {
    this.metrics.increment(ALPHA_METRICS.ragIngestions);
    this.logger.log("info", "rag", `ingested "${document.title}" into ${document.chunks} chunk(s)`, {
      documentId: document.id,
      tokens: document.tokens,
    });
  }

  recordRagRetrieval(answer: Pick<RagAnswer, "query" | "retrieved" | "contextTokens" | "sources">, traceId?: string): void {
    this.metrics.increment(ALPHA_METRICS.ragRetrievals);
    this.metrics.observe("alpha.rag.context_tokens", answer.contextTokens, "tokens");
    this.logger.log("info", "rag", `retrieved ${answer.retrieved} chunk(s) for a query`, {
      query: answer.query.slice(0, 120),
      sources: answer.sources.map((source) => source.title),
    }, traceId ?? null);
  }

  recordEmbedding(count: number, dimension: number): void {
    this.metrics.increment(ALPHA_METRICS.embeddingRequests, count);
    this.metrics.setGauge("alpha.embedding.dimension", dimension);
  }

  recordMemoryRetrieval(entries: MemoryRetrieval[], traceId?: string): void {
    this.metrics.observe("alpha.memory.retrieved", entries.length, "records");
    this.logger.log("debug", "memory", `retrieved ${entries.length} memory record(s)`, {
      scopes: entries.map((entry) => entry.record.scope),
      topRelevance: entries[0]?.relevance ?? null,
    }, traceId ?? null);
  }

  recordMemoryWrite(scope: string, key: string): void {
    this.metrics.increment(ALPHA_METRICS.memoryWrites);
    this.logger.log("info", "memory", `wrote ${scope} memory "${key}"`);
  }

  recordAgentRun(run: AgentRunResult): void {
    this.metrics.increment(ALPHA_METRICS.agentRuns);
    this.metrics.increment(ALPHA_METRICS.agentSteps, run.plan.steps.length);
    this.metrics.observe("alpha.agent.duration_ms", run.durationMs);
    if (run.status !== "completed") this.metrics.increment(ALPHA_METRICS.errors);
    this.logger.log(
      run.status === "completed" ? "info" : "warn",
      "agents",
      `agent run ${run.status} in ${run.durationMs}ms`,
      {
        goal: run.goal.slice(0, 120),
        steps: run.plan.steps.length,
        toolCalls: run.toolCalls.length,
        synthesis: run.synthesis.method,
        blocker: run.blocker,
      },
      run.traceId,
    );
  }

  recordWorkflowExecution(job: JobExecution): void {
    this.metrics.increment(ALPHA_METRICS.workflowExecutions);
    this.metrics.observe("alpha.workflow.duration_ms", job.durationMs);
    if (job.status === "failed") this.metrics.increment(ALPHA_METRICS.workflowFailures);
    this.logger.log(
      job.status === "succeeded" ? "info" : job.status === "failed" ? "error" : "warn",
      "automation",
      `workflow "${job.workflowName}" ${job.status}`,
      { attempts: job.attempts, errors: job.errors },
      job.traceId,
    );
  }

  recordPolicyDenial(actorId: string, permission: string, reason: string): void {
    this.metrics.increment(ALPHA_METRICS.policyDenials);
    this.logger.log("warn", "security", `denied "${permission}" for ${actorId}`, { reason });
  }

  recordRateLimit(actorId: string, kind: string): void {
    this.metrics.increment(ALPHA_METRICS.rateLimitHits);
    this.logger.log("warn", "security", `rate limit hit for ${actorId} on ${kind}`);
  }

  recordError(module: string, message: string, data?: Record<string, unknown>): void {
    this.metrics.increment(ALPHA_METRICS.errors);
    this.logger.log("error", module, message, data);
  }

  snapshot(options: { logs?: number; traces?: number; spans?: number } = {}): ObservabilitySnapshot {
    return {
      metrics: this.metrics.snapshot(),
      traces: this.tracer.summaries(options.traces ?? 20),
      logs: this.logger.list({ limit: options.logs ?? 60 }),
      spans: this.tracer.spanCount,
      openSpans: this.tracer.openSpanCount,
      logCounts: this.logger.counts(),
      errorCount: this.metrics.value(ALPHA_METRICS.errors),
      takenAt: Date.now(),
    };
  }
}

function stageOrdinal(stage: AlphaModelStage): number {
  switch (stage) {
    case "architecture":
      return 0;
    case "untrained":
      return 1;
    case "trained":
      return 2;
    case "fine-tuned":
      return 3;
    case "production":
      return 4;
  }
}

export { ALPHA_METRICS } from "./metrics";
export type { MetricSnapshot, HistogramSummary } from "./metrics";
export type { LogEntry, LogLevel } from "./logger";
export type { Span, SpanKind, TraceSummary } from "./tracer";
