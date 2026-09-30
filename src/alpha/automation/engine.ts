/**
 * Alpha Automation — workflows, triggers, actions, a job queue and history.
 *
 * A workflow is data, not a script:
 *   trigger (manual | interval | event) -> conditions -> ordered actions
 *
 * Actions are either a tool call, an agent run, a memory write or a log entry.
 * Every execution is queued, retried with exponential backoff, and written to
 * history with its duration, attempts and outcome. Scheduled workflows only
 * tick while `start()` has been called — the engine never runs a timer the host
 * did not ask for.
 */

import { alphaId, newTraceId } from "../core/types";
import { describeError } from "../core/errors";
import type { AlphaAuditLog } from "../security/audit";
import type { AlphaPolicyEngine } from "../security/policy";
import type { AlphaRateLimiter } from "../security/rate-limit";
import type { AlphaAgentRuntime } from "../agents/runtime";
import type { AgentRunResult } from "../agents/types";
import type { AlphaMemoryStore } from "../memory/store";
import type { AlphaToolRegistry } from "../tools/registry";

export type WorkflowTrigger =
  | { kind: "manual" }
  | { kind: "interval"; intervalMs: number }
  | { kind: "event"; eventName: string };

export type ConditionOperator = "equals" | "not-equals" | "greater-than" | "less-than" | "contains" | "exists" | "matches";

export type WorkflowCondition = {
  /** Dot path into the trigger payload, e.g. "document.title". */
  path: string;
  operator: ConditionOperator;
  value?: unknown;
};

export type WorkflowAction =
  | { kind: "tool"; toolName: string; args: Record<string, unknown>; actorId?: string }
  | { kind: "agent"; goal: string; actorId?: string; maxSteps?: number }
  | { kind: "memory"; key: string; content: string; scope: "conversation" | "session" | "long-term"; approved?: boolean }
  | { kind: "log"; message: string };

export type AlphaWorkflow = {
  id: string;
  name: string;
  description: string;
  trigger: WorkflowTrigger;
  conditions: WorkflowCondition[];
  actions: WorkflowAction[];
  enabled: boolean;
  createdAt: number;
  updatedAt: number;
  /** Actor that owns the workflow's permissions. */
  actorId: string;
};

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "skipped" | "retrying";

export type ActionOutcome = {
  index: number;
  action: WorkflowAction["kind"];
  ok: boolean;
  detail: string;
  durationMs: number;
  output: unknown;
};

export type JobExecution = {
  id: string;
  workflowId: string;
  workflowName: string;
  trigger: string;
  status: JobStatus;
  attempts: number;
  maxAttempts: number;
  payload: Record<string, unknown>;
  outcomes: ActionOutcome[];
  errors: string[];
  queuedAt: number;
  scheduledFor: number;
  startedAt: number | null;
  finishedAt: number | null;
  durationMs: number;
  traceId: string;
  /** Conditions that made the job skip, when applicable. */
  skippedReason: string | null;
};

export type WorkflowRunSummary = {
  job: JobExecution;
  retried: boolean;
};

export type AutomationEngineOptions = {
  registry: AlphaToolRegistry;
  policy: AlphaPolicyEngine;
  audit?: AlphaAuditLog;
  rateLimiter?: AlphaRateLimiter;
  agents?: AlphaAgentRuntime;
  memory?: AlphaMemoryStore;
  maxAttempts?: number;
  baseRetryDelayMs?: number;
  /** Cap on retained execution history. */
  historyLimit?: number;
};

function getPath(payload: Record<string, unknown>, path: string): unknown {
  return path.split(".").reduce<unknown>((acc, key) => {
    if (acc && typeof acc === "object" && key in (acc as Record<string, unknown>)) {
      return (acc as Record<string, unknown>)[key];
    }
    return undefined;
  }, payload);
}

/** Evaluate one condition against a payload. */
export function evaluateCondition(condition: WorkflowCondition, payload: Record<string, unknown>): boolean {
  const actual = getPath(payload, condition.path);
  switch (condition.operator) {
    case "exists":
      return actual !== undefined && actual !== null;
    case "equals":
      return actual === condition.value;
    case "not-equals":
      return actual !== condition.value;
    case "greater-than":
      return typeof actual === "number" && typeof condition.value === "number" && actual > condition.value;
    case "less-than":
      return typeof actual === "number" && typeof condition.value === "number" && actual < condition.value;
    case "contains":
      return typeof actual === "string" && typeof condition.value === "string" && actual.includes(condition.value);
    case "matches":
      return typeof actual === "string" && typeof condition.value === "string" && new RegExp(condition.value).test(actual);
  }
}

export class AlphaAutomationEngine {
  private workflows = new Map<string, AlphaWorkflow>();
  private queue: JobExecution[] = [];
  private history: JobExecution[] = [];
  private timers = new Map<string, ReturnType<typeof setInterval>>();
  private readonly options: AutomationEngineOptions;
  private readonly maxAttempts: number;
  private readonly baseRetryDelayMs: number;
  private readonly historyLimit: number;

  constructor(options: AutomationEngineOptions) {
    this.options = options;
    this.maxAttempts = options.maxAttempts ?? 3;
    this.baseRetryDelayMs = options.baseRetryDelayMs ?? 250;
    this.historyLimit = options.historyLimit ?? 200;
  }

  registerWorkflow(input: Omit<AlphaWorkflow, "id" | "createdAt" | "updatedAt"> & { id?: string }): AlphaWorkflow {
    const workflow: AlphaWorkflow = {
      id: input.id ?? alphaId("wf"),
      name: input.name,
      description: input.description,
      trigger: input.trigger,
      conditions: input.conditions,
      actions: input.actions,
      enabled: input.enabled,
      actorId: input.actorId,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    this.workflows.set(workflow.id, workflow);
    return workflow;
  }

  listWorkflows(): AlphaWorkflow[] {
    return [...this.workflows.values()].sort((a, b) => b.updatedAt - a.updatedAt);
  }

  getWorkflow(id: string): AlphaWorkflow | null {
    return this.workflows.get(id) ?? null;
  }

  setEnabled(id: string, enabled: boolean): AlphaWorkflow | null {
    const workflow = this.workflows.get(id);
    if (!workflow) return null;
    workflow.enabled = enabled;
    workflow.updatedAt = Date.now();
    if (!enabled) this.stop(id);
    return workflow;
  }

  /**
   * Enqueue a job. Conditions are evaluated here, so a job that does not meet
   * its conditions is recorded as `skipped` rather than silently dropped.
   */
  enqueue(workflowId: string, payload: Record<string, unknown> = {}, trigger = "manual"): JobExecution {
    const workflow = this.workflows.get(workflowId);
    if (!workflow) {
      throw new Error(`[alpha:automation] workflow "${workflowId}" is not registered`);
    }
    const traceId = newTraceId();
    const failedConditions = workflow.conditions.filter((condition) => !evaluateCondition(condition, payload));
    const job: JobExecution = {
      id: alphaId("job"),
      workflowId: workflow.id,
      workflowName: workflow.name,
      trigger,
      status: failedConditions.length > 0 ? "skipped" : workflow.enabled ? "queued" : "skipped",
      attempts: 0,
      maxAttempts: this.maxAttempts,
      payload,
      outcomes: [],
      errors: [],
      queuedAt: Date.now(),
      scheduledFor: Date.now(),
      startedAt: null,
      finishedAt: failedConditions.length > 0 ? Date.now() : null,
      durationMs: 0,
      traceId,
      skippedReason:
        failedConditions.length > 0
          ? `conditions not met: ${failedConditions.map((c) => `${c.path} ${c.operator}`).join(", ")}`
          : workflow.enabled
            ? null
            : "workflow is disabled",
    };
    if (job.status === "queued") this.queue.push(job);
    this.record(job);
    return job;
  }

  /**
   * Run every queued job now, including retries.
   * A job that fails is re-queued until it exhausts `maxAttempts`, so a single
   * drain call reports the final outcome rather than an intermediate state.
   */
  async drain(): Promise<JobExecution[]> {
    const processed: JobExecution[] = [];
    const guardLimit = Math.max(1, this.historyLimit) * (this.maxAttempts + 1);
    let guard = 0;
    while (this.queue.length > 0 && guard++ < guardLimit) {
      const job = this.queue.shift()!;
      const waitMs = Math.min(Math.max(0, job.scheduledFor - Date.now()), 5_000);
      if (waitMs > 0) await sleep(waitMs);
      const finished = await this.executeJob(job);
      processed.push(finished);
      if (finished.status === "retrying") this.queue.push(finished);
    }
    return processed;
  }

  /** Convenience: enqueue, drain (with retries) and return the final job state. */
  async run(workflowId: string, payload: Record<string, unknown> = {}, trigger = "manual"): Promise<JobExecution> {
    const job = this.enqueue(workflowId, payload, trigger);
    if (job.status !== "queued") return job;
    await this.drain();
    return job;
  }

  private async executeJob(job: JobExecution): Promise<JobExecution> {
    const workflow = this.workflows.get(job.workflowId);
    if (!workflow) {
      job.status = "failed";
      job.errors.push("workflow disappeared before execution");
      job.finishedAt = Date.now();
      this.record(job);
      return job;
    }
    if (this.options.rateLimiter) {
      const decision = this.options.rateLimiter.consume(workflow.actorId, "workflow");
      if (!decision.allowed) {
        job.status = "retrying";
        job.errors.push(`rate limited; retry after ${decision.retryAfterMs}ms`);
        job.scheduledFor = Date.now() + decision.retryAfterMs;
        this.record(job);
        return job;
      }
    }

    job.attempts++;
    job.status = "running";
    job.startedAt = Date.now();
    const started = Date.now();

    for (const [index, action] of workflow.actions.entries()) {
      const actionStarted = Date.now();
      try {
        const output = await this.runAction(action, workflow, job);
        job.outcomes.push({
          index,
          action: action.kind,
          ok: true,
          detail: describeAction(action),
          durationMs: Date.now() - actionStarted,
          output,
        });
      } catch (error) {
        const described = describeError(error);
        job.outcomes.push({
          index,
          action: action.kind,
          ok: false,
          detail: describeAction(action),
          durationMs: Date.now() - actionStarted,
          output: null,
        });
        job.errors.push(`${described.code}: ${described.message}`);
        this.options.audit?.append({
          actor: workflow.actorId,
          module: "automation",
          action: "workflow.action",
          resource: `${workflow.name}#${index}`,
          decision: "error",
          reason: described.message,
          traceId: job.traceId,
        });
      }
    }

    const failures = job.outcomes.filter((outcome) => !outcome.ok).length;
    job.durationMs = Date.now() - started;
    if (failures === 0) {
      job.status = "succeeded";
      job.finishedAt = Date.now();
    } else if (job.attempts < job.maxAttempts) {
      job.status = "retrying";
      job.scheduledFor = Date.now() + this.baseRetryDelayMs * 2 ** (job.attempts - 1);
      job.finishedAt = null;
    } else {
      job.status = "failed";
      job.finishedAt = Date.now();
    }

    this.record(job);
    this.options.audit?.append({
      actor: workflow.actorId,
      module: "automation",
      action: "workflow.run",
      resource: workflow.name,
      decision: job.status === "succeeded" ? "allow" : "error",
      reason: `${job.status} after ${job.attempts} attempt(s), ${failures} failed action(s)`,
      traceId: job.traceId,
    });
    return job;
  }

  private async runAction(
    action: WorkflowAction,
    workflow: AlphaWorkflow,
    job: JobExecution,
  ): Promise<unknown> {
    switch (action.kind) {
      case "tool": {
        const actorId = action.actorId ?? workflow.actorId;
        const result = await this.options.registry.execute(action.toolName, action.args, {
          actorId,
          traceId: job.traceId,
          meta: { workflowId: workflow.id, jobId: job.id },
        });
        if (!result.ok) {
          throw new Error(result.error?.message ?? `tool "${action.toolName}" failed`);
        }
        return result.output;
      }
      case "agent": {
        if (!this.options.agents) {
          throw new Error("no agent runtime is configured for this automation engine");
        }
        const iterator = this.options.agents.run({
          goal: action.goal,
          actorId: action.actorId ?? workflow.actorId,
          maxSteps: action.maxSteps,
        });
        let next = await iterator.next();
        let runResult: AgentRunResult | null = null;
        while (!next.done) {
          if (next.value.type === "done") runResult = next.value.result;
          next = await iterator.next();
        }
        return runResult
          ? { status: runResult.status, synthesis: runResult.synthesis.text }
          : { status: "no-result" };
      }
      case "memory": {
        if (!this.options.memory) {
          throw new Error("no memory store is configured for this automation engine");
        }
        const record = this.options.memory.write({
          scope: action.scope,
          key: action.key,
          content: action.content,
          ownerId: workflow.actorId,
          approved: action.scope === "long-term" ? action.approved === true : true,
          source: "system",
          sessionId: action.scope === "long-term" ? null : `workflow:${job.id}`,
          provenance: { origin: "workflow", referenceId: job.id, recordedBy: workflow.actorId },
        });
        return { id: record.id, scope: record.scope };
      }
      case "log":
        return { message: action.message, at: Date.now() };
    }
  }

  /** Start interval triggers. Idempotent; the host owns the lifecycle. */
  start(): number {
    let scheduled = 0;
    for (const workflow of this.workflows.values()) {
      if (!workflow.enabled || workflow.trigger.kind !== "interval") continue;
      this.startWorkflowTimer(workflow);
      scheduled++;
    }
    return scheduled;
  }

  private startWorkflowTimer(workflow: AlphaWorkflow): void {
    if (workflow.trigger.kind !== "interval") return;
    if (this.timers.has(workflow.id)) return;
    const intervalMs = Math.max(1000, workflow.trigger.intervalMs);
    const timer = setInterval(() => {
      const job = this.enqueue(workflow.id, { scheduledAt: Date.now() }, "interval");
      if (job.status === "queued") void this.drain();
    }, intervalMs);
    this.timers.set(workflow.id, timer);
  }

  stop(workflowId?: string): void {
    if (workflowId) {
      const timer = this.timers.get(workflowId);
      if (timer) clearInterval(timer);
      this.timers.delete(workflowId);
      return;
    }
    for (const timer of this.timers.values()) clearInterval(timer);
    this.timers.clear();
  }

  get running(): number {
    return this.timers.size;
  }

  pendingJobs(): JobExecution[] {
    return [...this.queue];
  }

  executionHistory(limit = 25): JobExecution[] {
    return this.history.slice(-limit).reverse();
  }

  stats(): {
    workflows: number;
    enabled: number;
    scheduled: number;
    executions: number;
    succeeded: number;
    failed: number;
    skipped: number;
    averageDurationMs: number;
  } {
    const executions = this.history.length;
    const succeeded = this.history.filter((job) => job.status === "succeeded").length;
    const failed = this.history.filter((job) => job.status === "failed").length;
    const skipped = this.history.filter((job) => job.status === "skipped").length;
    return {
      workflows: this.workflows.size,
      enabled: [...this.workflows.values()].filter((workflow) => workflow.enabled).length,
      scheduled: this.timers.size,
      executions,
      succeeded,
      failed,
      skipped,
      averageDurationMs:
        executions === 0
          ? 0
          : Number(
              (
                this.history.filter((job) => job.finishedAt).reduce((sum, job) => sum + job.durationMs, 0) /
                Math.max(1, this.history.filter((job) => job.finishedAt).length)
              ).toFixed(2),
            ),
    };
  }

  private record(job: JobExecution): void {
    const existing = this.history.findIndex((entry) => entry.id === job.id);
    if (existing >= 0) this.history[existing] = job;
    else this.history.push(job);
    if (this.history.length > this.historyLimit) this.history = this.history.slice(-this.historyLimit);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function describeAction(action: WorkflowAction): string {
  switch (action.kind) {
    case "tool":
      return `tool ${action.toolName}`;
    case "agent":
      return `agent goal "${action.goal.slice(0, 40)}"`;
    case "memory":
      return `memory write ${action.key} (${action.scope})`;
    case "log":
      return `log`;
  }
}
