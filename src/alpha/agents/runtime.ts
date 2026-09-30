/**
 * Alpha Agents — the runtime.
 *
 * The loop is explicit and bounded:
 *   plan -> (select tool -> authorize -> execute -> verify -> record)* -> synthesise
 *
 * Everything an agent does passes through Alpha's own inference engine, tool
 * registry, policy engine and sandbox. Every run produces a record: the plan,
 * each step with its duration and verification, every tool execution with its
 * arguments and outcome, and how the final answer was produced.
 *
 * The final answer is never invented. If the model is available it writes the
 * answer from the collected tool results; if not, Alpha returns a structured
 * summary of those results and labels it `structured` so nobody mistakes it for
 * generated text.
 */

import { alphaId, newTraceId } from "../core/types";
import { describeError } from "../core/errors";
import type { AlphaInferenceEngine } from "../inference/engine";
import type { AlphaMemoryStore } from "../memory/store";
import type { AlphaAuditLog } from "../security/audit";
import type { AlphaPolicyEngine } from "../security/policy";
import type { AlphaRateLimiter } from "../security/rate-limit";
import { AlphaSandbox, DEFAULT_SANDBOX, type SandboxSpec } from "../security/sandbox";
import type { AlphaToolRegistry, ToolExecutionRecord } from "../tools/registry";
import { planFromCapabilities, planWithModel } from "./planner";
import { verifyAgentOutcome, reconcileOutcome, type SandboxReport } from "./verify";
import type {
  AgentEvent,
  AgentPlan,
  AgentPlanStep,
  AgentRunResult,
  AgentRunStatus,
  AgentStepRecord,
  AgentSynthesis,
  AgentVerification,
  AgentTask,
} from "./types";

export type AgentRuntimeOptions = {
  registry: AlphaToolRegistry;
  policy: AlphaPolicyEngine;
  inference?: AlphaInferenceEngine;
  memory?: AlphaMemoryStore;
  audit?: AlphaAuditLog;
  rateLimiter?: AlphaRateLimiter;
  onEvent?: (event: AgentEvent) => void;
};

export type AgentRunRequest = {
  goal: string;
  /** Actor the agent runs as — normally an `agent`-role actor id. */
  actorId: string;
  sessionId?: string | null;
  context?: string;
  requiredCapabilities?: string[];
  maxSteps?: number;
  /** Allow the planner to ask Alpha's model for a plan when it is trained. */
  preferModelPlanner?: boolean;
  /**
   * Stop the run at the next step boundary. A handle for the caller to hold,
   * so a UI can abort a long run without waiting for a natural stopping point.
   */
  cancellation?: AgentCancellation | null;
};

/** Cooperative cancellation for an agent run, checked at every step boundary. */
export type AgentCancellation = {
  readonly cancelled: boolean;
  readonly reason: string | null;
  cancel(reason?: string): void;
};

export function createAgentCancellation(): AgentCancellation {
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

export class AlphaAgentRuntime {
  private readonly registry: AlphaToolRegistry;
  private readonly policy: AlphaPolicyEngine;
  private readonly inference: AlphaInferenceEngine | undefined;
  private readonly memory: AlphaMemoryStore | undefined;
  private readonly audit: AlphaAuditLog | undefined;
  private readonly rateLimiter: AlphaRateLimiter | undefined;
  private readonly onEvent: ((event: AgentEvent) => void) | undefined;
  private runs: AgentRunResult[] = [];

  constructor(options: AgentRuntimeOptions) {
    this.registry = options.registry;
    this.policy = options.policy;
    this.inference = options.inference;
    this.memory = options.memory;
    this.audit = options.audit;
    this.rateLimiter = options.rateLimiter;
    this.onEvent = options.onEvent;
  }

  /** Build the sandbox for a run from the actor's registered scope. */
  private sandboxFor(actorId: string, request: AgentRunRequest): AlphaSandbox {
    const scope = this.policy.scopeOf(actorId);
    const allowedTools = scope?.allowedTools?.length
      ? scope.allowedTools
      : this.registry.discover({ actorId }).map((tool) => tool.name);
    const spec: SandboxSpec = {
      agentId: actorId,
      allowedTools,
      maxSteps: request.maxSteps ?? scope?.maxSteps ?? DEFAULT_SANDBOX.maxSteps,
      maxToolCalls: DEFAULT_SANDBOX.maxToolCalls,
      maxIdenticalCalls: DEFAULT_SANDBOX.maxIdenticalCalls,
      maxGeneratedTokens: DEFAULT_SANDBOX.maxGeneratedTokens,
      maxDurationMs: DEFAULT_SANDBOX.maxDurationMs,
      allowNetwork: false,
      allowFileSystem: false,
    };
    return new AlphaSandbox(spec);
  }

  private plan(task: AgentTask, actorId: string, request: AgentRunRequest): AgentPlan {
    if (request.preferModelPlanner && this.inference) {
      const modelPlan = planWithModel(task, {
        inference: this.inference,
        registry: this.registry,
        actorId,
        maxSteps: request.maxSteps,
      });
      if (modelPlan) return modelPlan;
    }
    return planFromCapabilities(task, this.registry, actorId, { maxSteps: request.maxSteps });
  }

  /**
   * Run an agent task. Yields events as they happen and returns the final run
   * record, so a UI can stream progress and a test can assert on the record.
   */
  async *run(request: AgentRunRequest): AsyncGenerator<AgentEvent, AgentRunResult, void> {
    const startedAt = Date.now();
    const traceId = newTraceId();
    const task: AgentTask = {
      id: alphaId("task"),
      goal: request.goal,
      context: request.context,
      requiredCapabilities: request.requiredCapabilities,
      maxSteps: request.maxSteps,
      sessionId: request.sessionId ?? null,
      createdAt: startedAt,
    };
    const sandbox = this.sandboxFor(request.actorId, request);
    const history: AgentStepRecord[] = [];
    const toolCalls: ToolExecutionRecord[] = [];
    let tokensGenerated = 0;
    let blocker: string | null = null;

    yield { type: "start", task, traceId };
    this.emit({ type: "start", task, traceId });

    if (this.rateLimiter) {
      const decision = this.rateLimiter.consume(request.actorId, "agent");
      if (!decision.allowed) {
        blocker = `agent rate limit reached; retry in ${decision.retryAfterMs}ms`;
        yield { type: "blocked", reason: blocker };
        this.emit({ type: "blocked", reason: blocker });
      }
    }

    let plan: AgentPlan = { taskId: task.id, createdAt: Date.now(), plannerSource: "single-step", rationale: "", steps: [] };
    if (!blocker) {
      const planStarted = Date.now();
      plan = this.plan(task, request.actorId, request);
      const record: AgentStepRecord = {
        index: -1,
        phase: "plan",
        at: planStarted,
        durationMs: Date.now() - planStarted,
        detail: plan.rationale,
        toolName: null,
        ok: plan.steps.length > 0,
      };
      history.push(record);
      yield { type: "plan", plan };
      this.emit({ type: "plan", plan });
    }

    for (const step of plan.steps) {
      if (blocker) break;
      if (request.cancellation?.cancelled) {
        sandbox.cancel(request.cancellation.reason ?? "cancelled by caller");
        blocker = request.cancellation.reason ?? "the run was cancelled";
        history.push({
          index: step.index,
          phase: "blocked",
          at: Date.now(),
          durationMs: 0,
          detail: blocker,
          toolName: null,
          ok: false,
        });
        yield { type: "blocked", reason: blocker };
        this.emit({ type: "blocked", reason: blocker });
        break;
      }
      if (sandbox.exhausted) {
        blocker = `sandbox stopped the run: ${sandbox.spec.maxSteps} step budget or time budget reached`;
        yield { type: "blocked", reason: blocker };
        this.emit({ type: "blocked", reason: blocker });
        break;
      }
      const stepStarted = Date.now();
      const descriptor = step.toolName ? this.registry.describe().find((tool) => tool.name === step.toolName) : null;
      if (!step.toolName || !descriptor) {
        step.status = "skipped";
        step.error = "no tool bound to this step";
        history.push({
          index: step.index,
          phase: "execute",
          at: stepStarted,
          durationMs: 0,
          detail: step.description,
          toolName: null,
          ok: false,
        });
        yield { type: "step", step, history: history[history.length - 1] };
        continue;
      }

      try {
        sandbox.enterTool(
          descriptor.name,
          {
            networked: descriptor.characteristics.networked,
            fileSystem: descriptor.characteristics.fileSystem,
          },
          buildToolArguments(descriptor.inputSchema, task),
        );
      } catch (error) {
        const described = describeError(error);
        step.status = "failed";
        step.error = described.message;
        blocker = described.message;
        history.push({
          index: step.index,
          phase: "blocked",
          at: stepStarted,
          durationMs: Date.now() - stepStarted,
          detail: described.message,
          toolName: descriptor.name,
          ok: false,
        });
        yield { type: "step", step, history: history[history.length - 1] };
        yield { type: "blocked", reason: described.message };
        this.emit({ type: "blocked", reason: described.message });
        break;
      }

      step.status = "running";
      const args = buildToolArguments(descriptor.inputSchema, task);
      const execution = await this.registry.execute(descriptor.name, args, {
        actorId: request.actorId,
        traceId,
        meta: { taskId: task.id, goal: task.goal },
      });
      toolCalls.push(execution.record);
      yield { type: "tool", record: execution.record };
      this.emit({ type: "tool", record: execution.record });

      const verification = execution.ok
        ? this.registry.verify(descriptor.name, execution.output, args)
        : { ok: false, reason: execution.error?.message ?? "tool execution failed" };
      step.verification = verification;
      step.result = execution.output;
      step.error = execution.ok ? null : execution.error?.message ?? "tool execution failed";
      step.status = execution.ok && verification.ok ? "done" : "failed";
      step.durationMs = Date.now() - stepStarted;

      const stepRecord: AgentStepRecord = {
        index: step.index,
        phase: "verify",
        at: stepStarted,
        durationMs: step.durationMs,
        detail: `${execution.ok ? "executed" : "failed"}: ${descriptor.name} — ${verification.reason}`,
        toolName: descriptor.name,
        ok: step.status === "done",
      };
      history.push(stepRecord);
      yield { type: "verify", stepId: step.id, verification };
      yield { type: "step", step, history: stepRecord };
      this.emit({ type: "verify", stepId: step.id, verification });
    }

    // Synthesise the final answer from what the tools actually returned.
    const succeeded = plan.steps.filter((step) => step.status === "done");
    // Verify the objective *before* synthesising, so the final answer is built
    // on an honest account of what happened rather than an assumption.
    const verification = verifyAgentOutcome({
      plan,
      toolCalls,
      status: null,
      sandbox: sandbox.report() as SandboxReport,
      blocker,
    });
    yield { type: "verified", verification };
    this.emit({ type: "verified", verification });

    const synthesis = this.synthesise(task, plan, sandbox, verification);
    tokensGenerated += synthesis.tokensGenerated;
    if (this.inference) {
      try {
        sandbox.accountTokens(synthesis.tokensGenerated);
      } catch {
        blocker = blocker ?? "token budget exhausted during synthesis";
      }
    }
    yield { type: "synthesise", synthesis };
    this.emit({ type: "synthesise", synthesis });

    let status: AgentRunStatus;
    if (blocker) {
      status = succeeded.length > 0 ? "completed" : "blocked";
      if (blocker.includes("step budget") || blocker.includes("token budget")) {
        status = succeeded.length > 0 ? "completed" : "step-budget-exhausted";
      }
    } else if (succeeded.length === 0 && plan.steps.length > 0) {
      status = "failed";
    } else {
      status = "completed";
    }

    const finishedAt = Date.now();
    const result: AgentRunResult = {
      id: alphaId("run"),
      taskId: task.id,
      goal: task.goal,
      actorId: request.actorId,
      status,
      modelName: this.inference ? this.inference.model.config.name : null,
      modelVersion: this.inference ? this.inference.model.config.version : null,
      plan,
      history,
      toolCalls,
      synthesis,
      // The outcome is derived from what actually ran, never from `status`
      // alone, so a run that exited cleanly but achieved nothing says so.
      verification: {
        ...verification,
        outcome: reconcileOutcome(verification, status, blocker, sandbox.report() as SandboxReport),
      },
      startedAt,
      finishedAt,
      durationMs: finishedAt - startedAt,
      traceId,
      modelStage: this.inference ? this.inference.stage : null,
      sandbox: sandbox.report(),
      blocker,
      errors: plan.steps.filter((step) => step.error).map((step) => step.error as string),
    };

    this.runs.push(result);
    if (this.runs.length > 100) this.runs = this.runs.slice(-50);

    if (this.memory && request.sessionId) {
      try {
        this.memory.write({
          scope: "session",
          sessionId: request.sessionId,
          ownerId: request.actorId,
          key: `agent.last_run`,
          content: `Goal: ${task.goal} — ${status} via ${succeeded.map((s) => s.toolName).filter(Boolean).join(", ") || "no tools"}.`,
          source: "agent",
          provenance: { origin: "agent-run", referenceId: result.id, recordedBy: request.actorId },
          tags: ["agent", "run"],
          importance: 0.4,
        });
      } catch (error) {
        this.audit?.append({
          actor: request.actorId,
          module: "agents",
          action: "memory.write",
          resource: "session",
          decision: "error",
          reason: describeError(error).message,
          traceId,
        });
      }
    }

    this.audit?.append({
      actor: request.actorId,
      module: "agents",
      action: "agent.run",
      resource: task.id,
      decision: status === "completed" ? "allow" : "deny",
      reason: `${status} in ${result.durationMs}ms with ${toolCalls.length} tool call(s)`,
      traceId,
      data: { steps: plan.steps.length, succeeded: succeeded.length, blocker },
    });

    yield { type: "done", result };
    this.emit({ type: "done", result });
    return result;
  }

  /**
   * Produce the final answer. With an inference engine the model writes it from
   * the tool results; without one, Alpha returns a structured summary and says
   * so.
   */
  private synthesise(
    task: AgentTask,
    plan: AgentPlan,
    sandbox: AlphaSandbox,
    verification?: AgentVerification,
  ): AgentSynthesis {
    const usable = plan.steps.filter((step) => step.status === "done" && step.result !== null);
    const evidence = usable
      .map((step) => `[${step.toolName}] ${truncate(JSON.stringify(step.result), 600)}`)
      .join("\n");
    // The outcome is stated in the prompt so the model's answer cannot imply
    // more than the evidence supports.
    const outcomeLine = verification
      ? `Alpha's own verification of this run: ${verification.outcome} — ${verification.reason}`
      : null;

    if (!this.inference) {
      return {
        method: "structured",
        text: usable.length
          ? `No inference engine is attached to this runtime, so Alpha is reporting tool results verbatim instead of generating an answer:\n${evidence}`
          : "No tool produced a usable result, so there is nothing to report.",
        modelStage: null,
        warning: "Structured summary — no model was invoked.",
        tokensGenerated: 0,
      };
    }

    const prompt = [
      "Instruction: answer the goal using only the tool results below. If they do not contain the answer, say so.",
      ...(outcomeLine ? [outcomeLine] : []),
      "<<<RESULTS",
      evidence || "(no tool produced a result)",
      "RESULTS",
      `Goal: ${task.goal}`,
      "Answer:",
    ].join("\n");
    const generation = this.inference.generate(prompt, { maxNewTokens: 80, temperature: 0.4 });
    const remaining = sandbox.spec.maxGeneratedTokens - sandbox.state.generatedTokens;
    return {
      method: "model",
      text: generation.text,
      modelStage: generation.modelStage,
      warning:
        generation.warning ??
        (remaining < 0 ? "Generation exceeded the sandbox token budget." : null),
      tokensGenerated: generation.generatedTokens,
    };
  }

  private emit(event: AgentEvent): void {
    this.onEvent?.(event);
  }

  recentRuns(limit = 10): AgentRunResult[] {
    return this.runs.slice(-limit).reverse();
  }

  stats(): { runs: number; completed: number; failed: number; averageSteps: number; averageDurationMs: number } {
    const runs = this.runs;
    const completed = runs.filter((run) => run.status === "completed").length;
    const failed = runs.filter((run) => run.status === "failed" || run.status === "blocked").length;
    return {
      runs: runs.length,
      completed,
      failed,
      averageSteps: runs.length ? Number((runs.reduce((sum, run) => sum + run.plan.steps.length, 0) / runs.length).toFixed(2)) : 0,
      averageDurationMs: runs.length ? Number((runs.reduce((sum, run) => sum + run.durationMs, 0) / runs.length).toFixed(2)) : 0,
    };
  }
}

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

/**
 * Pull the arithmetic part out of a goal, e.g. "calculate 12 * 4" -> "12 * 4".
 * Returns the goal unchanged when no expression is present, so a tool that
 * cannot be satisfied fails validation visibly instead of silently succeeding.
 */
export function extractArithmeticExpression(goal: string): string {
  const runs = goal.match(/[0-9+\-*/%^(). ]+/g) ?? [];
  const candidates = runs
    .map((run) => run.trim())
    .filter((run) => /[0-9]/.test(run) && /[+\-*/%^]/.test(run))
    .sort((a, b) => b.length - a.length);
  return candidates[0] ?? goal;
}

/**
 * Build plausible arguments for a tool from the task text. Only string fields
 * are filled, and only from the goal — Alpha never fabricates numbers or
 * booleans, and a step whose arguments cannot be satisfied is expected to fail
 * validation visibly.
 */
export function buildToolArguments(
  schema: { properties?: Record<string, { type?: string | string[] }>; required?: string[] },
  task: AgentTask,
): Record<string, unknown> {
  const args: Record<string, unknown> = {};
  const properties = schema.properties ?? {};
  for (const [name, property] of Object.entries(properties)) {
    const type = Array.isArray(property.type) ? property.type[0] : property.type;
    if (type !== "string") continue;
    // Arithmetic inputs get the arithmetic part of the goal; everything else
    // receives the goal verbatim and lets the tool decide what to do with it.
    args[name] = name === "expression" ? extractArithmeticExpression(task.goal) : task.goal;
  }
  // Explicit, conservative defaults for the non-string fields Alpha defines.
  if (properties.topK) args.topK = 3;
  if (properties.confirm) args.confirm = false;
  if (properties.scope) args.scope = "session";
  if (properties.sessionId && task.sessionId) args.sessionId = task.sessionId;
  return args;
}

/** Convenience wrapper used by the workspace for one-shot agent runs. */
export async function runAgentTask(
  runtime: AlphaAgentRuntime,
  request: AgentRunRequest,
): Promise<AgentRunResult> {
  const iterator = runtime.run(request);
  let next = await iterator.next();
  let result: AgentRunResult | null = null;
  while (!next.done) {
    if (next.value.type === "done") result = next.value.result;
    next = await iterator.next();
  }
  if (!result) throw new Error("[alpha:agents] agent run produced no result");
  return result;
}
