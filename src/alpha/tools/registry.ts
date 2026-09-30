/**
 * Alpha Tools — registry, discovery and execution.
 *
 * A tool is a declaration plus a handler:
 *   name, description, module, JSON Schema, required permission, source, handler
 *
 * Execution is a fixed sequence with no shortcuts:
 *   rate limit -> authorization (permission + agent scope + approval)
 *   -> argument validation -> handler -> result normalisation -> audit record
 *
 * `execute` never throws for an expected failure case: a failed tool call comes
 * back as `{ ok: false, error }` so an agent can react to it. Authorization
 * failures do throw `AlphaPermissionError`, because those are bugs in the
 * orchestration above, not values to reason about.
 */

import { alphaId } from "../core/types";
import { AlphaToolError, describeError } from "../core/errors";
import type { AlphaPermission } from "../security/policy";
import { AlphaPolicyEngine } from "../security/policy";
import { AlphaRateLimiter } from "../security/rate-limit";
import { AlphaAuditLog } from "../security/audit";
import { assertAgainstSchema, type JsonSchema, objectSchema } from "./schema";

export type ToolSource = "builtin" | "mcp" | "custom";

export type ToolCharacteristics = {
  /** Talks to a network endpoint — denied by default agent sandboxes. */
  networked?: boolean;
  /** Reads or writes the local filesystem — denied by default sandboxes. */
  fileSystem?: boolean;
  /** Mutates persistent state. */
  mutates?: boolean;
};

export type ToolDescriptor = {
  name: string;
  description: string;
  module: string;
  version: string;
  inputSchema: JsonSchema;
  /**
   * Optional output contract. When declared, a handler returning something that
   * does not match fails the call — a tool cannot quietly change its shape and
   * feed a malformed result back into the model's context.
   */
  outputSchema: JsonSchema | null;
  /** Wall-clock budget for one call in milliseconds; 0 disables the timeout. */
  timeoutMs: number;
  permission: AlphaPermission;
  source: ToolSource;
  /** Requires an explicit, recorded human approval before each call. */
  requiresApproval: boolean;
  characteristics: ToolCharacteristics;
  /** Optional semantic tags used by the planner for capability matching. */
  tags: string[];
};

export type ToolExecutionRecord = {
  id: string;
  tool: string;
  actorId: string;
  args: Record<string, unknown>;
  ok: boolean;
  output: unknown;
  error: string | null;
  durationMs: number;
  startedAt: number;
  traceId: string | null;
  /** Set when an approval gate was satisfied. */
  approvalId?: string;
  /** True when the call was stopped by its timeout. */
  timedOut?: boolean;
};

export type ToolRunResult<TOutput = unknown> = {
  ok: boolean;
  tool: string;
  output: TOutput | null;
  error: { code: string; message: string; details: Record<string, unknown> } | null;
  durationMs: number;
  record: ToolExecutionRecord;
};

export type ToolVerification = {
  ok: boolean;
  reason: string;
};

export type ToolServices = Record<string, unknown>;

/**
 * Races a handler against a wall-clock budget. Rejects with
 * `AlphaToolTimeoutError` when the budget is spent, so a hanging tool becomes a
 * recorded failure rather than a stuck runtime.
 */
export async function withToolTimeout<T>(
  work: T | Promise<T>,
  toolName: string,
  timeoutMs: number,
): Promise<T> {
  if (!timeoutMs || timeoutMs <= 0) return await work;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      Promise.resolve(work),
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new AlphaToolTimeoutError(toolName, timeoutMs)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

export type ToolContext = {
  actorId: string;
  traceId: string | null;
  policy: AlphaPolicyEngine;
  services: ToolServices;
  /** Metadata the handler may attach to the execution record. */
  meta: Record<string, unknown>;
};

export type AlphaToolDefinition<TInput = unknown, TOutput = unknown> = {
  name: string;
  description: string;
  module: string;
  version?: string;
  inputSchema: JsonSchema;
  /** Optional output contract, enforced after the handler returns. */
  outputSchema?: JsonSchema;
  /** Wall-clock budget per call. Defaults to 10s; 0 disables the timeout. */
  timeoutMs?: number;
  permission: AlphaPermission;
  source?: ToolSource;
  requiresApproval?: boolean;
  dangerous?: boolean;
  characteristics?: ToolCharacteristics;
  tags?: string[];
  /**
   * Post-execution check used by the agent runtime to decide whether a result
   * is usable. Defaults to "output is not null/undefined".
   */
  verify?: (output: TOutput, input: TInput) => ToolVerification;
  handler: (input: TInput, context: ToolContext) => Promise<TOutput> | TOutput;
};

/** Default per-call wall-clock budget when a tool does not set one. */
export const DEFAULT_TOOL_TIMEOUT_MS = 10_000;

/** Raised when a tool exceeds its timeout. Reported, never silently ignored. */
export class AlphaToolTimeoutError extends AlphaToolError {
  constructor(toolName: string, timeoutMs: number) {
    super(`tool "${toolName}" exceeded its ${timeoutMs}ms timeout`, {
      tool: toolName,
      timeoutMs,
    });
    this.name = "AlphaToolTimeoutError";
  }
}

export type ToolRegistryOptions = {
  policy: AlphaPolicyEngine;
  rateLimiter?: AlphaRateLimiter;
  audit?: AlphaAuditLog;
  services?: ToolServices;
  onExecution?: (record: ToolExecutionRecord) => void;
};

export type ToolDiscoveryQuery = {
  /** Free text matched against name, description and tags. */
  query?: string;
  module?: string;
  source?: ToolSource;
  /** Only tools the given actor is allowed to call. */
  actorId?: string;
};

export class AlphaToolRegistry {
  private tools = new Map<string, AlphaToolDefinition<never, unknown>>();
  private executions: ToolExecutionRecord[] = [];
  private readonly policy: AlphaPolicyEngine;
  private readonly rateLimiter: AlphaRateLimiter | null;
  private readonly audit: AlphaAuditLog | null;
  private readonly onExecution: ((record: ToolExecutionRecord) => void) | null;
  private services: ToolServices;

  constructor(options: ToolRegistryOptions) {
    this.policy = options.policy;
    this.rateLimiter = options.rateLimiter ?? null;
    this.audit = options.audit ?? null;
    this.services = options.services ?? {};
    this.onExecution = options.onExecution ?? null;
  }

  setServices(services: ToolServices): void {
    this.services = { ...this.services, ...services };
  }

  register<TInput, TOutput>(tool: AlphaToolDefinition<TInput, TOutput>): ToolDescriptor {
    if (this.tools.has(tool.name)) {
      throw new AlphaToolError(`tool "${tool.name}" is already registered`, { tool: tool.name });
    }
    if (!/^[a-z][a-z0-9]*(\.[a-z0-9_]+)+$/.test(tool.name)) {
      throw new AlphaToolError(
        `tool name "${tool.name}" must be dotted lowercase, e.g. "alpha.text.stats"`,
        { tool: tool.name },
      );
    }
    this.tools.set(tool.name, tool as unknown as AlphaToolDefinition<never, unknown>);
    return describeTool(tool as unknown as AlphaToolDefinition<never, unknown>);
  }

  unregister(name: string): boolean {
    return this.tools.delete(name);
  }

  get(name: string): AlphaToolDefinition<never, unknown> | null {
    return this.tools.get(name) ?? null;
  }

  has(name: string): boolean {
    return this.tools.has(name);
  }

  list(): AlphaToolDefinition<never, unknown>[] {
    return [...this.tools.values()].sort((a, b) => (a.name < b.name ? -1 : 1));
  }

  describe(): ToolDescriptor[] {
    return this.list().map((tool) => describeTool(tool));
  }

  /**
   * Discovery. With an `actorId`, only tools that actor may actually call are
   * returned — an agent should never be shown a capability it cannot use.
   */
  discover(query: ToolDiscoveryQuery = {}): ToolDescriptor[] {
    let tools = this.describe();
    if (query.module) tools = tools.filter((tool) => tool.module === query.module);
    if (query.source) tools = tools.filter((tool) => tool.source === query.source);
    if (query.actorId) {
      const actorId = query.actorId;
      tools = tools.filter((tool) => this.policy.checkTool(actorId, {
        name: tool.name,
        permission: tool.permission,
        requiresApproval: tool.requiresApproval,
      }).allowed);
    }
    if (query.query && query.query.trim()) {
      const terms = query.query.toLowerCase().split(/\s+/).filter(Boolean);
      tools = tools
        .map((tool) => {
          const haystack = `${tool.name} ${tool.description} ${tool.tags.join(" ")}`.toLowerCase();
          const score = terms.reduce((sum, term) => sum + (haystack.includes(term) ? 1 : 0), 0);
          return { tool, score };
        })
        .filter((entry) => entry.score > 0)
        .sort((a, b) => b.score - a.score)
        .map((entry) => entry.tool);
    }
    return tools;
  }

  /** Capability match used by the planner: best tool for a required capability. */
  matchCapability(capability: string, actorId?: string): ToolDescriptor | null {
    const scored = this.discover({ query: capability, actorId });
    return scored[0] ?? null;
  }

  async execute<TOutput = unknown>(
    toolName: string,
    args: unknown,
    context: { actorId: string; traceId?: string | null; meta?: Record<string, unknown> },
  ): Promise<ToolRunResult<TOutput>> {
    const startedAt = Date.now();
    const id = alphaId("tool");
    const tool = this.tools.get(toolName);
    if (!tool) {
      const error = new AlphaToolError(`tool "${toolName}" is not registered`, { tool: toolName });
      return this.failure(id, toolName, context.actorId, args, error, startedAt, null);
    }

    const descriptor: ToolDescriptor = describeTool(tool);
    try {
      if (this.rateLimiter) {
        this.rateLimiter.assert(
          context.actorId,
          descriptor.requiresApproval ? "tool.execute.dangerous" : "tool",
        );
      }
      const decision = this.policy.assertTool(
        context.actorId,
        {
          name: descriptor.name,
          permission: descriptor.permission,
          requiresApproval: descriptor.requiresApproval,
        },
        descriptor.name,
      );
      assertAgainstSchema(args, descriptor.inputSchema, descriptor.name);
      const handlerContext: ToolContext = {
        actorId: context.actorId,
        traceId: context.traceId ?? null,
        policy: this.policy,
        services: this.services,
        meta: context.meta ?? {},
      };
      const handler = tool.handler as (input: unknown, ctx: ToolContext) => unknown;
      // The handler runs behind a timeout. Alpha is single-threaded JavaScript,
      // so this cannot preempt a synchronous handler mid-execution, but it does
      // bound every awaited handler, which is where an unbounded tool would
      // otherwise hang the runtime.
      const output = (await withToolTimeout(
        handler(args, handlerContext),
        descriptor.name,
        descriptor.timeoutMs,
      )) as TOutput;
      // The output contract is checked before the result is recorded or handed
      // back, so a malformed result never reaches the model's context.
      if (descriptor.outputSchema) {
        assertAgainstSchema(output, descriptor.outputSchema, `${descriptor.name} output`);
      }
      const record = this.success(
        id,
        descriptor,
        context.actorId,
        args,
        output,
        startedAt,
        context.traceId ?? null,
        decision.approvalId,
      );
      return {
        ok: true,
        tool: descriptor.name,
        output,
        error: null,
        durationMs: record.durationMs,
        record,
      };
    } catch (error) {
      return this.failure(
        id,
        descriptor.name,
        context.actorId,
        args,
        error,
        startedAt,
        context.traceId ?? null,
      );
    }
  }

  /** Verify a result using the tool's own verifier when it declares one. */
  verify(toolName: string, output: unknown, input: unknown): ToolVerification {
    const tool = this.tools.get(toolName);
    const verifier = tool?.verify as
      | ((output: unknown, input: unknown) => ToolVerification)
      | undefined;
    if (verifier) return verifier(output, input);
    if (output === null || output === undefined) {
      return { ok: false, reason: "tool returned no output" };
    }
    if (typeof output === "object" && "ok" in (output as Record<string, unknown>)) {
      const ok = (output as { ok?: unknown }).ok;
      if (ok === false) return { ok: false, reason: "tool reported failure" };
    }
    return { ok: true, reason: "output present and non-empty" };
  }

  recentExecutions(limit = 25): ToolExecutionRecord[] {
    return this.executions.slice(-limit).reverse();
  }

  stats(): {
    registered: number;
    executions: number;
    failures: number;
    averageDurationMs: number;
    byTool: { tool: string; executions: number; failures: number; averageDurationMs: number }[];
  } {
    const total = this.executions.length;
    const failures = this.executions.filter((record) => !record.ok).length;
    const averageDurationMs =
      total === 0 ? 0 : Number((this.executions.reduce((sum, r) => sum + r.durationMs, 0) / total).toFixed(2));
    const grouped = new Map<string, { executions: number; failures: number; duration: number }>();
    for (const record of this.executions) {
      const entry = grouped.get(record.tool) ?? { executions: 0, failures: 0, duration: 0 };
      entry.executions++;
      if (!record.ok) entry.failures++;
      entry.duration += record.durationMs;
      grouped.set(record.tool, entry);
    }
    return {
      registered: this.tools.size,
      executions: total,
      failures,
      averageDurationMs,
      byTool: [...grouped.entries()].map(([tool, entry]) => ({
        tool,
        executions: entry.executions,
        failures: entry.failures,
        averageDurationMs: Number((entry.duration / entry.executions).toFixed(2)),
      })),
    };
  }

  private success(
    id: string,
    tool: ToolDescriptor,
    actorId: string,
    args: unknown,
    output: unknown,
    startedAt: number,
    traceId: string | null,
    approvalId?: string,
  ): ToolExecutionRecord {
    const record: ToolExecutionRecord = {
      id,
      tool: tool.name,
      actorId,
      args: (args ?? {}) as Record<string, unknown>,
      ok: true,
      output,
      error: null,
      durationMs: Date.now() - startedAt,
      startedAt,
      traceId,
      approvalId,
    };
    this.record(record);
    return record;
  }

  private failure(
    id: string,
    toolName: string,
    actorId: string,
    args: unknown,
    error: unknown,
    startedAt: number,
    traceId: string | null,
  ): ToolRunResult<never> {
    const described = describeError(error);
    const record: ToolExecutionRecord = {
      id,
      tool: toolName,
      actorId,
      args: (args ?? {}) as Record<string, unknown>,
      ok: false,
      output: null,
      error: described.message,
      durationMs: Date.now() - startedAt,
      startedAt,
      traceId,
      ...(error instanceof AlphaToolTimeoutError ? { timedOut: true } : {}),
    };
    this.record(record);
    return {
      ok: false,
      tool: toolName,
      output: null,
      error: described,
      durationMs: record.durationMs,
      record,
    };
  }

  private record(record: ToolExecutionRecord): void {
    this.executions.push(record);
    if (this.executions.length > 500) this.executions = this.executions.slice(-250);
    this.audit?.append({
      actor: record.actorId,
      module: "tools",
      action: "tool.execute",
      resource: record.tool,
      decision: record.ok ? "allow" : "error",
      reason: record.ok ? "tool completed" : record.error ?? "tool failed",
      traceId: record.traceId,
      data: { durationMs: record.durationMs, approvalId: record.approvalId },
    });
    this.onExecution?.(record);
  }
}

/** Public, serialisable view of a tool. */
export function describeTool(tool: AlphaToolDefinition<never, unknown>): ToolDescriptor {
  return {
    name: tool.name,
    description: tool.description,
    module: tool.module,
    version: tool.version ?? "0.1.0",
    inputSchema: tool.inputSchema,
    outputSchema: tool.outputSchema ?? null,
    timeoutMs: tool.timeoutMs ?? DEFAULT_TOOL_TIMEOUT_MS,
    permission: tool.permission,
    source: tool.source ?? "custom",
    requiresApproval: tool.requiresApproval ?? false,
    characteristics: tool.characteristics ?? {},
    tags: tool.tags ?? [],
  };
}

/** Helper for tools that take a single required string argument. */
export function singleStringInput(
  name: string,
  description: string,
  extra: Record<string, JsonSchema> = {},
  required: string[] = [],
): JsonSchema {
  return objectSchema(
    { [name]: { type: "string", description, minLength: 1 }, ...extra },
    [name, ...required],
    undefined,
  );
}
