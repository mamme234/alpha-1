/**
 * Alpha Security — sandbox boundaries for agents.
 *
 * An agent runs inside a boundary: a step budget, a tool allow-list, a
 * permission set, and a hard "no network, no filesystem, no shell" default.
 * The boundary is data, so it can be inspected in the workspace and asserted in
 * tests, rather than being an implicit property of the code that happens to
 * exist today.
 */

import { AlphaPermissionError } from "../core/errors";

export type SandboxSpec = {
  agentId: string;
  /** Tools the agent may call. */
  allowedTools: string[];
  /** Maximum reasoning steps before the run is stopped. */
  maxSteps: number;
  /** Maximum tool calls across the whole run, independent of steps. */
  maxToolCalls: number;
  /**
   * How many times the same tool may be called with identical arguments
   * before the run is treated as looping. Set to 0 to disable the check.
   */
  maxIdenticalCalls: number;
  /** Maximum tokens the agent may generate across the whole run. */
  maxGeneratedTokens: number;
  /** Wall-clock budget. */
  maxDurationMs: number;
  /** Network tools are denied unless explicitly enabled. */
  allowNetwork: boolean;
  /** Filesystem tools are denied unless explicitly enabled. */
  allowFileSystem: boolean;
};

export const DEFAULT_SANDBOX: Omit<SandboxSpec, "agentId" | "allowedTools"> = {
  maxSteps: 6,
  maxToolCalls: 12,
  maxIdenticalCalls: 2,
  maxGeneratedTokens: 1024,
  maxDurationMs: 120_000,
  allowNetwork: false,
  allowFileSystem: false,
};

export type SandboxViolation = {
  code: string;
  reason: string;
};

export type SandboxState = {
  startedAt: number;
  steps: number;
  toolCalls: number;
  generatedTokens: number;
  violations: SandboxViolation[];
  /** Fingerprint -> how many times it has been called. Drives loop detection. */
  callCounts: Map<string, number>;
  cancelled: boolean;
  cancelReason: string | null;
};

export class AlphaSandbox {
  readonly spec: SandboxSpec;
  readonly state: SandboxState;

  constructor(spec: SandboxSpec) {
    this.spec = spec;
    this.state = {
      startedAt: Date.now(),
      steps: 0,
      toolCalls: 0,
      generatedTokens: 0,
      violations: [],
      callCounts: new Map(),
      cancelled: false,
      cancelReason: null,
    };
  }

  /** Stop the run at the next boundary check. Cooperative, not preemptive. */
  cancel(reason = "cancelled by caller"): void {
    this.state.cancelled = true;
    this.state.cancelReason = reason;
  }

  get cancelled(): boolean {
    return this.state.cancelled;
  }

  /**
   * A stable fingerprint for a call, used to notice a loop. Arguments are
   * sorted so key order cannot disguise a repeated identical call.
   */
  static fingerprint(toolName: string, args: unknown): string {
    let serialised: string;
    try {
      serialised = JSON.stringify(args ?? null, (_key, value) => {
        if (value && typeof value === "object" && !Array.isArray(value)) {
          return Object.fromEntries(Object.entries(value as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : 1)));
        }
        return value;
      });
    } catch {
      serialised = "[unserialisable]";
    }
    return `${toolName}:${serialised}`;
  }

  private deny(code: string, reason: string): never {
    this.state.violations.push({ code, reason });
    throw new AlphaPermissionError("security", reason, { agentId: this.spec.agentId, code });
  }

  /** Check a tool call is inside the boundary, then account for the step. */
  enterTool(
    toolName: string,
    characteristics: { networked?: boolean; fileSystem?: boolean } = {},
    args: unknown = null,
  ): void {
    if (this.state.cancelled) {
      this.deny("sandbox.cancelled", this.state.cancelReason ?? "run cancelled");
    }
    if (!this.spec.allowedTools.includes(toolName)) {
      this.deny("sandbox.tool_not_allowed", `tool "${toolName}" is not allowed for agent "${this.spec.agentId}"`);
    }
    if (characteristics.networked && !this.spec.allowNetwork) {
      this.deny("sandbox.network_denied", `tool "${toolName}" requires network access, which this sandbox denies`);
    }
    if (characteristics.fileSystem && !this.spec.allowFileSystem) {
      this.deny("sandbox.filesystem_denied", `tool "${toolName}" requires filesystem access, which this sandbox denies`);
    }
    if (this.state.steps >= this.spec.maxSteps) {
      this.deny("sandbox.step_budget", `step budget of ${this.spec.maxSteps} exhausted`);
    }
    if (this.state.toolCalls >= this.spec.maxToolCalls) {
      this.deny("sandbox.tool_call_budget", `tool call budget of ${this.spec.maxToolCalls} exhausted`);
    }
    if (Date.now() - this.state.startedAt > this.spec.maxDurationMs) {
      this.deny("sandbox.time_budget", `time budget of ${this.spec.maxDurationMs}ms exhausted`);
    }
    // Loop detection: the same call repeated with the same arguments is almost
    // never progress, and an agent that does it has stopped reasoning.
    if (this.spec.maxIdenticalCalls > 0) {
      const key = AlphaSandbox.fingerprint(toolName, args);
      const seen = (this.state.callCounts.get(key) ?? 0) + 1;
      this.state.callCounts.set(key, seen);
      if (seen > this.spec.maxIdenticalCalls) {
        this.deny(
          "sandbox.loop_detected",
          `tool "${toolName}" was called ${seen} times with identical arguments; treating this as a loop`,
        );
      }
    }
    this.state.steps++;
    this.state.toolCalls++;
  }

  /** Account for generated tokens and enforce the run-wide budget. */
  accountTokens(tokens: number): void {
    this.state.generatedTokens += tokens;
    if (this.state.generatedTokens > this.spec.maxGeneratedTokens) {
      this.deny(
        "sandbox.token_budget",
        `token budget of ${this.spec.maxGeneratedTokens} exhausted`,
      );
    }
  }

  get exhausted(): boolean {
    return (
      this.state.cancelled ||
      this.state.steps >= this.spec.maxSteps ||
      this.state.toolCalls >= this.spec.maxToolCalls ||
      Date.now() - this.state.startedAt > this.spec.maxDurationMs
    );
  }

  report(): {
    agentId: string;
    steps: number;
    maxSteps: number;
    toolCalls: number;
    maxToolCalls: number;
    generatedTokens: number;
    maxGeneratedTokens: number;
    elapsedMs: number;
    maxDurationMs: number;
    allowNetwork: boolean;
    allowFileSystem: boolean;
    cancelled: boolean;
    cancelReason: string | null;
    violations: SandboxViolation[];
  } {
    return {
      agentId: this.spec.agentId,
      steps: this.state.steps,
      maxSteps: this.spec.maxSteps,
      toolCalls: this.state.toolCalls,
      maxToolCalls: this.spec.maxToolCalls,
      generatedTokens: this.state.generatedTokens,
      maxGeneratedTokens: this.spec.maxGeneratedTokens,
      elapsedMs: Date.now() - this.state.startedAt,
      maxDurationMs: this.spec.maxDurationMs,
      allowNetwork: this.spec.allowNetwork,
      allowFileSystem: this.spec.allowFileSystem,
      cancelled: this.state.cancelled,
      cancelReason: this.state.cancelReason,
      violations: [...this.state.violations],
    };
  }
}
