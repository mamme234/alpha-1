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
  generatedTokens: number;
  violations: SandboxViolation[];
};

export class AlphaSandbox {
  readonly spec: SandboxSpec;
  readonly state: SandboxState;

  constructor(spec: SandboxSpec) {
    this.spec = spec;
    this.state = { startedAt: Date.now(), steps: 0, generatedTokens: 0, violations: [] };
  }

  private deny(code: string, reason: string): never {
    this.state.violations.push({ code, reason });
    throw new AlphaPermissionError("security", reason, { agentId: this.spec.agentId, code });
  }

  /** Check a tool call is inside the boundary, then account for the step. */
  enterTool(toolName: string, characteristics: { networked?: boolean; fileSystem?: boolean } = {}): void {
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
    if (Date.now() - this.state.startedAt > this.spec.maxDurationMs) {
      this.deny("sandbox.time_budget", `time budget of ${this.spec.maxDurationMs}ms exhausted`);
    }
    this.state.steps++;
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
      this.state.steps >= this.spec.maxSteps ||
      Date.now() - this.state.startedAt > this.spec.maxDurationMs
    );
  }

  report(): {
    agentId: string;
    steps: number;
    maxSteps: number;
    generatedTokens: number;
    maxGeneratedTokens: number;
    elapsedMs: number;
    maxDurationMs: number;
    allowNetwork: boolean;
    allowFileSystem: boolean;
    violations: SandboxViolation[];
  } {
    return {
      agentId: this.spec.agentId,
      steps: this.state.steps,
      maxSteps: this.spec.maxSteps,
      generatedTokens: this.state.generatedTokens,
      maxGeneratedTokens: this.spec.maxGeneratedTokens,
      elapsedMs: Date.now() - this.state.startedAt,
      maxDurationMs: this.spec.maxDurationMs,
      allowNetwork: this.spec.allowNetwork,
      allowFileSystem: this.spec.allowFileSystem,
      violations: [...this.state.violations],
    };
  }
}
