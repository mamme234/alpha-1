/**
 * Alpha Agents — shared types.
 *
 * An agent run is a record: a goal, a plan, the steps taken, the tools called,
 * what was verified, and how it ended. Nothing about a run is implicit, which
 * is what makes the run inspectable after the fact.
 */

import type { AlphaModelStage } from "../core/types";
import type { ToolExecutionRecord } from "../tools/registry";
import type { ToolVerification } from "../tools/registry";

export type AgentRunStatus =
  | "completed"
  | "failed"
  | "step-budget-exhausted"
  | "blocked"
  | "cancelled";

export type AgentStepStatus = "pending" | "running" | "done" | "failed" | "skipped";

export type AgentTask = {
  id: string;
  /** What the user asked for, verbatim. */
  goal: string;
  /** Extra grounding supplied by the caller (never instructions from documents). */
  context?: string;
  /** Capabilities the caller believes are needed; helps the planner. */
  requiredCapabilities?: string[];
  maxSteps?: number;
  sessionId?: string | null;
  createdAt: number;
};

export type AgentPlanStep = {
  id: string;
  index: number;
  description: string;
  /** Capability this step satisfies, e.g. "search the corpus". */
  capability: string;
  toolName: string | null;
  expectedOutcome: string;
  status: AgentStepStatus;
  result: unknown;
  error: string | null;
  verification: ToolVerification | null;
  durationMs: number;
};

export type PlannerSource = "capability-match" | "model" | "single-step";

export type AgentPlan = {
  taskId: string;
  createdAt: number;
  plannerSource: PlannerSource;
  rationale: string;
  steps: AgentPlanStep[];
};

export type AgentStepRecord = {
  index: number;
  phase: "plan" | "execute" | "verify" | "synthesise" | "blocked";
  at: number;
  durationMs: number;
  detail: string;
  toolName?: string | null;
  ok: boolean;
};

export type AgentSynthesis = {
  /** How the final answer was produced — never implied, always stated. */
  method: "model" | "structured";
  text: string;
  modelStage: AlphaModelStage | null;
  warning: string | null;
  tokensGenerated: number;
};

export type AgentRunResult = {
  id: string;
  taskId: string;
  goal: string;
  status: AgentRunStatus;
  plan: AgentPlan;
  history: AgentStepRecord[];
  toolCalls: ToolExecutionRecord[];
  synthesis: AgentSynthesis;
  startedAt: number;
  finishedAt: number;
  durationMs: number;
  traceId: string;
  modelStage: AlphaModelStage | null;
  sandbox: ReturnType<import("../security/sandbox").AlphaSandbox["report"]>;
  blocker: string | null;
};

export type AgentEvent =
  | { type: "start"; task: AgentTask; traceId: string }
  | { type: "plan"; plan: AgentPlan }
  | { type: "step"; step: AgentPlanStep; history: AgentStepRecord }
  | { type: "tool"; record: ToolExecutionRecord }
  | { type: "verify"; stepId: string; verification: ToolVerification }
  | { type: "synthesise"; synthesis: AgentSynthesis }
  | { type: "blocked"; reason: string }
  | { type: "done"; result: AgentRunResult };
