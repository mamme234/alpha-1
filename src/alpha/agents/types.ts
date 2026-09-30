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

/**
 * Whether the run actually achieved what it was asked to do. This is deliberately
 * separate from `AgentRunStatus`: a run can finish cleanly and still have failed
 * to complete its objective, and Alpha must not report success it cannot show.
 */
export type AgentOutcome = "COMPLETED" | "PARTIAL" | "FAILED" | "CANCELLED" | "REQUIRES_APPROVAL";

/** Why Alpha concluded the run had the outcome it reports. */
export type AgentVerification = {
  outcome: AgentOutcome;
  /** True only when Alpha has evidence the objective was met. */
  objectiveMet: boolean;
  /** Steps that succeeded, failed, or were never attempted. */
  stepsCompleted: number;
  stepsFailed: number;
  stepsSkipped: number;
  /** Per-step verification as recorded by each tool. */
  stepVerifications: { stepId: string; ok: boolean; reason: string }[];
  /** Why this outcome, stated in one sentence. */
  reason: string;
  /** What would be needed to reach COMPLETED, when it was not reached. */
  outstanding: string | null;
};

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
  /** The account that started the run. */
  actorId: string;
  status: AgentRunStatus;
  /** Architecture version of the weights the run used, for provenance. */
  modelName: string | null;
  modelVersion: string | null;
  plan: AgentPlan;
  history: AgentStepRecord[];
  toolCalls: ToolExecutionRecord[];
  synthesis: AgentSynthesis;
  /** Did the run achieve its objective? Derived, never assumed. */
  verification: AgentVerification;
  startedAt: number;
  finishedAt: number;
  durationMs: number;
  traceId: string;
  modelStage: AlphaModelStage | null;
  sandbox: ReturnType<import("../security/sandbox").AlphaSandbox["report"]>;
  blocker: string | null;
  /** Errors encountered that did not stop the run. */
  errors: string[];
};

export type AgentEvent =
  | { type: "start"; task: AgentTask; traceId: string }
  | { type: "plan"; plan: AgentPlan }
  | { type: "step"; step: AgentPlanStep; history: AgentStepRecord }
  | { type: "tool"; record: ToolExecutionRecord }
  | { type: "verify"; stepId: string; verification: ToolVerification }
  | { type: "verified"; verification: AgentVerification }
  | { type: "synthesise"; synthesis: AgentSynthesis }
  | { type: "blocked"; reason: string }
  | { type: "done"; result: AgentRunResult };
