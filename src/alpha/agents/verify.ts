/**
 * Alpha Agents — objective verification.
 *
 * An agent that finishes its loop has not necessarily achieved what it was
 * asked. This module looks only at evidence — which steps ran, which tools
 * succeeded, what each tool's own verification said, and what the sandbox
 * recorded — and reports one of five outcomes. There is no path that reports
 * COMPLETED without at least one step having produced a verified result.
 */

import type { AgentOutcome, AgentPlan, AgentRunStatus, AgentVerification } from "./types";
import type { ToolExecutionRecord } from "../tools/registry";

type SandboxReport = {
  cancelled: boolean;
  cancelReason: string | null;
  toolCalls: number;
  maxToolCalls: number;
  violations: { code: string; reason: string }[];
};

export type { SandboxReport };

export type VerifyAgentOutcomeInput = {
  plan: AgentPlan;
  toolCalls: ToolExecutionRecord[];
  status: AgentRunStatus | null;
  sandbox: SandboxReport;
  blocker: string | null;
};

/**
 * Derive the outcome from evidence. `status` is not trusted for the outcome —
 * it is only used to explain a failure that already happened.
 */
export function verifyAgentOutcome(input: VerifyAgentOutcomeInput): AgentVerification {
  const { plan, sandbox, blocker } = input;
  const stepsCompleted = plan.steps.filter((step) => step.status === "done").length;
  const stepsFailed = plan.steps.filter((step) => step.status === "failed").length;
  const stepsSkipped = plan.steps.filter((step) => step.status === "skipped").length;
  const stepVerifications = plan.steps
    .filter((step) => step.verification !== null)
    .map((step) => ({
      stepId: step.id,
      ok: step.verification?.ok ?? false,
      reason: step.verification?.reason ?? "no verification recorded",
    }));
  const failedToolCalls = input.toolCalls.filter((record) => !record.ok).length;

  // A gated tool that was never approved means the run needs a human, not a
  // verdict. This is the one case where "incomplete" is not the agent's fault.
  if (sandbox.violations.some((violation) => violation.reason.includes("approval"))) {
    return {
      outcome: "REQUIRES_APPROVAL",
      objectiveMet: false,
      stepsCompleted,
      stepsFailed,
      stepsSkipped,
      stepVerifications,
      reason: "a tool call was blocked by an approval gate; the run cannot continue without a human decision",
      outstanding: "record an approval for the gated tool and re-run",
    };
  }

  if (sandbox.cancelled) {
    return {
      outcome: "CANCELLED",
      objectiveMet: false,
      stepsCompleted,
      stepsFailed,
      stepsSkipped,
      stepVerifications,
      reason: sandbox.cancelReason ?? "the run was cancelled before it finished",
      outstanding: stepsCompleted > 0 ? "resume or re-run to attempt the remaining steps" : "re-run the objective",
    };
  }

  if (plan.steps.length === 0) {
    return {
      outcome: "FAILED",
      objectiveMet: false,
      stepsCompleted: 0,
      stepsFailed: 0,
      stepsSkipped: 0,
      stepVerifications,
      reason: "the planner produced no executable steps, so the objective was never attempted",
      outstanding: "check that the goal maps to a capability the agent can reach",
    };
  }

  const budgetExhausted =
    sandbox.toolCalls >= sandbox.maxToolCalls ||
    sandbox.violations.some((violation) => violation.code.endsWith("budget"));

  if (stepsCompleted === 0) {
    return {
      outcome: "FAILED",
      objectiveMet: false,
      stepsCompleted,
      stepsFailed,
      stepsSkipped,
      stepVerifications,
      reason:
        blocker ??
        (failedToolCalls > 0
          ? `all ${failedToolCalls} tool call(s) failed, so no step produced a usable result`
          : "no step produced a verified result"),
      outstanding: "inspect the recorded tool errors before retrying",
    };
  }

  const anyFailed = stepsFailed > 0 || stepsSkipped > 0 || failedToolCalls > 0;
  if (anyFailed || budgetExhausted) {
    const unmet = stepsFailed + stepsSkipped;
    return {
      outcome: "PARTIAL",
      objectiveMet: false,
      stepsCompleted,
      stepsFailed,
      stepsSkipped,
      stepVerifications,
      reason: `${stepsCompleted} of ${plan.steps.length} step(s) produced a verified result; ${unmet} did not${
        budgetExhausted ? " and a budget was reached" : ""
      }`,
      outstanding: `${unmet} step(s) remain unverified; Alpha cannot claim the objective was met`,
    };
  }

  return {
    outcome: "COMPLETED",
    objectiveMet: true,
    stepsCompleted,
    stepsFailed,
    stepsSkipped,
    stepVerifications,
    reason: `all ${stepsCompleted} step(s) executed and each tool reported a verified result`,
    outstanding: null,
  };
}

/**
 * Narrow the outcome when the run's own status contradicts the evidence — a
 * run that was cut short can never be reported as COMPLETED.
 */
export function reconcileOutcome(
  verification: AgentVerification,
  status: AgentRunStatus,
  blocker: string | null,
  sandbox: SandboxReport,
): AgentOutcome {
  if (sandbox.cancelled) return "CANCELLED";
  if (verification.outcome === "REQUIRES_APPROVAL") return "REQUIRES_APPROVAL";
  if (verification.outcome === "COMPLETED") {
    // A completed run with a blocker is at best partial: something went wrong
    // that Alpha can point at, so it does not get to claim full success.
    if (status === "failed" || status === "blocked" || blocker) return "PARTIAL";
    return "COMPLETED";
  }
  if (verification.outcome === "FAILED" && status === "completed") {
    // The loop finished but nothing was achieved — that is a failure, and the
    // synthesis must not present it as anything else.
    return "FAILED";
  }
  return verification.outcome;
}
