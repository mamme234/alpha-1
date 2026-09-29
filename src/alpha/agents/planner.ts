/**
 * Alpha Agents — planning.
 *
 * Two planners, and the difference between them is honest:
 *
 * 1. `planFromCapabilities` — deterministic. It extracts capability phrases
 *    from the goal and matches them against registered tools. It works today,
 *    with no training, and it is what runs by default.
 *
 * 2. `planWithModel` — asks Alpha's own inference engine to produce a JSON
 *    plan. It refuses to run on an untrained model, because asking random
 *    weights to plan is theatre, and it falls back to (1) whenever the model's
 *    output is not a valid plan referencing registered tools.
 */

import { alphaId } from "../core/types";
import type { AlphaModelStage } from "../core/types";
import type { AlphaInferenceEngine } from "../inference/engine";
import type { AlphaToolRegistry, ToolDescriptor } from "../tools/registry";
import type { AgentPlan, AgentPlanStep, AgentTask, PlannerSource } from "./types";

const STOP_WORDS = new Set([
  "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
  "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "doing", "done",
  "to", "of", "in", "on", "at", "for", "with", "without", "from", "by", "as", "into", "about",
  "me", "my", "mine", "you", "your", "yours", "it", "its", "we", "our", "they", "their", "them",
  "please", "can", "could", "would", "should", "will", "shall", "may", "might", "must",
  "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "here", "there",
  "i", "im", "ive", "dont", "not", "no", "yes", "all", "any", "some", "very", "just",
]);

/** Extract the content words that describe what the task needs. */
export function extractCapabilityPhrases(goal: string, extra: string[] = []): string[] {
  const words = `${goal} ${extra.join(" ")}`
    .toLowerCase()
    .replace(/[^a-z0-9\s.+-]/g, " ")
    .split(/\s+/)
    .filter((word) => word.length > 2 && !STOP_WORDS.has(word));
  const unique = [...new Set(words)];
  // Prefer the caller's declared capabilities, then the goal's content words.
  return [...new Set([...extra.map((item) => item.toLowerCase()), ...unique])];
}

function stepFor(
  index: number,
  description: string,
  capability: string,
  tool: ToolDescriptor | null,
  expectedOutcome: string,
): AgentPlanStep {
  return {
    id: alphaId("step"),
    index,
    description,
    capability,
    toolName: tool ? tool.name : null,
    expectedOutcome,
    status: "pending",
    result: null,
    error: null,
    verification: null,
    durationMs: 0,
  };
}

/**
 * Deterministic planner. Each capability phrase becomes at most one step bound
 * to the best-matching tool available to this actor.
 */
export function planFromCapabilities(
  task: AgentTask,
  registry: AlphaToolRegistry,
  actorId: string,
  options: { maxSteps?: number } = {},
): AgentPlan {
  const maxSteps = options.maxSteps ?? task.maxSteps ?? 4;
  const phrases = extractCapabilityPhrases(task.goal, task.requiredCapabilities);
  const steps: AgentPlanStep[] = [];
  const usedTools = new Set<string>();

  for (const phrase of phrases) {
    if (steps.length >= maxSteps) break;
    const tool = registry.matchCapability(phrase, actorId);
    if (!tool || usedTools.has(tool.name)) continue;
    usedTools.add(tool.name);
    steps.push(
      stepFor(
        steps.length,
        `Use ${tool.name} to ${phrase}.`,
        phrase,
        tool,
        `A result from ${tool.name} that answers "${phrase}".`,
      ),
    );
  }

  if (steps.length === 0) {
    steps.push(
      stepFor(
        0,
        "No registered tool matches this goal, so no tool step can be planned.",
        "no-op",
        null,
        "An explanation that no capable tool is registered for this task.",
      ),
    );
  }

  return {
    taskId: task.id,
    createdAt: Date.now(),
    plannerSource: "capability-match",
    rationale: `Matched ${steps.length} step(s) from ${phrases.length} capability phrase(s) using the tool registry; no model call was made.`,
    steps,
  };
}

export type ModelPlannerOptions = {
  inference: AlphaInferenceEngine;
  registry: AlphaToolRegistry;
  actorId: string;
  maxSteps?: number;
  sampling?: Parameters<AlphaInferenceEngine["generate"]>[1];
};

/** Stages where a model is allowed to plan at all. */
const PLANNING_STAGES: AlphaModelStage[] = ["trained", "fine-tuned", "production"];

export function buildPlannerPrompt(task: AgentTask, tools: ToolDescriptor[], maxSteps: number): string {
  const toolLines = tools.map(
    (tool) => `- ${tool.name}: ${tool.description} (permission: ${tool.permission})`,
  );
  return [
    "Instruction: you are Alpha's planner. Reply with JSON only.",
    'Schema: {"steps":[{"tool":"<tool.name>","description":"<one sentence>"}],"rationale":"<one sentence>"}',
    `Use at most ${maxSteps} steps. Use only the tools listed below.`,
    `Goal: ${task.goal}`,
    "Tools:",
    ...toolLines,
    "JSON:",
  ].join("\n");
}

export function parsePlannerJson(raw: string): { steps: { tool: string; description: string }[]; rationale: string } | null {
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    const parsed = JSON.parse(raw.slice(start, end + 1)) as {
      steps?: unknown;
      rationale?: unknown;
    };
    if (!Array.isArray(parsed.steps)) return null;
    const steps = parsed.steps
      .filter((step): step is { tool: unknown; description?: unknown } => typeof step === "object" && step !== null)
      .map((step) => ({
        tool: String((step as { tool?: unknown }).tool ?? ""),
        description: String((step as { description?: unknown }).description ?? ""),
      }))
      .filter((step) => step.tool.length > 0);
    if (steps.length === 0) return null;
    return { steps, rationale: typeof parsed.rationale === "string" ? parsed.rationale : "" };
  } catch {
    return null;
  }
}

/**
 * Model-based planner. Returns `null` whenever the model may not or cannot
 * plan, so the caller falls back to the deterministic planner.
 */
export function planWithModel(task: AgentTask, options: ModelPlannerOptions): AgentPlan | null {
  const stage = options.inference.stage;
  if (!PLANNING_STAGES.includes(stage)) return null;
  const tools = options.registry.discover({ actorId: options.actorId });
  if (tools.length === 0) return null;
  const maxSteps = options.maxSteps ?? task.maxSteps ?? 4;
  const prompt = buildPlannerPrompt(task, tools, maxSteps);
  const generation = options.inference.generate(prompt, {
    maxNewTokens: 160,
    temperature: 0.2,
    stopSequences: ["\n\n"],
    ...options.sampling,
  });
  const parsed = parsePlannerJson(generation.text);
  if (!parsed) return null;
  const steps: AgentPlanStep[] = [];
  for (const proposed of parsed.steps.slice(0, maxSteps)) {
    const tool = tools.find((candidate) => candidate.name === proposed.tool);
    if (!tool) continue;
    steps.push(
      stepFor(
        steps.length,
        proposed.description || `Use ${tool.name}.`,
        proposed.description || tool.description,
        tool,
        `A result from ${tool.name}.`,
      ),
    );
  }
  if (steps.length === 0) return null;
  return {
    taskId: task.id,
    createdAt: Date.now(),
    plannerSource: "model" satisfies PlannerSource,
    rationale: parsed.rationale || "Plan produced by Alpha's own model.",
    steps,
  };
}
