/**
 * Work — agents, tools and automation.
 *
 * An agent run is a record: plan, steps, tool calls, verification and the
 * method used to produce the final answer. Tool permissions are visible next to
 * each tool, and gated tools need a recorded approval before they can run.
 */

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { EmptyNote, Eyebrow, Frame, KeyValue, Mono, Pill, Stat, StatGrid } from "@/components/alpha/studio";
import type { AlphaRuntime } from "@/hooks/use-alpha";
import { useState } from "react";
import { CheckCircle2, KeyRound, Loader2, Play, ShieldAlert, Wand2, XCircle } from "lucide-react";
import { toast } from "sonner";

const SAMPLE_GOALS = [
  "calculate 12 * 4 and count the characters in the alpha corpus",
  "search the corpus for how approvals are recorded",
  "analyse the tokenizer for the sentence alpha owns its stack",
];

export function WorkPanel({ alpha }: { alpha: AlphaRuntime }) {
  const snapshot = alpha.snapshot;
  const [goal, setGoal] = useState(SAMPLE_GOALS[0]);
  const [maxSteps, setMaxSteps] = useState(3);
  const [preferModelPlanner, setPreferModelPlanner] = useState(false);
  const [workflowRun, setWorkflowRun] = useState<string | null>(null);

  const tools = snapshot?.tools.registered ?? [];
  const workflowStats = snapshot?.automation.stats;
  const workflows = snapshot?.automation.workflows ?? [];

  const runAgent = async () => {
    const result = await alpha.runAgent(goal, { maxSteps, preferModelPlanner });
    if (result) {
      toast.success(`Agent ${result.status} — ${result.plan.steps.length} step(s), ${result.toolCalls.length} tool call(s)`);
    } else {
      toast.error("Agent run did not produce a result");
    }
  };

  const registerSampleWorkflow = async () => {
    const workflow = await alpha.registerWorkflow({
      name: "corpus health check",
      description: "Runs a text statistics tool over a sample of the corpus, then logs the outcome.",
      trigger: { kind: "manual" },
      conditions: [{ path: "run", operator: "equals", value: true }],
      actions: [
        { kind: "tool", toolName: "alpha.text.stats", args: { text: "alpha owns its own stack and its own corpus" } },
        { kind: "log", message: "corpus health check complete" },
      ],
      enabled: true,
    });
    if (workflow) toast.success(`Registered workflow "${workflow.name}"`);
  };

  return (
    <div className="space-y-6">
      <Frame
        title="Agent runtime"
        status={snapshot?.statuses.agents ?? "planned"}
        lede="plan → select tool → authorize → execute → verify → synthesise. Bounded by a step budget inside a sandbox, with every step recorded."
        actions={
          <Button size="sm" onClick={runAgent} disabled={Boolean(alpha.busy)}>
            {alpha.busy === "agent" ? <Loader2 className="mr-2 size-3.5 animate-spin" /> : <Wand2 className="mr-2 size-3.5" />}
            Run agent
          </Button>
        }
      >
        <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
          <div className="space-y-3">
            <Textarea value={goal} onChange={(event) => setGoal(event.target.value)} className="min-h-20 font-mono text-xs" />
            <div className="flex flex-wrap gap-2">
              {SAMPLE_GOALS.map((sample) => (
                <button
                  key={sample}
                  type="button"
                  onClick={() => setGoal(sample)}
                  className="rounded-full border border-border px-3 py-1 text-[11px] text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground"
                >
                  {sample.slice(0, 46)}…
                </button>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2">
                <Label className="text-[11px] text-muted-foreground">Max steps</Label>
                <Input
                  type="number"
                  min={1}
                  max={8}
                  value={maxSteps}
                  onChange={(event) => setMaxSteps(Number(event.target.value))}
                  className="w-20"
                />
              </div>
              <label className="flex items-center gap-2 text-[11px] text-muted-foreground">
                <input
                  type="checkbox"
                  checked={preferModelPlanner}
                  onChange={(event) => setPreferModelPlanner(event.target.checked)}
                  className="size-3.5"
                />
                let Alpha's model plan (requires trained weights)
              </label>
            </div>
            <p className="text-[11px] leading-4 text-muted-foreground">
              With untrained weights the model planner is refused outright and the deterministic capability planner runs
              instead — a random model cannot plan, and this runtime will not pretend it can.
            </p>
          </div>

          <div className="space-y-3">
            <div className="rounded-md border border-border p-4">
              <Eyebrow>Sandbox for this run</Eyebrow>
              <div className="mt-2">
                <KeyValue label="actor">{alpha.workspace?.agentId ?? "alpha.agent"}</KeyValue>
                <KeyValue label="allowed tools">{snapshot?.tools.registered.length ?? 0} of 7 non-destructive</KeyValue>
                <KeyValue label="network">denied</KeyValue>
                <KeyValue label="filesystem">denied</KeyValue>
                <KeyValue label="step budget">{maxSteps}</KeyValue>
              </div>
            </div>
            {snapshot ? (
              <StatGrid className="grid-cols-2">
                <Stat label="Runs" value={snapshot.agents.runs} />
                <Stat label="Completed" value={snapshot.agents.completed} />
                <Stat label="Failed" value={snapshot.agents.failed} />
                <Stat label="Mean steps" value={snapshot.agents.averageSteps.toFixed(2)} />
              </StatGrid>
            ) : null}
          </div>
        </div>

        <div className="mt-6 space-y-3">
          <Eyebrow>Run records</Eyebrow>
          {alpha.agentRuns.length === 0 ? (
            <EmptyNote>No agent runs yet. Run one above to see the plan, the tool calls and the verification steps.</EmptyNote>
          ) : (
            alpha.agentRuns.map((run) => (
              <div key={run.id} className="rounded-md border border-border px-4 py-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="font-mono text-[11px] text-foreground">{run.goal}</p>
                  <div className="flex items-center gap-2">
                    <Pill>{run.status}</Pill>
                    <Pill>{run.steps} steps</Pill>
                    <Pill>{run.durationMs} ms</Pill>
                    <Pill>synthesis: {run.method}</Pill>
                  </div>
                </div>
                <div className="mt-2 space-y-1">
                  {run.toolCalls.map((call, index) => (
                    <p key={`${call.tool}-${index}`} className="flex items-center gap-2 text-[11px] text-muted-foreground">
                      {call.ok ? (
                        <CheckCircle2 className="size-3.5 text-chart-2" />
                      ) : (
                        <XCircle className="size-3.5 text-destructive" />
                      )}
                      <span className="font-mono">{call.tool}</span>
                      <span>{call.durationMs} ms</span>
                    </p>
                  ))}
                </div>
                <pre className="mt-3 max-h-40 overflow-auto whitespace-pre-wrap rounded-sm border border-border bg-muted/30 p-3 font-mono text-[11px] leading-5 text-foreground">
                  {run.synthesis}
                </pre>
                {run.blocker ? <p className="mt-2 text-[11px] text-destructive">{run.blocker}</p> : null}
              </div>
            ))
          )}
        </div>
      </Frame>

      <Frame
        title="Tool layer"
        status={snapshot?.statuses.tools ?? "planned"}
        lede="Every call passes rate limiting, authorization, JSON Schema validation, the handler, verification and the audit log — in that order."
        actions={
          <Button
            variant="outline"
            size="sm"
            onClick={async () => {
              const id = await alpha.approveTool("alpha.admin.clear_vector_store");
              if (id) toast.success("Approval recorded for the destructive tool");
            }}
            disabled={Boolean(alpha.busy)}
          >
            <KeyRound className="mr-2 size-3.5" />
            Approve destructive tool
          </Button>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-border">
                {["Tool", "Module", "Permission", "Approval", "Calls", "Failures"].map((heading) => (
                  <th key={heading} className="px-3 py-2 text-[10px] font-medium uppercase tracking-[0.16em] text-muted-foreground">
                    {heading}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tools.map((tool) => {
                const stats = snapshot?.tools.stats.byTool.find((entry) => entry.tool === tool.name);
                return (
                  <tr key={tool.name} className="border-b border-border/60 align-top">
                    <td className="px-3 py-2">
                      <p className="font-mono text-[11px] text-foreground">{tool.name}</p>
                      <p className="mt-1 max-w-md text-[11px] leading-4 text-muted-foreground">{tool.description}</p>
                    </td>
                    <td className="px-3 py-2">
                      <Pill>{tool.module}</Pill>
                    </td>
                    <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">{tool.permission}</td>
                    <td className="px-3 py-2">
                      {tool.requiresApproval ? (
                        <span className="inline-flex items-center gap-1 text-[11px] text-chart-4">
                          <ShieldAlert className="size-3.5" /> required
                        </span>
                      ) : (
                        <span className="text-[11px] text-muted-foreground">—</span>
                      )}
                    </td>
                    <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">{stats?.executions ?? 0}</td>
                    <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">{stats?.failures ?? 0}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="mt-5 grid gap-6 lg:grid-cols-2">
          <div>
            <Eyebrow>Recent executions</Eyebrow>
            <div className="mt-2 space-y-2">
              {(snapshot?.tools.recent ?? []).length === 0 ? (
                <EmptyNote>No tool has run yet.</EmptyNote>
              ) : (
                snapshot?.tools.recent.map((record) => (
                  <div key={record.id} className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2">
                    <div className="min-w-0">
                      <p className="truncate font-mono text-[11px] text-foreground">{record.tool}</p>
                      <Mono>
                        {record.actorId} · {record.durationMs} ms
                        {record.approvalId ? " · approved" : ""}
                      </Mono>
                    </div>
                    {record.ok ? (
                      <Pill className="border-chart-2/40 text-chart-2">ok</Pill>
                    ) : (
                      <Pill className="border-destructive/40 text-destructive">{record.error?.slice(0, 40) ?? "failed"}</Pill>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
          <div>
            <Eyebrow>MCP layer</Eyebrow>
            <div className="mt-2 rounded-md border border-border p-4">
              <p className="text-[11px] leading-5 text-muted-foreground">
                Alpha speaks the Model Context Protocol shape: `tools/list` and `tools/call` over JSON-RPC 2.0, with the
                server's JSON Schema converted into Alpha's own validation. The client is implemented and tested; it reports
                <span className="text-foreground"> NOT CONFIGURED </span>
                until an endpoint is supplied, because there is no MCP server to talk to by default.
              </p>
              <div className="mt-3">
                <KeyValue label="transport">http (JSON-RPC 2.0)</KeyValue>
                <KeyValue label="handshake">initialize → tools/list → tools/call</KeyValue>
                <KeyValue label="permissions">inherited from Alpha's policy engine</KeyValue>
                <KeyValue label="sandbox">network denied by default</KeyValue>
              </div>
            </div>
          </div>
        </div>
      </Frame>

      <Frame
        title="Automation"
        status={snapshot?.statuses.automation ?? "planned"}
        lede="Workflows with triggers, conditions, ordered actions, a job queue with retries and execution history."
        actions={
          <>
            <Button variant="outline" size="sm" onClick={registerSampleWorkflow} disabled={Boolean(alpha.busy)}>
              Register sample workflow
            </Button>
            <Button
              size="sm"
              onClick={async () => {
                const workflow = workflows[0];
                if (!workflow) {
                  toast.error("Register a workflow first");
                  return;
                }
                const job = await alpha.runWorkflow(workflow.id, { run: true });
                if (job) {
                  setWorkflowRun(`${job.status} in ${job.durationMs} ms (${job.attempts} attempt(s))`);
                  toast.success(`Workflow ${job.status}`);
                }
              }}
              disabled={Boolean(alpha.busy) || workflows.length === 0}
            >
              <Play className="mr-2 size-3.5" />
              Run latest workflow
            </Button>
          </>
        }
      >
        {workflowStats ? (
          <StatGrid className="mb-5">
            <Stat label="Workflows" value={workflowStats.workflows} />
            <Stat label="Enabled" value={workflowStats.enabled} />
            <Stat label="Executions" value={workflowStats.executions} />
            <Stat label="Failures" value={workflowStats.failed} hint={`${workflowStats.skipped} skipped by conditions`} />
          </StatGrid>
        ) : null}

        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-2">
            <Eyebrow>Registered workflows</Eyebrow>
            {workflows.length === 0 ? (
              <EmptyNote>No workflows registered in this session.</EmptyNote>
            ) : (
              workflows.map((workflow) => (
                <div key={workflow.id} className="rounded-md border border-border px-3 py-2">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-xs text-foreground">{workflow.name}</p>
                    <Pill>{workflow.enabled ? "enabled" : "disabled"}</Pill>
                  </div>
                  <Mono>
                    trigger {workflow.trigger.kind} · {workflow.conditions.length} condition(s) · {workflow.actions.length} action(s)
                  </Mono>
                  <p className="mt-1 text-[11px] leading-4 text-muted-foreground">{workflow.description}</p>
                </div>
              ))
            )}
          </div>
          <div className="space-y-2">
            <Eyebrow>Execution history</Eyebrow>
            {workflowRun ? <Pill className="mb-1">last run: {workflowRun}</Pill> : null}
            {(snapshot?.automation.history ?? []).length === 0 ? (
              <EmptyNote>No workflow has executed yet.</EmptyNote>
            ) : (
              snapshot?.automation.history.map((job) => (
                <div key={job.id} className="rounded-md border border-border px-3 py-2">
                  <div className="flex items-center justify-between gap-3">
                    <p className="font-mono text-[11px] text-foreground">{job.workflowName}</p>
                    <Pill>{job.status}</Pill>
                  </div>
                  <Mono>
                    {job.attempts} attempt(s) · {job.durationMs} ms · {job.outcomes.length} action(s)
                  </Mono>
                  {job.errors.length > 0 ? (
                    <p className="mt-1 text-[11px] text-destructive">{job.errors[0]}</p>
                  ) : null}
                </div>
              ))
            )}
          </div>
        </div>
      </Frame>
    </div>
  );
}
