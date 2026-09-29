/**
 * useAlpha — the browser-side Alpha runtime.
 *
 * Owns one `AlphaWorkspace` per signed-in user, drives it from React, and
 * pushes its artifacts into Convex. Everything the UI shows comes from
 * `workspace.snapshot()`; nothing is mocked, and long operations (training)
 * yield to the event loop between steps so the page keeps painting.
 */

import { api } from "@/convex/_generated/api";
import { useAuth } from "@/hooks/use-auth";
import { createConvexPersistence, createToolSync } from "@/lib/alpha-persistence";
import {
  ALPHA_SEED_CORPUS,
  AlphaWorkspace,
  datasetStats,
  type AlphaCheckpoint,
  type AlphaWorkspaceSnapshot,
  type AgentRunResult,
  type GenerationResult,
  type IngestedDocument,
  type MemoryRecord,
  type MemoryRetrieval,
  type MemoryScope,
  type RagAnswer,
  type SamplingConfig,
  type TrainingConfig,
  type TrainingSummary,
} from "@/alpha";
import { useConvex, useMutation, useQuery } from "convex/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export type TrainingProgress = {
  running: boolean;
  step: number;
  totalSteps: number;
  loss: number | null;
  validationLoss: number | null;
  uniformLoss: number | null;
  history: { step: number; loss: number }[];
  summary: TrainingSummary | null;
};

export type ChatTurn = {
  id: string;
  prompt: string;
  answer: string;
  result: GenerationResult | null;
  sources: RagAnswer["sources"];
  mode: "generate" | "rag";
  at: number;
};

export type AgentRunSummary = {
  id: string;
  goal: string;
  status: AgentRunResult["status"];
  steps: number;
  toolCalls: { tool: string; ok: boolean; durationMs: number }[];
  synthesis: string;
  method: AgentRunResult["synthesis"]["method"];
  durationMs: number;
  blocker: string | null;
};

/** One workspace per actor, reused across component remounts. */
let runtime: { actorId: string; workspace: AlphaWorkspace } | null = null;

const nextTick = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

export function useAlpha() {
  const { user, sessionToken } = useAuth();
  const convex = useConvex();
  const actorId = user?.id ?? null;

  // Every Alpha function resolves the caller from this token, so the client
  // never names its own account. Keyed through a ref so the wrappers below
  // always send the current one.
  const sessionTokenRef = useRef<string | null>(sessionToken);
  sessionTokenRef.current = sessionToken;
  const requireToken = () => sessionTokenRef.current ?? "";

  const recordModel = useMutation(api.alpha.models.record);
  const saveTokenizer = useMutation(api.alpha.training.saveTokenizer);
  const saveCheckpoint = useMutation(api.alpha.training.saveCheckpoint);
  const saveVectors = useMutation(api.alpha.vector.upsertVectors);
  const saveMemories = useMutation(api.alpha.memory.upsertMemories);
  const saveSpans = useMutation(api.alpha.observability.recordSpans);
  const saveAudit = useMutation(api.alpha.observability.appendAudit);
  const recordRun = useMutation(api.alpha.observability.recordRun);
  const saveWorkflow = useMutation(api.alpha.workflows.saveWorkflow);
  const saveJob = useMutation(api.alpha.workflows.saveJob);
  const syncTools = useMutation(api.alpha.tools.syncTools);

  /**
   * The persistence adapter stays provider-agnostic: it takes plain functions.
   * Attaching the session token is this file's job, done once here, so no call
   * site can forget it and no Alpha module needs to know about sessions.
   */
  const alphaMutations: Parameters<typeof createConvexPersistence>[0] = {
    recordModel: (args) => recordModel({ ...args, sessionToken: requireToken() }),
    saveTokenizer: (args) => saveTokenizer({ ...args, sessionToken: requireToken() }),
    saveCheckpoint: (args) => saveCheckpoint({ ...args, sessionToken: requireToken() }),
    saveVectors: (args) => saveVectors({ ...args, sessionToken: requireToken() }),
    saveMemories: (args) => saveMemories({ ...args, sessionToken: requireToken() }),
    saveSpans: (args) => saveSpans({ ...args, sessionToken: requireToken() }),
    saveAudit: (args) => saveAudit({ ...args, sessionToken: requireToken() }),
    recordRun: (args) => recordRun({ ...args, sessionToken: requireToken() }),
    saveWorkflow: (args) => saveWorkflow({ ...args, sessionToken: requireToken() }),
    saveJob: (args) => saveJob({ ...args, sessionToken: requireToken() }),
    syncTools: (args) => syncTools({ ...args, sessionToken: requireToken() }),
  };

  // Conversations live in Alpha's own tables so a transcript survives a reload.
  const startConversation = useMutation(api.alpha.conversations.start);
  const appendMessage = useMutation(api.alpha.conversations.appendMessage);
  const removeConversation = useMutation(api.alpha.conversations.remove);
  const conversations =
    useQuery(api.alpha.conversations.list, sessionToken ? { sessionToken } : "skip") ?? [];
  const conversationRef = useRef<string | null>(null);

  /**
   * Store one exchange. Best-effort by design: a storage problem must never
   * lose the generation the user just watched appear.
   */
  const recordTurn = useCallback(
    async (turn: {
      prompt: string;
      answer: string;
      mode: "generate" | "rag";
      result: GenerationResult | null;
      sources: RagAnswer["sources"];
    }) => {
      try {
        const token = requireToken();
        if (!token) return;
        let conversationId = conversationRef.current;
        if (!conversationId) {
          const created = await startConversation({ sessionToken: token, title: turn.prompt.slice(0, 80), kind: turn.mode });
          conversationId = created.conversationId;
          conversationRef.current = conversationId;
        }
        await appendMessage({ sessionToken: token, conversationId, role: "user", content: turn.prompt });
        await appendMessage({
          sessionToken: token,
          conversationId,
          role: "assistant",
          content: turn.answer,
          sources: turn.sources.map((source) => ({
            title: source.title,
            score: source.score,
            chunkId: source.chunkId,
          })),
          modelStage: turn.result?.modelStage,
          tokens: turn.result?.generatedTokens,
        });
      } catch (error) {
        console.warn("[alpha] conversation not stored:", error);
      }
    },
    [appendMessage, startConversation],
  );

  const deleteConversation = useCallback(
    async (conversationId: string) => {
      const token = requireToken();
      if (!token) return;
      await removeConversation({ sessionToken: token, conversationId });
      if (conversationRef.current === conversationId) conversationRef.current = null;
    },
    [removeConversation],
  );

  const [workspace, setWorkspace] = useState<AlphaWorkspace | null>(null);
  const [snapshot, setSnapshot] = useState<AlphaWorkspaceSnapshot | null>(null);
  const [training, setTraining] = useState<TrainingProgress | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [agentRuns, setAgentRuns] = useState<AgentRunSummary[]>([]);
  const [streaming, setStreaming] = useState<string | null>(null);

  useEffect(() => {
    if (!actorId) return;
    if (runtime && runtime.actorId === actorId) {
      setWorkspace(runtime.workspace);
      setSnapshot(runtime.workspace.snapshot());
      return;
    }
    let cancelled = false;
    setBootError(null);
    const boot = async () => {
      try {
        const persistence = createConvexPersistence(alphaMutations);
        const ws = new AlphaWorkspace({
          actorId,
          persistence,
          dataset: ALPHA_SEED_CORPUS,
          logLevel: "info",
        });
        await ws.initialise();
        if (cancelled) return;
        runtime = { actorId, workspace: ws };
        // A different account gets its own transcript.
        conversationRef.current = null;
        const initial = ws.snapshot();
        setWorkspace(ws);
        setSnapshot(initial);
        // Mirror the tool surface so permissions and approval gates stay
        // reviewable between sessions.
        void createToolSync(alphaMutations)?.(
          initial.tools.registered.map((tool) => ({
            name: tool.name,
            description: tool.description,
            module: tool.module,
            permission: tool.permission,
            requiresApproval: tool.requiresApproval,
            dangerous: tool.permission === "tool.execute.dangerous",
            source: tool.source,
            inputSchema: tool.inputSchema,
            stats: initial.tools.stats.byTool.find((entry) => entry.tool === tool.name) ?? {
              executions: 0,
              failures: 0,
              averageDurationMs: 0,
            },
          })),
        );
      } catch (error) {
        if (!cancelled) {
          setBootError(error instanceof Error ? error.message : String(error));
        }
      }
    };
    void boot();
    return () => {
      cancelled = true;
    };
  }, [actorId]);

  const refresh = useCallback(() => {
    if (!runtime) return;
    setSnapshot(runtime.workspace.snapshot());
  }, []);

  const run = useCallback(
    async <T,>(label: string, action: (ws: AlphaWorkspace) => Promise<T>): Promise<T | null> => {
      const active = runtime?.workspace;
      if (!active) return null;
      setBusy(label);
      try {
        const result = await action(active);
        return result;
      } catch (error) {
        console.error(`[alpha] ${label} failed`, error);
        return null;
      } finally {
        setBusy(null);
        setSnapshot(active.snapshot());
      }
    },
    [],
  );

  /** Real training, chunked so the browser stays responsive. */
  const train = useCallback(
    async (options: Partial<TrainingConfig> = {}): Promise<TrainingSummary | null> => {
      const active = runtime?.workspace;
      if (!active || active.isTraining) return null;
      const totalSteps = Number(options.totalSteps ?? 60);
      setBusy("training");
      setTraining({
        running: true,
        step: 0,
        totalSteps,
        loss: null,
        validationLoss: null,
        uniformLoss: null,
        history: [],
        summary: null,
      });
      const history: { step: number; loss: number }[] = [];
      let uniformLoss: number | null = null;
      let lastValidation: number | null = null;
      try {
        const iterator = active.train({ totalSteps, ...options });
        let next = await iterator.next();
        while (!next.done) {
          const event = next.value;
          if (event.type === "start") {
            uniformLoss = event.uniformLoss;
            setTraining((previous) => (previous ? { ...previous, uniformLoss: event.uniformLoss } : previous));
          }
          if (event.type === "step") {
            history.push({ step: event.point.step, loss: event.point.loss });
            setTraining({
              running: true,
              step: event.point.step,
              totalSteps,
              loss: event.point.loss,
              validationLoss: lastValidation,
              uniformLoss,
              history: [...history],
              summary: null,
            });
            // Yield every few steps so React can paint and the page stays alive.
            if (event.point.step % 2 === 0) await nextTick();
          }
          if (event.type === "eval") {
            lastValidation = event.evaluation.loss;
            setTraining((previous) =>
              previous ? { ...previous, validationLoss: event.evaluation.loss } : previous,
            );
          }
          next = await iterator.next();
        }
        const summary = next.value;
        setTraining({
          running: false,
          step: active.snapshot().training.step,
          totalSteps,
          loss: summary.lastLoss,
          validationLoss: summary.validationLoss,
          uniformLoss: summary.uniformLossBaseline,
          history: [...history],
          summary,
        });
        return summary;
      } catch (error) {
        console.error("[alpha] training failed", error);
        setTraining((previous) => (previous ? { ...previous, running: false } : previous));
        return null;
      } finally {
        setBusy(null);
        setSnapshot(active.snapshot());
      }
    },
    [],
  );

  const resumeFromStored = useCallback(async () => {
    const active = runtime?.workspace;
    if (!active) return null;
    setBusy("resuming");
    try {
      const row = await convex.query(api.alpha.training.latestCheckpoint, { sessionToken: requireToken() });
      if (!row) return null;
      const checkpoint: AlphaCheckpoint = {
        id: row.checkpointId,
        label: row.label,
        modelName: row.modelName,
        modelVersion: row.modelVersion,
        config: row.config as AlphaCheckpoint["config"],
        tokenizerVersion: active.snapshot().tokenizer.version,
        datasetName: row.datasetName,
        datasetLicense: row.datasetLicense,
        step: row.step,
        tokensSeen: row.tokensSeen,
        learningRate: row.learningRate,
        metrics: {
          trainLoss: row.trainLoss,
          validationLoss: row.validationLoss ?? null,
          validationPerplexity: null,
        },
        weights: JSON.parse(row.weights) as AlphaCheckpoint["weights"],
        optimizer: JSON.parse(row.optimizer) as AlphaCheckpoint["optimizer"],
        rng: row.rng as AlphaCheckpoint["rng"],
        createdAt: row.createdAt,
        sizeBytes: row.sizeBytes,
        stage: row.stage as AlphaCheckpoint["stage"],
        notes: [`Restored from Convex at ${new Date().toLocaleString()}.`],
      };
      active.resumeFrom(checkpoint);
      return checkpoint;
    } catch (error) {
      console.error("[alpha] resume failed", error);
      return null;
    } finally {
      setBusy(null);
      setSnapshot(active.snapshot());
    }
  }, [convex]);

  const generate = useCallback(
    async (prompt: string, sampling: Partial<SamplingConfig> = {}, mode: "generate" | "rag" = "generate") => {
      const active = runtime?.workspace;
      if (!active || !prompt.trim()) return null;
      setBusy(mode === "rag" ? "retrieving" : "generating");
      setStreaming("");
      const id = `turn_${Date.now()}`;
      try {
        if (mode === "rag") {
          const answer = active.ask(prompt, sampling);
          setTurns((previous) => [
            {
              id,
              prompt,
              answer: answer.answer,
              result: answer.generation,
              sources: answer.sources,
              mode,
              at: Date.now(),
            },
            ...previous,
          ]);
          await recordTurn({
            prompt,
            answer: answer.answer,
            mode,
            result: answer.generation,
            sources: answer.sources,
          });
          setStreaming(null);
          return answer;
        }
        const stream = active.streamGenerate(prompt, sampling);
        let next = await stream.next();
        while (!next.done) {
          setStreaming(next.value.text);
          await nextTick();
          next = await stream.next();
        }
        const result = next.value;
        if (result) {
          setTurns((previous) => [
            { id, prompt, answer: result.text, result, sources: [], mode, at: Date.now() },
            ...previous,
          ]);
          await recordTurn({ prompt, answer: result.text, mode, result, sources: [] });
        }
        setStreaming(null);
        return result;
      } catch (error) {
        console.error("[alpha] generation failed", error);
        setStreaming(null);
        return null;
      } finally {
        setBusy(null);
        setSnapshot(active.snapshot());
      }
    },
    [recordTurn],
  );

  const ingest = useCallback(
    async (input: { title: string; content: string; license?: string }): Promise<IngestedDocument | null> =>
      run("ingesting", (ws) => ws.ingestDocument(input)),
    [run],
  );

  const removeDocument = useCallback(
    (documentId: string): Promise<number | null> => run("rag", async (ws) => ws.removeDocument(documentId)),
    [run],
  );

  const runAgent = useCallback(
    async (goal: string, options: { maxSteps?: number; preferModelPlanner?: boolean } = {}) => {
      const result = await run("agent", (ws) => ws.runAgent(goal, options));
      if (result) {
        setAgentRuns((previous) => [
          {
            id: result.id,
            goal: result.goal,
            status: result.status,
            steps: result.plan.steps.length,
            toolCalls: result.toolCalls.map((call) => ({
              tool: call.tool,
              ok: call.ok,
              durationMs: call.durationMs,
            })),
            synthesis: result.synthesis.text,
            method: result.synthesis.method,
            durationMs: result.durationMs,
            blocker: result.blocker,
          },
          ...previous,
        ]);
      }
      return result;
    },
    [run],
  );

  const writeMemory = useCallback(
    (input: {
      scope: MemoryScope;
      key: string;
      content: string;
      importance?: number;
      approved?: boolean;
      sessionId?: string | null;
    }): Promise<MemoryRecord | null> => run("memory", async (ws) => ws.writeMemory(input)),
    [run],
  );

  const forgetMemory = useCallback(
    (id: string): Promise<boolean | null> => run("memory", async (ws) => ws.forgetMemory(id)),
    [run],
  );

  const recall = useCallback(
    (query: string, options: { scopes?: MemoryScope[]; topK?: number } = {}): Promise<MemoryRetrieval[] | null> =>
      run("recall", async (ws) => ws.recall(query, options)),
    [run],
  );

  const approveTool = useCallback(
    (toolName: string): Promise<string | null> => run("approval", async (ws) => ws.approveTool(toolName)),
    [run],
  );

  const runWorkflow = useCallback(
    (workflowId: string, payload: Record<string, unknown> = {}) =>
      run("workflow", (ws) => ws.runWorkflow(workflowId, payload)),
    [run],
  );

  const registerWorkflow = useCallback(
    (input: Parameters<AlphaWorkspace["registerWorkflow"]>[0]) =>
      run("workflow", async (ws) => ws.registerWorkflow(input)),
    [run],
  );

  const persistAll = useCallback(() => run("persisting", (ws) => ws.persist("all")), [run]);

  return useMemo(
    () => ({
      actorId,
      workspace,
      snapshot,
      training,
      busy,
      bootError,
      turns,
      agentRuns,
      streaming,
      conversations,
      deleteConversation,
      dataset: ALPHA_SEED_CORPUS,
      datasetInfo: datasetStats(ALPHA_SEED_CORPUS),
      ready: Boolean(workspace && snapshot),
      refresh,
      train,
      resumeFromStored,
      generate,
      ingest,
      removeDocument,
      runAgent,
      writeMemory,
      forgetMemory,
      recall,
      approveTool,
      runWorkflow,
      registerWorkflow,
      persistAll,
    }),
    [
      actorId,
      workspace,
      snapshot,
      training,
      busy,
      bootError,
      turns,
      agentRuns,
      streaming,
      conversations,
      deleteConversation,
      refresh,
      train,
      resumeFromStored,
      generate,
      ingest,
      removeDocument,
      runAgent,
      writeMemory,
      forgetMemory,
      recall,
      approveTool,
      runWorkflow,
      registerWorkflow,
      persistAll,
    ],
  );
}

export type AlphaRuntime = ReturnType<typeof useAlpha>;
