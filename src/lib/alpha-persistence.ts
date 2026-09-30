/**
 * Alpha ⇄ Convex persistence adapter.
 *
 * Alpha's workspace owns the model, the checkpoints and the vectors; Convex is
 * where that work is stored so it survives a reload. This file is the only
 * place the two meet, and it implements `AlphaPersistenceAdapter` exactly — no
 * extra behaviour, no shadowing of Alpha's own logic.
 */

import type { AlphaPersistenceAdapter, PersistedRunInput } from "@/alpha";
import type { AlphaCheckpoint } from "@/alpha";
import type { AlphaTrainingJob } from "@/alpha";
import { checkpointToJson } from "@/alpha";
import type { AlphaModelArtifact } from "@/alpha";
import type { AlphaTokenizerSnapshot } from "@/alpha";
import type { AuditRecord } from "@/alpha";
import type { JobExecution } from "@/alpha";
import type { MemoryRecord } from "@/alpha";
import type { Span } from "@/alpha";
import type { VectorRecord } from "@/alpha";
import type { AlphaWorkflow } from "@/alpha";

type Mutations = {
  recordModel: (args: {
    name: string;
    version: string;
    stage: string;
    parameterCount: number;
    config: unknown;
    checkpointId?: string;
    trainedTokens?: number;
    validationLoss?: number;
    notes: string[];
  }) => Promise<unknown>;
  saveTokenizer: (args: {
    version: string;
    trainedOn: string;
    vocabSize: number;
    mergeSteps: number;
    documents: number;
    characters: number;
    snapshot: unknown;
  }) => Promise<unknown>;
  saveCheckpoint: (args: {
    checkpointId: string;
    label: string;
    modelName: string;
    modelVersion: string;
    datasetName: string;
    datasetLicense: string;
    stage: string;
    step: number;
    tokensSeen: number;
    trainLoss: number;
    validationLoss?: number;
    learningRate: number;
    sizeBytes: number;
    parameterCount: number;
    weights: string;
    optimizer: string;
    rng: unknown;
    config: unknown;
    runId?: string;
    seed?: number;
    datasetVersion?: string;
    tokenizerFingerprint?: string;
    formatVersion?: string;
    checkpoint?: string;
  }) => Promise<unknown>;
  saveTrainingJob: (args: {
    jobId: string;
    state: string;
    modelName: string;
    modelVersion: string;
    tokenizerVersion: string;
    tokenizerFingerprint: string;
    datasetName: string;
    datasetVersion: string;
    datasetFingerprint: string;
    datasetLicense: string;
    corpusTokens: number;
    corpusDocuments: number;
    config: unknown;
    seed: number;
    step: number;
    totalSteps: number;
    epochs?: number;
    tokensSeen: number;
    trainLoss?: number;
    bestLoss?: number;
    validationLoss?: number;
    learningRate?: number;
    checkpointIds: string[];
    lastCheckpointId?: string;
    resumedFromCheckpointId?: string;
    resumes: number;
    error?: string;
    startedAt?: number;
    completedAt?: number;
  }) => Promise<unknown>;
  saveVectors: (args: {
    collection: string;
    dimension: number;
    records: {
      recordId: string;
      text: string;
      embedding: number[];
      metadata: unknown;
      sourceId?: string;
    }[];
  }) => Promise<unknown>;
  saveMemories: (args: {
    records: {
      memoryId: string;
      scope: string;
      sessionId?: string;
      key: string;
      content: string;
      embedding: number[];
      tags: string[];
      importance: number;
      approved: boolean;
      source: string;
      accessCount: number;
      createdAt: number;
      updatedAt: number;
    }[];
  }) => Promise<unknown>;
  saveSpans: (args: {
    spans: {
      traceId: string;
      spanId: string;
      parentId?: string;
      name: string;
      kind: string;
      module: string;
      status: string;
      startMs: number;
      durationMs?: number;
      attributes: unknown;
    }[];
  }) => Promise<unknown>;
  saveAudit: (args: {
    records: {
      recordId: string;
      actor: string;
      module: string;
      action: string;
      resource?: string;
      decision: string;
      reason: string;
      traceId?: string;
      data: unknown;
      hash: string;
      prevHash: string;
      at: number;
    }[];
  }) => Promise<unknown>;
  recordRun: (args: {
    kind: string;
    status: string;
    traceId: string;
    modelStage: string;
    input: string;
    output: string;
    metrics: unknown;
    error?: string;
  }) => Promise<unknown>;
  saveWorkflow: (args: {
    workflowId: string;
    name: string;
    description: string;
    trigger: unknown;
    conditions: unknown;
    actions: unknown;
    enabled: boolean;
    actor: string;
  }) => Promise<unknown>;
  saveJob: (args: {
    jobId: string;
    workflowId: string;
    workflowName: string;
    status: string;
    attempts: number;
    outcomes: unknown;
    errors: string[];
    durationMs: number;
    traceId: string;
  }) => Promise<unknown>;
  syncTools: (args: {
    tools: {
      name: string;
      description: string;
      module: string;
      permission: string;
      requiresApproval: boolean;
      dangerous: boolean;
      source: string;
      inputSchema: unknown;
      stats: unknown;
    }[];
  }) => Promise<unknown>;
};

/** Convex validators reject NaN/Infinity, so an unmeasured metric is omitted. */
function finite(value: number | null): number | undefined {
  return value !== null && Number.isFinite(value) ? value : undefined;
}

/** Best-effort: persistence problems must never break an Alpha run. */
async function safe(action: () => Promise<unknown>): Promise<void> {
  try {
    await action();
  } catch (error) {
    console.warn("[alpha] persistence skipped:", error instanceof Error ? error.message : error);
  }
}

export function createConvexPersistence(mutations: Mutations): AlphaPersistenceAdapter {
  return {
    saveModel: (artifact: AlphaModelArtifact) =>
      safe(() =>
        mutations.recordModel({
          name: artifact.name,
          version: artifact.version,
          stage: artifact.stage,
          parameterCount: artifact.parameterCount,
          config: artifact.config,
          checkpointId: artifact.checkpointId,
          trainedTokens: artifact.trainedTokens,
          validationLoss: artifact.validationLoss,
          notes: artifact.notes,
        }),
      ),

    saveTokenizer: (snapshot: AlphaTokenizerSnapshot) =>
      safe(() =>
        mutations.saveTokenizer({
          version: snapshot.version,
          trainedOn: snapshot.trainedOn,
          vocabSize: snapshot.tokens.length,
          mergeSteps: snapshot.stats.mergeSteps,
          documents: snapshot.stats.documents,
          characters: snapshot.stats.characters,
          snapshot,
        }),
      ),

    saveCheckpoint: (checkpoint: AlphaCheckpoint) =>
      safe(() =>
        mutations.saveCheckpoint({
          checkpointId: checkpoint.id,
          label: checkpoint.label,
          modelName: checkpoint.modelName,
          modelVersion: checkpoint.modelVersion,
          datasetName: checkpoint.datasetName,
          datasetLicense: checkpoint.datasetLicense,
          stage: checkpoint.stage,
          step: checkpoint.step,
          tokensSeen: checkpoint.tokensSeen,
          trainLoss: Number.isFinite(checkpoint.metrics.trainLoss) ? checkpoint.metrics.trainLoss : 0,
          validationLoss: checkpoint.metrics.validationLoss ?? undefined,
          learningRate: checkpoint.learningRate,
          sizeBytes: checkpoint.sizeBytes,
          parameterCount: Object.values(checkpoint.weights.shapes).reduce(
            (sum, shape) => sum + shape.reduce((a, b) => a * b, 1),
            0,
          ),
          weights: JSON.stringify(checkpoint.weights),
          optimizer: JSON.stringify(checkpoint.optimizer),
          rng: checkpoint.rng,
          config: checkpoint.config,
          runId: checkpoint.runId,
          seed: checkpoint.seed,
          datasetVersion: checkpoint.datasetVersion,
          tokenizerFingerprint: checkpoint.tokenizer.fingerprint,
          formatVersion: checkpoint.formatVersion,
          // The whole document, so a resume restores the real artefact instead
          // of a lookalike reassembled from columns.
          checkpoint: checkpointToJson(checkpoint),
        }),
      ),

    saveTrainingJob: (job: AlphaTrainingJob) =>
      safe(() =>
        mutations.saveTrainingJob({
          jobId: job.id,
          state: job.state,
          modelName: job.modelName,
          modelVersion: job.modelVersion,
          tokenizerVersion: job.tokenizerVersion,
          tokenizerFingerprint: job.tokenizerFingerprint,
          datasetName: job.datasetName,
          datasetVersion: job.datasetVersion,
          datasetFingerprint: job.datasetFingerprint,
          datasetLicense: job.datasetLicense,
          corpusTokens: job.corpusTokens,
          corpusDocuments: job.corpusDocuments,
          config: job.config,
          seed: job.seed,
          step: job.step,
          totalSteps: job.totalSteps,
          epochs: job.epochs ?? undefined,
          tokensSeen: job.tokensSeen,
          trainLoss: finite(job.trainLoss),
          bestLoss: finite(job.bestLoss),
          validationLoss: finite(job.validationLoss),
          learningRate: finite(job.learningRate),
          checkpointIds: job.checkpointIds,
          lastCheckpointId: job.lastCheckpointId ?? undefined,
          resumedFromCheckpointId: job.resumedFromCheckpointId ?? undefined,
          resumes: job.resumes,
          error: job.error ?? undefined,
          startedAt: job.startedAt ?? undefined,
          completedAt: job.completedAt ?? undefined,
        }),
      ),

    saveVectors: (records: VectorRecord[]) => {
      if (records.length === 0) return Promise.resolve();
      const collection = records[0].collection;
      const dimension = records[0].vector.length;
      // Chunked so a large ingest does not exceed a single mutation's limits.
      const chunks: VectorRecord[][] = [];
      for (let i = 0; i < records.length; i += 40) chunks.push(records.slice(i, i + 40));
      return safe(async () => {
        for (const chunk of chunks) {
          await mutations.saveVectors({
            collection,
            dimension,
            records: chunk.map((record) => ({
              recordId: record.id,
              text: record.text,
              embedding: record.vector,
              metadata: record.metadata,
              sourceId: record.sourceId,
            })),
          });
        }
      });
    },

    saveMemories: (records: MemoryRecord[]) =>
      safe(() =>
        mutations.saveMemories({
          records: records.map((record) => ({
            memoryId: record.id,
            scope: record.scope,
            sessionId: record.sessionId ?? undefined,
            key: record.key,
            content: record.content,
            embedding: record.embedding,
            tags: record.tags,
            importance: record.importance,
            approved: record.approved,
            source: record.source,
            accessCount: record.accessCount,
            createdAt: record.createdAt,
            updatedAt: record.updatedAt,
          })),
        }),
      ),

    saveSpans: (spans: Span[]) =>
      safe(() =>
        mutations.saveSpans({
          spans: spans.map((span) => ({
            traceId: span.traceId,
            spanId: span.id,
            parentId: span.parentId ?? undefined,
            name: span.name,
            kind: span.kind,
            module: span.module,
            status: span.status,
            startMs: span.startMs,
            durationMs: span.durationMs ?? undefined,
            attributes: span.attributes,
          })),
        }),
      ),

    saveAudit: (records: AuditRecord[]) =>
      safe(() =>
        mutations.saveAudit({
          records: records.map((record) => ({
            recordId: record.id,
            actor: record.actor,
            module: record.module,
            action: record.action,
            resource: record.resource ?? undefined,
            decision: record.decision,
            reason: record.reason,
            traceId: record.traceId ?? undefined,
            data: record.data,
            hash: record.hash,
            prevHash: record.prevHash,
            at: record.at,
          })),
        }),
      ),

    saveRun: (run: PersistedRunInput) =>
      safe(() =>
        mutations.recordRun({
          kind: run.kind,
          status: run.status,
          traceId: run.traceId,
          modelStage: run.modelStage,
          input: run.input,
          output: run.output,
          metrics: run.metrics,
          error: run.error ?? undefined,
        }),
      ),

    saveWorkflow: (workflow: AlphaWorkflow) =>
      safe(() =>
        mutations.saveWorkflow({
          workflowId: workflow.id,
          name: workflow.name,
          description: workflow.description,
          trigger: workflow.trigger,
          conditions: workflow.conditions,
          actions: workflow.actions,
          enabled: workflow.enabled,
          actor: workflow.actorId,
        }),
      ),

    saveJob: (job: JobExecution) =>
      safe(() =>
        mutations.saveJob({
          jobId: job.id,
          workflowId: job.workflowId,
          workflowName: job.workflowName,
          status: job.status,
          attempts: job.attempts,
          outcomes: job.outcomes,
          errors: job.errors,
          durationMs: job.durationMs,
          traceId: job.traceId,
        }),
      ),
  };
}

/** Mirrors the tool registry so the tool surface is reviewable across sessions. */
export function createToolSync(mutations: Mutations) {
  return (tools: Parameters<Mutations["syncTools"]>[0]["tools"]) => safe(() => mutations.syncTools({ tools }));
}
