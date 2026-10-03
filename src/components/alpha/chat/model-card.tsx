/**
 * The served model, described by the server itself.
 *
 * Every value on this card comes from `alpha.chat:modelStatus`, which reads the
 * loaded Step 5 artefact — the fingerprints, the training run, the evaluation
 * gate and the live token budget. The card states the weak part too: Alpha's
 * answers come from a real model that was trained for 32 steps, so they are
 * real and poor; the note at the bottom says exactly that.
 */

import { KeyValue, Mono, StageBadge, Stat, StatGrid, WarningNote } from "@/components/alpha/studio";
import { Skeleton } from "@/components/ui/skeleton";
import type { ModelStatus } from "@/hooks/use-chat";
import type { AlphaModelStage } from "@/alpha";

export function ModelCard({
  status,
  error,
  compact = false,
}: {
  status: ModelStatus | null;
  error: string | null;
  compact?: boolean;
}) {
  if (error) {
    return <WarningNote>Alpha could not describe its serving model: {error}</WarningNote>;
  }
  if (!status) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-3 w-48" />
        <Skeleton className="h-3 w-40" />
      </div>
    );
  }
  if (!status.available) {
    return (
      <WarningNote>
        The serving model is unavailable, so chat cannot run: <Mono>{status.error}</Mono>
      </WarningNote>
    );
  }

  const { info, limits } = status;

  return (
    <div className="space-y-5">
      <div>
        <p className="studio-eyebrow">Served model</p>
        <p className="studio-serif mt-1 text-lg leading-tight">
          {info.modelName}
          <span className="text-muted-foreground">@{info.modelVersion}</span>
        </p>
        <div className="mt-2 flex items-center gap-2">
          <StageBadge stage={info.stage as AlphaModelStage} />
          <Mono>{info.parameterCount.toLocaleString()} params</Mono>
        </div>
      </div>

      <StatGrid className={compact ? "grid-cols-2" : "grid-cols-2"}>
        <Stat label="window" value={`${info.contextLength} tok`} hint={`${limits.maxRequestTokens} usable per request`} />
        <Stat
          label="trained"
          value={`${info.trainingSteps} steps`}
          hint={`${info.trainingTokens.toLocaleString()} tokens seen`}
        />
        <Stat
          label="validation"
          value={info.validationLoss !== null ? info.validationLoss.toFixed(3) : "—"}
          hint={info.validationPerplexity !== null ? `ppl ${info.validationPerplexity.toFixed(1)}` : undefined}
        />
        <Stat
          label="gate"
          value={info.gatePassed ? "passed" : "not passed"}
          hint={`${info.suiteCases} evaluation cases`}
        />
      </StatGrid>

      <div>
        <KeyValue label="model id">{info.modelId}</KeyValue>
        <KeyValue label="weights">{info.configFingerprint}</KeyValue>
        <KeyValue label="tokenizer">{info.tokenizerFingerprint}</KeyValue>
        <KeyValue label="data">{info.datasetFingerprint}</KeyValue>
        <KeyValue label="artefact">{info.artifactFormat}</KeyValue>
        <KeyValue label="load">{info.loadMs} ms on the server</KeyValue>
        <KeyValue label="external">{info.externalModels}</KeyValue>
      </div>

      <WarningNote>
        These are Alpha's own weights — trained {info.trainingSteps} steps on{" "}
        {info.trainingTokens.toLocaleString()} tokens. Answers are real model output and they are weak: expect broken
        words and repetition. Nothing here is canned, and no external model is involved.
      </WarningNote>

      <details className="rounded-md border border-border/70 px-3 py-2">
        <summary className="cursor-pointer text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
          Tools available ({status.tools.length})
        </summary>
        <ul className="mt-2 space-y-1">
          {status.tools.map((tool) => (
            <li key={tool.name} className="text-[11px] leading-4 text-muted-foreground">
              <Mono className="text-foreground">{tool.name}</Mono> — {tool.description}
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}
