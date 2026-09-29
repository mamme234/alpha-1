/**
 * Overview — what Alpha is, in facts rather than claims.
 *
 * Every value here is read from the live workspace: the vocabulary Alpha
 * trained, the parameter count of the model it instantiated, the stage its
 * checkpoints imply, and the honest status of each subsystem.
 */

import { EmptyNote, Eyebrow, Frame, KeyValue, Pill, Stat, StatGrid, StatusBadge, StageBadge } from "@/components/alpha/studio";
import { describeArchitecture } from "@/alpha";
import type { AlphaRuntime } from "@/hooks/use-alpha";
import { formatBytes } from "@/alpha";

export function OverviewPanel({ alpha }: { alpha: AlphaRuntime }) {
  const snapshot = alpha.snapshot;
  if (!snapshot) {
    return <EmptyNote>Alpha is initialising: training its tokenizer on the bundled corpus.</EmptyNote>;
  }

  const architecture = describeArchitecture(snapshot.model.config);
  const modules = snapshot.modules;

  return (
    <div className="space-y-6">
      <Frame
        title="Alpha LLM Core"
        status={snapshot.statuses.model}
        lede="A decoder-only transformer defined, initialised and trained entirely inside this repository. No weights are downloaded and no provider is called."
        actions={<StageBadge stage={snapshot.model.stage} />}
      >
        <StatGrid>
          <Stat label="Model" value={snapshot.model.name} hint={`architecture v${snapshot.model.version}`} />
          <Stat
            label="Parameters"
            value={snapshot.model.parameterCount.toLocaleString()}
            hint="counted from the architecture, not estimated"
          />
          <Stat
            label="Layers / width"
            value={`${snapshot.model.config.nLayers} × ${snapshot.model.config.dModel}`}
            hint={`${snapshot.model.config.nHeads} attention heads`}
          />
          <Stat
            label="Context"
            value={`${snapshot.model.config.contextLength} tokens`}
            hint={`feed-forward ${snapshot.model.config.dFeedForward}`}
          />
          <Stat
            label="Vocabulary"
            value={snapshot.tokenizer.vocabSize.toLocaleString()}
            hint={`trained on ${snapshot.tokenizer.documents} documents`}
          />
          <Stat
            label="Checkpoint"
            value={snapshot.training.checkpoint ? `step ${snapshot.training.checkpoint.step}` : "none"}
            hint={
              snapshot.training.checkpoint
                ? `${snapshot.training.checkpoint.tokensSeen.toLocaleString()} tokens seen`
                : "no training run has produced one yet"
            }
          />
          <Stat
            label="Validation loss"
            value={snapshot.model.validationLoss !== null ? snapshot.model.validationLoss.toFixed(4) : "—"}
            hint="cross-entropy, nats per token"
          />
          <Stat
            label="Training tokens"
            value={snapshot.model.trainedTokens.toLocaleString()}
            hint="across all runs on this model"
          />
        </StatGrid>

        <div className="mt-6 rounded-md border border-border bg-muted/30 px-4 py-3">
          <Eyebrow>What this means right now</Eyebrow>
          <p className="mt-2 text-xs leading-5 text-muted-foreground">
            {snapshot.model.stage === "untrained"
              ? "The architecture is real and every parameter is addressable, but the weights are random initialisation. Alpha will generate text — it will be meaningless text. Run a training pass to change that, and the label changes with it."
              : `Alpha was trained from scratch on ${snapshot.corpus?.name ?? "its own corpus"} by this repository's training engine. Its competence is exactly the corpus it saw; nothing here comes from a larger external model.`}
          </p>
        </div>
      </Frame>

      <Frame
        title="Architecture"
        status="ready"
        lede="The tensors Alpha instantiates, with their real shapes and their contribution to the parameter count."
      >
        <div className="grid gap-6 md:grid-cols-2">
          <div className="space-y-4">
            <KeyValue label="positional signal">{snapshot.model.config.positionalEncoding}</KeyValue>
            <KeyValue label="output projection">
              {snapshot.model.config.tieEmbeddings ? "tied to token embedding" : "separate matrix"}
            </KeyValue>
            <KeyValue label="dropout">{snapshot.model.config.dropout}</KeyValue>
            <KeyValue label="layer norm eps">{snapshot.model.config.normEps}</KeyValue>
            <KeyValue label="initialisation std">{snapshot.model.config.initStd}</KeyValue>
            <KeyValue label="tensor count">{architecture.length}</KeyValue>
          </div>
          <div className="max-h-72 overflow-auto rounded-md border border-border">
            <table className="w-full border-collapse text-left">
              <thead className="sticky top-0 bg-muted/80 backdrop-blur">
                <tr>
                  <th className="px-3 py-2 text-[10px] font-medium uppercase tracking-[0.16em] text-muted-foreground">Tensor</th>
                  <th className="px-3 py-2 text-[10px] font-medium uppercase tracking-[0.16em] text-muted-foreground">Shape</th>
                  <th className="px-3 py-2 text-right text-[10px] font-medium uppercase tracking-[0.16em] text-muted-foreground">
                    Params
                  </th>
                </tr>
              </thead>
              <tbody>
                {architecture.map((row) => (
                  <tr key={row.name} className="border-t border-border/60">
                    <td className="px-3 py-1.5 font-mono text-[11px] text-foreground">{row.name}</td>
                    <td className="px-3 py-1.5 font-mono text-[11px] text-muted-foreground">{row.shape}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-[11px] text-muted-foreground">
                      {row.parameters.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Frame>

      <Frame
        title="Alpha stack"
        status="ready"
        lede="The manifest the workspace, the documentation and the tests all read from. A subsystem is never labelled better than it is."
      >
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {modules.map((module) => (
            <article key={module.id} className="rounded-md border border-border bg-card px-4 py-3">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="studio-serif text-sm text-foreground">{module.name}</h3>
                  <Pill className="mt-1">src/alpha/{module.id}</Pill>
                </div>
                <StatusBadge status={snapshot.statuses[module.id] ?? module.status} />
              </div>
              <p className="mt-2 text-xs leading-5 text-muted-foreground">{module.summary}</p>
              <ul className="mt-2 space-y-1">
                {module.notes.map((note) => (
                  <li key={note} className="flex gap-2 text-[11px] leading-4 text-muted-foreground">
                    <span aria-hidden className="mt-[6px] size-1 shrink-0 rounded-full bg-border" />
                    <span>{note}</span>
                  </li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </Frame>

      {snapshot.errors.length > 0 ? (
        <Frame title="Recorded problems" status="in-development" lede="Errors the workspace caught and recorded rather than hiding.">
          <ul className="space-y-2">
            {snapshot.errors.slice(0, 10).map((error, index) => (
              <li key={`${error}-${index}`} className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 font-mono text-[11px] text-foreground">
                {error}
              </li>
            ))}
          </ul>
        </Frame>
      ) : null}

      <Frame
        title="Provenance"
        status="ready"
        lede="Where Alpha's numbers come from, stated plainly."
      >
        <div className="grid gap-6 md:grid-cols-2">
          <div>
            <Eyebrow>Corpus</Eyebrow>
            <div className="mt-2">
              <KeyValue label="dataset">{snapshot.corpus?.name ?? alpha.dataset.name}</KeyValue>
              <KeyValue label="version">{snapshot.corpus?.version ?? alpha.dataset.version}</KeyValue>
              <KeyValue label="licence">{snapshot.corpus?.license ?? alpha.dataset.license}</KeyValue>
              <KeyValue label="documents">{alpha.datasetInfo.documents}</KeyValue>
              <KeyValue label="characters">{alpha.datasetInfo.characters.toLocaleString()}</KeyValue>
            </div>
          </div>
          <div>
            <Eyebrow>Artifacts</Eyebrow>
            <div className="mt-2">
              <KeyValue label="tokenizer version">{snapshot.tokenizer.version}</KeyValue>
              <KeyValue label="merge steps">{snapshot.tokenizer.mergeSteps}</KeyValue>
              <KeyValue label="checkpoint size">
                {snapshot.training.checkpoint ? formatBytes(snapshot.training.checkpoint.sizeBytes) : "—"}
              </KeyValue>
              <KeyValue label="stored in">Convex (Alpha's own tables)</KeyValue>
            </div>
          </div>
        </div>
      </Frame>
    </div>
  );
}
