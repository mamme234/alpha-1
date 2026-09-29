/**
 * Training — Alpha learning, live.
 *
 * The loss curve is plotted from the trainer's own history in this browser tab.
 * Nothing is seeded with illustrative values: if the curve is empty, no
 * training has happened yet.
 */

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { EmptyNote, Eyebrow, Frame, KeyValue, Mono, Stat, StatGrid } from "@/components/alpha/studio";
import { formatBytes, type TrainingConfig } from "@/alpha";
import type { AlphaRuntime } from "@/hooks/use-alpha";
import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useState } from "react";
import { Loader2, Play, RotateCcw } from "lucide-react";
import { toast } from "sonner";

export function TrainingPanel({ alpha }: { alpha: AlphaRuntime }) {
  const snapshot = alpha.snapshot;
  const [steps, setSteps] = useState(60);
  const [batchSize, setBatchSize] = useState(8);
  const [seqLen, setSeqLen] = useState(32);
  const [learningRate, setLearningRate] = useState(0.003);
  const [evalInterval, setEvalInterval] = useState(15);

  const progress = alpha.training;
  const history = progress?.history ?? snapshot?.training.history.map((point) => ({ step: point.step, loss: point.loss })) ?? [];
  const uniformLoss = progress?.uniformLoss ?? (snapshot ? Math.log(snapshot.tokenizer.vocabSize) : null);

  const startTraining = async () => {
    const overrides: Partial<TrainingConfig> = {
      totalSteps: steps,
      batchSize,
      seqLen,
      learningRate,
      warmupSteps: Math.max(1, Math.round(steps * 0.1)),
      evalInterval,
      evalBatches: 4,
      checkpointInterval: Math.max(10, Math.round(steps / 2)),
    };
    const summary = await alpha.train(overrides);
    if (summary) {
      toast.success(
        `Trained ${summary.steps} steps — loss ${summary.lastLoss?.toFixed(3)} (uniform baseline ${summary.uniformLossBaseline.toFixed(3)})`,
      );
    } else {
      toast.error("Training did not complete");
    }
  };

  const resume = async () => {
    const checkpoint = await alpha.resumeFromStored();
    if (checkpoint) {
      toast.success(`Resumed at step ${checkpoint.step} from the stored checkpoint`);
    } else {
      toast.error("No stored checkpoint was found");
    }
  };

  return (
    <div className="space-y-6">
      <Frame
        title="Training engine"
        status={snapshot?.statuses.training}
        lede="Corpus → token window → forward pass → cross-entropy → backpropagation through Alpha's own autodiff → AdamW. Real gradients, real updates, in this tab."
        actions={
          <>
            <Button variant="outline" size="sm" onClick={resume} disabled={Boolean(alpha.busy)}>
              <RotateCcw className="mr-2 size-3.5" />
              Resume from store
            </Button>
            <Button size="sm" onClick={startTraining} disabled={Boolean(alpha.busy)}>
              {progress?.running ? <Loader2 className="mr-2 size-3.5 animate-spin" /> : <Play className="mr-2 size-3.5" />}
              {progress?.running ? "Training…" : "Train Alpha"}
            </Button>
          </>
        }
      >
        <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label htmlFor="steps" className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                  Steps
                </Label>
                <Input
                  id="steps"
                  type="number"
                  min={5}
                  max={400}
                  value={steps}
                  onChange={(event) => setSteps(Number(event.target.value))}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="batch" className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                  Batch
                </Label>
                <Input
                  id="batch"
                  type="number"
                  min={1}
                  max={16}
                  value={batchSize}
                  onChange={(event) => setBatchSize(Number(event.target.value))}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="seq" className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                  Sequence
                </Label>
                <Input
                  id="seq"
                  type="number"
                  min={8}
                  max={snapshot?.model.config.contextLength ?? 64}
                  value={seqLen}
                  onChange={(event) => setSeqLen(Number(event.target.value))}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="lr" className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                  Learning rate
                </Label>
                <Input
                  id="lr"
                  type="number"
                  step={0.0005}
                  min={0.0001}
                  max={0.05}
                  value={learningRate}
                  onChange={(event) => setLearningRate(Number(event.target.value))}
                />
              </div>
            </div>
            <div className="space-y-1">
              <Label htmlFor="eval" className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                Validate every N steps
              </Label>
              <Input
                id="eval"
                type="number"
                min={0}
                max={100}
                value={evalInterval}
                onChange={(event) => setEvalInterval(Number(event.target.value))}
              />
            </div>
            <p className="text-[11px] leading-4 text-muted-foreground">
              Training runs in slices and yields to the browser between steps, so the curve below updates while it works.
              Longer runs are slower, not fake — every step is a real forward and backward pass.
            </p>
          </div>

          <div className="space-y-4">
            <div className="rounded-md border border-border bg-muted/20 p-4">
              {history.length === 0 ? (
                <EmptyNote>
                  No loss recorded yet. Press <span className="text-foreground">Train Alpha</span> to run real gradient
                  descent over the seed corpus and watch the curve appear.
                </EmptyNote>
              ) : (
                <div className="h-56 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history} margin={{ top: 8, right: 12, bottom: 4, left: -12 }}>
                      <XAxis
                        dataKey="step"
                        tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                        stroke="var(--color-border)"
                      />
                      <YAxis
                        tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                        stroke="var(--color-border)"
                        domain={["auto", "auto"]}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "var(--color-popover)",
                          border: "1px solid var(--color-border)",
                          borderRadius: 6,
                          fontSize: 11,
                        }}
                        labelFormatter={(value) => `step ${value}`}
                        formatter={(value: number) => [value.toFixed(4), "train loss"]}
                      />
                      {uniformLoss ? (
                        <ReferenceLine
                          y={uniformLoss}
                          stroke="var(--color-chart-5)"
                          strokeDasharray="4 4"
                          label={{
                            value: "uniform baseline",
                            position: "insideTopRight",
                            fontSize: 10,
                            fill: "var(--color-muted-foreground)",
                          }}
                        />
                      ) : null}
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="var(--color-chart-1)"
                        strokeWidth={1.5}
                        dot={false}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            <StatGrid>
              <Stat
                label="Progress"
                value={`${progress?.step ?? snapshot?.training.step ?? 0} / ${progress?.totalSteps ?? snapshot?.training.totalSteps ?? 0}`}
                hint={progress?.running ? "running" : "idle"}
              />
              <Stat
                label="Train loss"
                value={progress?.loss !== null && progress?.loss !== undefined ? progress.loss.toFixed(4) : "—"}
                hint="nats per token"
              />
              <Stat
                label="Validation loss"
                value={progress?.validationLoss !== null && progress?.validationLoss !== undefined ? progress.validationLoss.toFixed(4) : "—"}
                hint="held-out split"
              />
              <Stat
                label="Uniform baseline"
                value={uniformLoss ? uniformLoss.toFixed(4) : "—"}
                hint="a predictor that guesses uniformly"
              />
            </StatGrid>
          </div>
        </div>
      </Frame>

      <div className="grid gap-6 lg:grid-cols-2">
        <Frame
          title="Tokenizer"
          status={snapshot?.statuses.tokenizer}
          lede="Alpha's own BPE vocabulary, trained from the corpus in the browser and stored with its merge table."
        >
          {snapshot ? (
            <div>
              <KeyValue label="version">{snapshot.tokenizer.version}</KeyValue>
              <KeyValue label="vocabulary">{snapshot.tokenizer.vocabSize} tokens</KeyValue>
              <KeyValue label="merge operations">{snapshot.tokenizer.mergeSteps}</KeyValue>
              <KeyValue label="trained on">{snapshot.tokenizer.trainedOn}</KeyValue>
              <KeyValue label="documents">{snapshot.tokenizer.documents}</KeyValue>
              <KeyValue label="characters">{snapshot.tokenizer.characters.toLocaleString()}</KeyValue>
              <KeyValue label="special tokens">
                {"<pad> <unk> <bos> <eos>"}
              </KeyValue>
            </div>
          ) : (
            <EmptyNote>Tokenizer is training.</EmptyNote>
          )}
          <p className="mt-4 text-[11px] leading-4 text-muted-foreground">
            Characters outside the trained alphabet map to <Mono>{"<unk>"}</Mono> and are counted, never silently dropped.
          </p>
        </Frame>

        <Frame
          title="Checkpoints"
          status={snapshot?.training.checkpoint ? "ready" : "untrained"}
          lede="A checkpoint is the unit of 'Alpha has actually trained'. It carries weights, optimiser moments, the RNG position and the metrics measured so far."
        >
          {snapshot?.training.checkpoint ? (
            <div>
              <KeyValue label="id">{snapshot.training.checkpoint.id}</KeyValue>
              <KeyValue label="step">{snapshot.training.checkpoint.step}</KeyValue>
              <KeyValue label="tokens seen">{snapshot.training.checkpoint.tokensSeen.toLocaleString()}</KeyValue>
              <KeyValue label="train loss">{snapshot.training.checkpoint.metrics.trainLoss.toFixed(4)}</KeyValue>
              <KeyValue label="validation loss">
                {snapshot.training.checkpoint.metrics.validationLoss?.toFixed(4) ?? "—"}
              </KeyValue>
              <KeyValue label="payload">{formatBytes(snapshot.training.checkpoint.sizeBytes)}</KeyValue>
              <KeyValue label="stage">{snapshot.training.checkpoint.stage}</KeyValue>
              <ul className="mt-4 space-y-1">
                {snapshot.training.checkpoint.notes.map((note) => (
                  <li key={note} className="text-[11px] leading-4 text-muted-foreground">
                    · {note}
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <EmptyNote>
              No checkpoint yet. Until one exists Alpha is labelled UNTRAINED everywhere it appears, including in generated
              output.
            </EmptyNote>
          )}
        </Frame>
      </div>

      {snapshot?.corpus ? (
        <Frame title="Dataset" status="ready" lede="What Alpha is trained on, with its licence recorded next to it.">
          <div className="grid gap-6 md:grid-cols-2">
            <div>
              <Eyebrow>Corpus</Eyebrow>
              <div className="mt-2">
                <KeyValue label="name">{snapshot.corpus.name}</KeyValue>
                <KeyValue label="version">{snapshot.corpus.version}</KeyValue>
                <KeyValue label="licence">{snapshot.corpus.license}</KeyValue>
                <KeyValue label="documents">{snapshot.corpus.documents}</KeyValue>
                <KeyValue label="characters">{snapshot.corpus.characters.toLocaleString()}</KeyValue>
              </div>
            </div>
            <div>
              <Eyebrow>Split</Eyebrow>
              <div className="mt-2">
                <KeyValue label="train tokens">{snapshot.corpus.trainTokens.toLocaleString()}</KeyValue>
                <KeyValue label="validation tokens">{snapshot.corpus.validationTokens.toLocaleString()}</KeyValue>
                <KeyValue label="objective">next-token cross-entropy</KeyValue>
                <KeyValue label="optimiser">AdamW (decoupled weight decay)</KeyValue>
              </div>
            </div>
          </div>
        </Frame>
      ) : null}
    </div>
  );
}
