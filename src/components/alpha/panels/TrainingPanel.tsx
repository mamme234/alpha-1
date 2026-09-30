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
import { EmptyNote, Eyebrow, Frame, KeyValue, Mono, Pill, Stat, StatGrid, WarningNote } from "@/components/alpha/studio";
import { formatBytes, trainingJobStateLabel, type TrainingConfig } from "@/alpha";
import type { AlphaRuntime } from "@/hooks/use-alpha";
import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useState } from "react";
import { Loader2, Pause, Play, RotateCcw, ShieldCheck, Square } from "lucide-react";
import { toast } from "sonner";

/** Loss values are printed to four places, or as an em dash when unmeasured. */
const loss = (value: number | null | undefined) => (value === null || value === undefined ? "—" : value.toFixed(4));

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
  const job = alpha.trainingJob;
  const jobSummary = alpha.jobSummary;
  const verification = alpha.verification;
  const corpus = alpha.corpusReport ?? snapshot?.corpusReport ?? null;
  const resources = alpha.resources ?? snapshot?.resources ?? null;

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

  /** A pause is a request: the trainer honours it at the next step boundary. */
  const pause = () => {
    alpha.pauseTraining();
    toast.message("Pause requested — it takes effect at the next optimiser step");
  };

  const stop = () => {
    alpha.stopTraining();
    toast.message("Stop requested — the run writes a checkpoint and ends");
  };

  const continueRun = async () => {
    const summary = await alpha.resumeTraining();
    if (!summary) {
      toast.error("The run could not continue");
      return;
    }
    toast.success(`Run ${summary.state} at step ${summary.steps} (loss ${loss(summary.lastLoss)})`);
  };

  const runVerification = async () => {
    const report = await alpha.verify();
    if (!report) {
      toast.error("Verification could not run");
      return;
    }
    const failed = report.checks.filter((check) => !check.passed).map((check) => check.id);
    if (report.passed) {
      toast.success(`Verification passed: all ${report.checks.length} checks (A–I) on a fresh model instance`);
    } else {
      toast.error(`Verification failed: ${failed.join(", ")}`);
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
            {job?.state === "running" ? (
              <Button variant="outline" size="sm" onClick={pause}>
                <Pause className="mr-2 size-3.5" />
                Pause
              </Button>
            ) : null}
            {job?.state === "paused" ? (
              <Button variant="outline" size="sm" onClick={continueRun} disabled={Boolean(alpha.busy)}>
                <Play className="mr-2 size-3.5" />
                Continue run
              </Button>
            ) : null}
            {job && (job.state === "running" || job.state === "paused") ? (
              <Button variant="outline" size="sm" onClick={stop}>
                <Square className="mr-2 size-3.5" />
                Stop
              </Button>
            ) : null}
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
          title="Run record"
          status={job ? (job.state === "completed" ? "ready" : job.state === "failed" ? "untrained" : "in-development") : "untrained"}
          lede="One record per run: what was trained, on which corpus and tokenizer, from which checkpoint, with what result. Written at every step and kept in Convex."
          actions={job ? <Pill>{trainingJobStateLabel(job.state)}</Pill> : null}
        >
          {job ? (
            <div>
              <KeyValue label="run id">{job.id}</KeyValue>
              <KeyValue label="model">{`${job.modelName} ${job.modelVersion}`}</KeyValue>
              <KeyValue label="tokenizer">{`${job.tokenizerVersion} · ${job.tokenizerFingerprint}`}</KeyValue>
              <KeyValue label="dataset">{`${job.datasetName}@${job.datasetVersion} · ${job.datasetLicense}`}</KeyValue>
              <KeyValue label="dataset fingerprint">{job.datasetFingerprint}</KeyValue>
              <KeyValue label="seed">{job.seed}</KeyValue>
              <KeyValue label="step">{`${job.step} / ${job.totalSteps}`}</KeyValue>
              <KeyValue label="tokens seen">{job.tokensSeen.toLocaleString()}</KeyValue>
              <KeyValue label="epochs over corpus">{job.epochs === null ? "—" : job.epochs.toFixed(3)}</KeyValue>
              <KeyValue label="train loss">{loss(job.trainLoss)}</KeyValue>
              <KeyValue label="best loss">{loss(job.bestLoss)}</KeyValue>
              <KeyValue label="validation loss">{loss(job.validationLoss)}</KeyValue>
              <KeyValue label="checkpoints">{job.checkpointIds.length}</KeyValue>
              <KeyValue label="resumed from">{job.resumedFromCheckpointId ?? "— new run"}</KeyValue>
              <KeyValue label="resumes">{job.resumes}</KeyValue>
              {job.error ? (
                <div className="mt-4">
                  <WarningNote>This run failed: {job.error}</WarningNote>
                </div>
              ) : null}
              <ul className="mt-4 space-y-1">
                {job.notes.map((note) => (
                  <li key={note} className="text-[11px] leading-4 text-muted-foreground">
                    · {note}
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <EmptyNote>
              No run has been started in this session. The record is created when you press Train Alpha, and every checkpoint it
              writes quotes its run id.
            </EmptyNote>
          )}
          {jobSummary ? <p className="mt-4"><Mono>{jobSummary}</Mono></p> : null}
        </Frame>

        <Frame
          title="Verification — checks A–I"
          status={verification ? (verification.passed ? "ready" : "untrained") : "in-development"}
          lede="Deterministic checks on a fresh instance of the configured architecture: serialisation, loss agreement against an independent computation, numerical gradients, checkpoint round-trip, resume continuity and greedy decoding."
          actions={
            <Button variant="outline" size="sm" onClick={runVerification} disabled={Boolean(alpha.busy)}>
              <ShieldCheck className="mr-2 size-3.5" />
              Run verification
            </Button>
          }
        >
          {verification ? (
            <div className="space-y-4">
              <StatGrid>
                <Stat
                  label="Result"
                  value={verification.passed ? "PASS" : "FAIL"}
                  hint={`${verification.checks.filter((check) => check.passed).length} of ${verification.checks.length} checks`}
                />
                <Stat label="Steps in run" value={verification.training.steps} hint={`seed ${verification.training.seed}`} />
                <Stat
                  label="Loss"
                  value={`${loss(verification.training.firstLoss)} → ${loss(verification.training.lastLoss)}`}
                  hint={`validation ${loss(verification.training.validationLoss)}`}
                />
                <Stat
                  label="Uniform baseline"
                  value={verification.training.uniformLoss.toFixed(4)}
                  hint="the loss to beat"
                />
              </StatGrid>
              <ul className="space-y-2">
                {verification.checks.map((check) => (
                  <li key={check.id} className="flex items-start gap-3 border-b border-border/60 pb-2 last:border-b-0">
                    <Pill className={check.passed ? "" : "border-destructive/50 text-destructive"}>{check.id}</Pill>
                    <div className="space-y-0.5">
                      <p className="text-xs text-foreground">{check.label}</p>
                      <p className="text-[11px] leading-4 text-muted-foreground">{check.detail}</p>
                    </div>
                  </li>
                ))}
              </ul>
              <p className="text-[11px] leading-4 text-muted-foreground">
                Verification trains its own weights from the configured seed; the model this session is holding is never touched by
                it, so a passing report does not claim the live weights were verified.
              </p>
            </div>
          ) : (
            <EmptyNote>
              Verification has not run in this session. Running it takes a few seconds and reports every check, including the ones it
              fails.
            </EmptyNote>
          )}
        </Frame>
      </div>

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
        <Frame
          title="Dataset"
          status="ready"
          lede="What Alpha is trained on, with its licence recorded next to it — and what the trainer will actually consume, measured from the corpus rather than estimated."
        >
          <div className="grid gap-6 md:grid-cols-3">
            <div>
              <Eyebrow>Corpus</Eyebrow>
              <div className="mt-2">
                <KeyValue label="name">{snapshot.corpus.name}</KeyValue>
                <KeyValue label="version">{snapshot.corpus.version}</KeyValue>
                <KeyValue label="licence">{snapshot.corpus.license}</KeyValue>
                <KeyValue label="documents">{snapshot.corpus.documents}</KeyValue>
                <KeyValue label="characters">{snapshot.corpus.characters.toLocaleString()}</KeyValue>
                {corpus ? <KeyValue label="fingerprint">{corpus.fingerprint}</KeyValue> : null}
              </div>
            </div>
            <div>
              <Eyebrow>Split</Eyebrow>
              <div className="mt-2">
                <KeyValue label="train tokens">{snapshot.corpus.trainTokens.toLocaleString()}</KeyValue>
                <KeyValue label="validation tokens">{snapshot.corpus.validationTokens.toLocaleString()}</KeyValue>
                <KeyValue label="train examples">{corpus ? corpus.trainExamples.toLocaleString() : "—"}</KeyValue>
                <KeyValue label="validation examples">{corpus ? corpus.validationExamples.toLocaleString() : "—"}</KeyValue>
                <KeyValue label="unknown tokens">{corpus ? corpus.unknownTokens.toLocaleString() : "—"}</KeyValue>
              </div>
            </div>
            <div>
              <Eyebrow>Engine</Eyebrow>
              <div className="mt-2">
                <KeyValue label="objective">next-token cross-entropy</KeyValue>
                <KeyValue label="optimiser">AdamW (decoupled weight decay)</KeyValue>
                <KeyValue label="sequence length">{corpus ? corpus.sequenceLength : snapshot.model.config.contextLength}</KeyValue>
                <KeyValue label="batch size">{corpus ? corpus.batchSize : "—"}</KeyValue>
                <KeyValue label="padding">
                  {corpus ? (corpus.padded ? `${corpus.paddingTokensPerBatch ?? 0} tokens/batch` : "none — random windows") : "—"}
                </KeyValue>
              </div>
            </div>
          </div>
          {corpus && corpus.unknownTokens > 0 ? (
            <p className="mt-4 text-[11px] leading-4 text-muted-foreground">
              {corpus.unknownTokens.toLocaleString()} tokens fell outside the trained alphabet and are recorded as <Mono>{"<unk>"}</Mono>
              . They are counted here rather than dropped quietly.
            </p>
          ) : null}
        </Frame>
      ) : null}

      {resources ? (
        <Frame
          title="Resource envelope"
          status="ready"
          lede="Alpha's hard ceilings and the memory a run of the current shape needs. Nothing here is a hosted model budget: it is what this tab allocates."
        >
          <div className="grid gap-6 md:grid-cols-4">
            <Stat label="Parameters" value={resources.estimate.parameterCount.toLocaleString()} hint="from the config, not a pretrained file" />
            <Stat label="Weights" value={formatBytes(resources.estimate.weightsBytes)} hint="float32" />
            <Stat label="Optimiser" value={formatBytes(resources.estimate.optimizerBytes)} hint="AdamW moments" />
            <Stat label="Estimated peak" value={formatBytes(resources.estimate.totalBytes)} hint="weights + gradients + moments + one batch of activations" />
          </div>
          <p className="mt-4 text-[11px] leading-4 text-muted-foreground">{resources.estimate.note}</p>
          <div className="mt-4 grid gap-x-6 gap-y-3 md:grid-cols-3">
            <KeyValue label="max sequence">{resources.limits.maxSeqLen}</KeyValue>
            <KeyValue label="max batch">{resources.limits.maxBatchSize}</KeyValue>
            <KeyValue label="max parameters">{resources.limits.maxParameterCount.toLocaleString()}</KeyValue>
            <KeyValue label="max documents">{resources.limits.maxDocuments.toLocaleString()}</KeyValue>
            <KeyValue label="max document characters">{resources.limits.maxDocumentCharacters.toLocaleString()}</KeyValue>
            <KeyValue label="max new tokens">{resources.limits.maxNewTokens}</KeyValue>
          </div>
        </Frame>
      ) : null}
    </div>
  );
}
