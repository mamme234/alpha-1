/**
 * Inference — Alpha generating from its own weights.
 *
 * While the model is untrained this panel says so, twice: once as a standing
 * note and once attached to every answer. Output that comes from random
 * weights must never look like a finished assistant's answer.
 */

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { EmptyNote, Eyebrow, Frame, KeyValue, Mono, Pill, Stat, StatGrid, WarningNote } from "@/components/alpha/studio";
import type { AlphaRuntime } from "@/hooks/use-alpha";
import { useState } from "react";
import { Loader2, Sparkles, SquareStack } from "lucide-react";

const PROMPTS = [
  "Alpha is a self owned",
  "The training engine stores",
  "Retrieval augmented generation",
  "Memory in Alpha is",
];

const BLOCK_STATUS_STYLES: Record<"included" | "truncated" | "dropped", string> = {
  included: "border-chart-2/35 text-chart-2",
  truncated: "border-chart-4/40 text-chart-4",
  dropped: "border-chart-3/30 text-chart-3",
};

export function InferencePanel({ alpha }: { alpha: AlphaRuntime }) {
  const snapshot = alpha.snapshot;
  const [prompt, setPrompt] = useState(PROMPTS[0]);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.95);
  const [maxNewTokens, setMaxNewTokens] = useState(40);
  const [seed, setSeed] = useState(2026);
  const [mode, setMode] = useState<"generate" | "rag">("generate");

  const sampling = { temperature, topK, topP, maxNewTokens, seed };
  const streaming = alpha.streaming;
  const untrained = snapshot?.model.stage === "untrained" || snapshot?.model.stage === "architecture";
  const assembled = snapshot?.context.last ?? null;

  const submit = async () => {
    if (!prompt.trim()) return;
    await alpha.generate(prompt, sampling, mode);
  };

  return (
    <div className="space-y-6">
      <Frame
        title="Inference engine"
        status={snapshot?.statuses.inference ?? "planned"}
        lede="Prompt → tokenizer → Alpha's transformer → temperature, top-k, top-p sampling → tokens. No provider is contacted and there is no fallback path."
        actions={
          <div className="flex items-center gap-1 rounded-md border border-border p-0.5">
            {(["generate", "rag"] as const).map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => setMode(option)}
                className={`rounded-sm px-3 py-1 text-[11px] uppercase tracking-[0.14em] transition-colors ${
                  mode === option ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {option === "generate" ? "Raw" : "With retrieval"}
              </button>
            ))}
          </div>
        }
      >
        {untrained ? (
          <WarningNote>
            These weights are random initialisation, so the tokens below are sampled from an untrained transformer. That is
            what it looks like before training — not a bug, and not a hidden model answering for Alpha.
          </WarningNote>
        ) : null}

        <div className="mt-4 grid gap-6 lg:grid-cols-[1fr_300px]">
          <div className="space-y-4">
            <Textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="Write a prompt for Alpha…"
              className="min-h-24 resize-y font-mono text-xs"
            />
            <div className="flex flex-wrap gap-2">
              {PROMPTS.map((example) => (
                <button
                  key={example}
                  type="button"
                  onClick={() => setPrompt(example)}
                  className="rounded-full border border-border px-3 py-1 text-[11px] text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground"
                >
                  {example}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={submit} disabled={Boolean(alpha.busy) || !prompt.trim()}>
                {alpha.busy === "generating" || alpha.busy === "retrieving" ? (
                  <Loader2 className="mr-2 size-3.5 animate-spin" />
                ) : mode === "rag" ? (
                  <SquareStack className="mr-2 size-3.5" />
                ) : (
                  <Sparkles className="mr-2 size-3.5" />
                )}
                {mode === "rag" ? "Retrieve and answer" : "Generate"}
              </Button>
              <span className="text-[11px] text-muted-foreground">
                {snapshot?.model.name} · {snapshot?.model.parameterCount.toLocaleString()} parameters · context{" "}
                {snapshot?.model.config.contextLength}
              </span>
            </div>

            <div className="rounded-md border border-border bg-muted/25 p-4">
              <Eyebrow>Output</Eyebrow>
              {streaming !== null ? (
                <pre className="mt-2 whitespace-pre-wrap font-mono text-xs leading-5 text-foreground">
                  {streaming}
                  <span className="ml-0.5 inline-block h-3 w-1 animate-pulse bg-foreground/60 align-middle" />
                </pre>
              ) : alpha.turns[0] ? (
                <div className="mt-2 space-y-3">
                  <pre className="whitespace-pre-wrap font-mono text-xs leading-5 text-foreground">
                    {alpha.turns[0].answer || "(empty generation — the model produced only stop tokens)"}
                  </pre>
                  {alpha.turns[0].sources.length > 0 ? (
                    <div className="space-y-2 border-t border-border pt-3">
                      <Eyebrow>Cited sources</Eyebrow>
                      {alpha.turns[0].sources.map((source) => (
                        <p key={source.chunkId} className="text-[11px] leading-4 text-muted-foreground">
                          <Mono>[{source.rank}]</Mono> {source.title} · score {source.score.toFixed(4)} ·{" "}
                          {source.excerpt.slice(0, 90)}…
                        </p>
                      ))}
                    </div>
                  ) : null}
                  {alpha.turns[0].result ? (
                    <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-border pt-3 text-[11px] text-muted-foreground">
                      <span>{alpha.turns[0].result.generatedTokens} tokens generated</span>
                      <span>{alpha.turns[0].result.promptTokens} prompt tokens</span>
                      <span>{alpha.turns[0].result.latencyMs} ms</span>
                      <span>{alpha.turns[0].result.tokensPerSecond} tok/s</span>
                      <span>stop: {alpha.turns[0].result.stopReason}</span>
                      <span>stage: {alpha.turns[0].result.modelStage}</span>
                    </div>
                  ) : null}
                  {alpha.turns[0].result?.warning ? (
                    <p className="text-[11px] leading-4 text-muted-foreground">{alpha.turns[0].result.warning}</p>
                  ) : null}
                </div>
              ) : (
                <EmptyNote>Nothing generated in this session yet.</EmptyNote>
              )}
            </div>
          </div>

          <div className="space-y-4">
            <div className="space-y-3 rounded-md border border-border p-4">
              <Eyebrow>Sampling</Eyebrow>
              <div className="space-y-1">
                <Label className="text-[11px] text-muted-foreground">Temperature · {temperature.toFixed(2)}</Label>
                <Input
                  type="number"
                  step={0.05}
                  min={0}
                  max={2}
                  value={temperature}
                  onChange={(event) => setTemperature(Number(event.target.value))}
                />
              </div>
              <div className="space-y-1">
                <Label className="text-[11px] text-muted-foreground">Top-k · {topK}</Label>
                <Input
                  type="number"
                  min={0}
                  max={200}
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                />
              </div>
              <div className="space-y-1">
                <Label className="text-[11px] text-muted-foreground">Top-p · {topP.toFixed(2)}</Label>
                <Input
                  type="number"
                  step={0.01}
                  min={0.05}
                  max={1}
                  value={topP}
                  onChange={(event) => setTopP(Number(event.target.value))}
                />
              </div>
              <div className="space-y-1">
                <Label className="text-[11px] text-muted-foreground">Max new tokens · {maxNewTokens}</Label>
                <Input
                  type="number"
                  min={1}
                  max={200}
                  value={maxNewTokens}
                  onChange={(event) => setMaxNewTokens(Number(event.target.value))}
                />
              </div>
              <div className="space-y-1">
                <Label className="text-[11px] text-muted-foreground">Seed · {seed}</Label>
                <Input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
              </div>
              <p className="text-[11px] leading-4 text-muted-foreground">
                The same seed reproduces the same sample: Alpha's sampler draws from its own deterministic generator.
              </p>
            </div>

            <div className="rounded-md border border-border p-4">
              <Eyebrow>Measured</Eyebrow>
              <div className="mt-2">
                <KeyValue label="requests">{snapshot?.inference.requests ?? 0}</KeyValue>
                <KeyValue label="tokens generated">{(snapshot?.inference.tokensGenerated ?? 0).toLocaleString()}</KeyValue>
                <KeyValue label="mean latency">{snapshot?.inference.averageLatencyMs.toFixed(1) ?? "0"} ms</KeyValue>
                <KeyValue label="p95 latency">{snapshot?.inference.p95LatencyMs.toFixed(1) ?? "0"} ms</KeyValue>
                <KeyValue label="KV cache">not implemented (re-runs the prefix)</KeyValue>
              </div>
            </div>
          </div>
        </div>
      </Frame>

      <Frame
        title="Context window"
        status={snapshot?.statuses.context ?? "planned"}
        lede="Every prompt is assembled before the model sees it. Each block reports what it asked for and what it actually got, so a trimmed prompt is visible rather than silent."
      >
        {assembled ? (
          <div className="space-y-5">
            <StatGrid>
              <Stat
                label="Model window"
                value={assembled.budgetTokens + assembled.reserveForOutput}
                hint="tokens the model can read at once"
              />
              <Stat label="Reserved for output" value={assembled.reserveForOutput} hint="held back for the answer" />
              <Stat label="Prompt budget" value={assembled.budgetTokens} hint="window minus the reserve" />
              <Stat
                label="Prompt used"
                value={assembled.usedTokens}
                hint={`${assembled.droppedTokens} token(s) left unspent`}
              />
            </StatGrid>
            <div className="space-y-2">
              {assembled.blocks.map((block) => (
                <div key={block.id} className="rounded-md border border-border px-4 py-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] uppercase tracking-[0.14em] text-foreground">{block.kind}</span>
                      <Pill className={BLOCK_STATUS_STYLES[block.status]}>{block.status}</Pill>
                    </div>
                    <Mono>
                      {block.includedTokens} / {block.requestedTokens} tokens
                    </Mono>
                  </div>
                  <p className="mt-1 text-[11px] leading-4 text-muted-foreground">{block.reason}</p>
                </div>
              ))}
            </div>
            {assembled.notes.length > 0 ? (
              <div className="space-y-1">
                <Eyebrow>Adjustments</Eyebrow>
                {assembled.notes.map((note) => (
                  <p key={note} className="font-mono text-[11px] leading-4 text-muted-foreground">
                    {note}
                  </p>
                ))}
              </div>
            ) : null}
          </div>
        ) : (
          <EmptyNote>
            No prompt assembled yet in this session. Generate once and the window report appears here.
          </EmptyNote>
        )}
        <p className="mt-4 text-[11px] leading-4 text-muted-foreground">
          This report covers raw generation. Retrieval answers fence their sources and budget them separately under{" "}
          <Mono>rag.maxContextTokens</Mono>.
        </p>
      </Frame>

      <Frame title="Session history" status="ready" lede="What Alpha produced in this tab, with the settings that produced it.">
        {alpha.turns.length === 0 ? (
          <EmptyNote>No generations yet in this session.</EmptyNote>
        ) : (
          <div className="space-y-3">
            <StatGrid className="mb-4">
              <Stat label="Answers" value={alpha.turns.length} />
              <Stat label="Retrieval answers" value={alpha.turns.filter((turn) => turn.mode === "rag").length} />
              <Stat
                label="Mean tokens"
                value={
                  alpha.turns.length
                    ? Math.round(
                        alpha.turns.reduce((sum, turn) => sum + (turn.result?.generatedTokens ?? 0), 0) / alpha.turns.length,
                      )
                    : 0
                }
              />
              <Stat
                label="Mean latency"
                value={
                  alpha.turns.length
                    ? `${Math.round(alpha.turns.reduce((sum, turn) => sum + (turn.result?.latencyMs ?? 0), 0) / alpha.turns.length)} ms`
                    : "—"
                }
              />
            </StatGrid>
            {alpha.turns.map((turn) => (
              <div key={turn.id} className="rounded-md border border-border px-4 py-3">
                <div className="flex items-center justify-between gap-3">
                  <Mono>{new Date(turn.at).toLocaleTimeString()}</Mono>
                  <Mono>{turn.mode === "rag" ? "with retrieval" : "raw generation"}</Mono>
                </div>
                <p className="mt-2 font-mono text-[11px] text-muted-foreground">› {turn.prompt}</p>
                <pre className="mt-1 whitespace-pre-wrap font-mono text-xs leading-5 text-foreground">
                  {turn.answer || "(no tokens)"}
                </pre>
              </div>
            ))}
          </div>
        )}
      </Frame>
    </div>
  );
}
