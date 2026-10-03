/**
 * The transcript, rendered.
 *
 * Assistant turns carry their provenance: the model stage that produced them,
 * how long the turn took, how many tokens it cost, why generation stopped, and
 * which request produced it. Sources that Alpha actually retrieved are listed;
 * a turn that failed says so instead of showing substitute text.
 */

import { Mono, Pill, StageBadge } from "@/components/alpha/studio";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import type { ChatMessage, ChatStream } from "@/hooks/use-chat";
import type { AlphaModelStage } from "@/alpha";
import { cn } from "@/lib/utils";
import { AlertTriangle, CornerDownRight } from "lucide-react";
import { useEffect, useRef } from "react";

function formatWhen(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function UserBubble({ message }: { message: ChatMessage }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-md rounded-br-sm bg-primary px-4 py-2.5 text-sm leading-6 text-primary-foreground">
        <p className="whitespace-pre-wrap break-words">{message.content}</p>
        <p className="mt-1 text-right text-[10px] uppercase tracking-[0.14em] opacity-60">
          {formatWhen(message.createdAt)}
        </p>
      </div>
    </div>
  );
}

function AssistantBubble({ message }: { message: ChatMessage }) {
  const failed = Boolean(message.error);
  return (
    <div className="flex justify-start">
      <div
        className={cn(
          "studio-frame max-w-[92%] px-4 py-3",
          failed && "border-destructive/40 bg-destructive/5",
        )}
      >
        {message.content ? (
          <p className="whitespace-pre-wrap break-words text-sm leading-6 text-foreground">{message.content}</p>
        ) : failed ? (
          <p className="text-sm leading-6 text-muted-foreground">
            <AlertTriangle className="mr-2 inline size-4 text-destructive" />
            Alpha produced no text for this turn.
          </p>
        ) : (
          <p className="text-sm leading-6 text-muted-foreground">(empty answer)</p>
        )}

        {message.error ? (
          <p className="mt-2 rounded-sm border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs leading-5 text-foreground">
            <span className="font-medium">generation failed:</span> {message.error}
          </p>
        ) : null}

        {message.sources.length > 0 ? (
          <ul className="mt-3 space-y-1 border-t border-border/70 pt-2">
            {message.sources.map((source, index) => (
              <li key={`${source.chunkId}-${index}`} className="flex items-baseline gap-2 text-[11px] leading-4 text-muted-foreground">
                <CornerDownRight className="size-3 shrink-0" />
                <span className="truncate">{source.title}</span>
                <Mono>{source.score.toFixed(3)}</Mono>
              </li>
            ))}
          </ul>
        ) : null}

        <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border/70 pt-2">
          {message.modelStage ? <StageBadge stage={message.modelStage as AlphaModelStage} /> : null}
          {typeof message.tokens === "number" ? <Pill>{message.tokens} tok</Pill> : null}
          {typeof message.latencyMs === "number" ? <Pill>{(message.latencyMs / 1000).toFixed(2)}s</Pill> : null}
          {message.stopReason ? <Pill>stop: {message.stopReason}</Pill> : null}
          {message.requestId ? <Pill>{message.requestId}</Pill> : null}
        </div>
      </div>
    </div>
  );
}

function LiveBubble({ stream }: { stream: ChatStream }) {
  const text = stream.text || stream.chunks.join("");
  return (
    <div className="flex justify-start">
      <div className="studio-frame max-w-[92%] border-chart-4/40 px-4 py-3">
        <p className="whitespace-pre-wrap break-words text-sm leading-6 text-foreground">
          {text || <span className="text-muted-foreground">Generating the first token…</span>}
          <span className="ml-0.5 inline-block h-4 w-[2px] animate-pulse bg-foreground align-text-bottom" />
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-border/70 pt-2">
          <Pill className="border-chart-4/40 text-chart-4">
            streaming · {stream.outputTokens} tok
          </Pill>
          {stream.promptTokens !== null ? <Pill>{stream.promptTokens} tok context</Pill> : null}
          <Pill>{stream.streamId}</Pill>
          {stream.stopRequested ? <Pill className="border-destructive/40 text-destructive">stopping…</Pill> : null}
        </div>
      </div>
    </div>
  );
}

function WaitingCard({ streaming }: { streaming: boolean }) {
  if (!streaming) return null;
  return (
    <div className="flex justify-start">
      <div className="studio-frame max-w-[92%] px-4 py-3">
        <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
          Alpha is loading its weights and assembling the turn…
        </p>
        <div className="mt-3 space-y-2">
          <Skeleton className="h-3 w-3/4" />
          <Skeleton className="h-3 w-1/2" />
        </div>
      </div>
    </div>
  );
}

function EmptyThread({ onSuggestion }: { onSuggestion: (text: string) => void }) {
  const suggestions = ["Hello", "What is 2+2?", "Who are you?"];
  return (
    <div className="studio-frame px-6 py-8 text-center">
      <p className="studio-eyebrow">A real conversation with a small model</p>
      <h2 className="studio-serif mt-2 text-2xl leading-tight">Ask Alpha something short.</h2>
      <p className="mx-auto mt-3 max-w-md text-xs leading-5 text-muted-foreground">
        Every answer is generated server-side by Alpha's own weights — the same checkpoint its verification run
        measured. The window is small, so short prompts work best; longer ones are refused with a real token count.
      </p>
      <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
        {suggestions.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            onClick={() => onSuggestion(suggestion)}
            className="rounded-full border border-border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-foreground/30 hover:text-foreground"
          >
            “{suggestion}”
          </button>
        ))}
      </div>
    </div>
  );
}

export function ChatThread({
  messages,
  stream,
  streaming,
  waiting,
  transcriptLoaded,
  onSuggestion,
}: {
  messages: ChatMessage[];
  stream: ChatStream | null;
  streaming: boolean;
  /** A turn is running but its stream row is not visible yet. */
  waiting: boolean;
  transcriptLoaded: boolean;
  onSuggestion: (text: string) => void;
}) {
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const streamText = stream?.text ?? "";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, streamText, streaming, waiting]);

  return (
    <ScrollArea className="min-h-0 flex-1">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-4 py-6">
        {!transcriptLoaded && messages.length === 0 ? (
          <div className="space-y-3">
            <Skeleton className="h-3 w-2/3" />
            <Skeleton className="h-3 w-1/2" />
          </div>
        ) : null}
        {transcriptLoaded && messages.length === 0 && !streaming && !waiting ? (
          <EmptyThread onSuggestion={onSuggestion} />
        ) : null}
        {messages.map((message) =>
          message.role === "user" ? (
            <UserBubble key={message.messageId} message={message} />
          ) : (
            <AssistantBubble key={message.messageId} message={message} />
          ),
        )}
        {streaming && stream ? <LiveBubble stream={stream} /> : null}
        {waiting && !streaming ? <WaitingCard streaming={waiting} /> : null}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
