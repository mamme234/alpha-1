/**
 * Alpha chat — the product page.
 *
 * One screen: the stored conversations, the transcript with its provenance, a
 * composer that calls Alpha's own backend, and the served model's own status.
 * The page contains no model and no generation logic; it is a client of the
 * `alpha.chat` API, and every token it renders came from the server, streamed
 * through the `alphaStreams` table.
 */

import { ConversationList } from "@/components/alpha/chat/conversations";
import { ModelCard } from "@/components/alpha/chat/model-card";
import { ChatThread } from "@/components/alpha/chat/thread";
import { Pill } from "@/components/alpha/studio";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useAuth } from "@/hooks/use-auth";
import { useChat } from "@/hooks/use-chat";
import { cn } from "@/lib/utils";
import {
  Loader2,
  Menu,
  PanelLeft,
  RotateCcw,
  SendHorizonal,
  Settings2,
  Square,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { toast } from "sonner";

export default function Chat() {
  const navigate = useNavigate();
  const { signOut } = useAuth();
  const chat = useChat();

  const [draft, setDraft] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const lastMessage = chat.messages[chat.messages.length - 1];
  const canRegenerate = !chat.streaming && !chat.generating && lastMessage?.role === "assistant";
  const waiting = chat.generating && !chat.streaming;

  useEffect(() => {
    if (!chat.streaming) textareaRef.current?.focus();
  }, [chat.streaming, chat.conversationId]);

  const submit = useCallback(async () => {
    const value = draft.trim();
    if (!value || chat.generating) return;
    setDraft("");
    try {
      const outcome = await chat.send(value);
      if (outcome.status === "error") {
        toast.error(outcome.error ?? "Alpha could not finish that answer.");
      } else if (outcome.status === "stopped") {
        toast.message("Generation stopped. The partial answer was kept.");
      }
    } catch (error) {
      toast.error(chat.errorText(error));
      setDraft(value);
    }
  }, [draft, chat]);

  const regenerate = useCallback(async () => {
    try {
      const outcome = await chat.regenerate();
      if (outcome.status === "error") {
        toast.error(outcome.error ?? "Alpha could not regenerate that answer.");
      }
    } catch (error) {
      toast.error(chat.errorText(error));
    }
  }, [chat]);

  const status = chat.modelStatus;
  const statusLabel =
    status === null
      ? "checking model…"
      : status.available
        ? `${status.info.modelName}@${status.info.modelVersion}`
        : "model unavailable";

  const sidebar = (
    <ConversationList
      conversations={chat.conversations}
      activeId={chat.conversationId}
      loaded={chat.conversationsLoaded}
      onSelect={(id) => {
        chat.selectConversation(id);
        setSidebarOpen(false);
      }}
      onNew={() => {
        chat.newConversation();
        setSidebarOpen(false);
      }}
      onRename={async (id, title) => {
        try {
          await chat.rename(id, title);
          toast.success("Conversation renamed.");
        } catch (error) {
          toast.error(chat.errorText(error));
        }
      }}
      onDelete={async (id) => {
        try {
          await chat.remove(id);
          toast.success("Conversation deleted.");
        } catch (error) {
          toast.error(chat.errorText(error));
        }
      }}
    />
  );

  return (
    <div className="flex h-[100dvh] flex-col bg-background text-foreground">
      <header className="z-30 border-b border-border bg-background/90 backdrop-blur">
        <div className="mx-auto flex w-full max-w-[110rem] items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="lg:hidden" aria-label="Conversations">
                  <Menu className="size-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-80">
                <SheetHeader>
                  <SheetTitle className="studio-serif text-left">Conversations</SheetTitle>
                  <SheetDescription className="text-left">
                    Stored transcripts, owned by your account.
                  </SheetDescription>
                </SheetHeader>
                <div className="mt-4 h-[calc(100dvh-8rem)]">{sidebar}</div>
              </SheetContent>
            </Sheet>
            <button
              type="button"
              onClick={() => navigate("/")}
              className="flex items-baseline gap-2 text-left"
            >
              <span className="studio-serif text-lg tracking-tight">Alpha</span>
              <span className="studio-eyebrow hidden sm:inline">chat</span>
            </button>
            <Popover>
              <PopoverTrigger asChild>
                <button type="button" className="ml-1">
                  <Pill
                    className={cn(
                      status?.available === true && "border-chart-2/40 text-chart-2",
                      status?.available === false && "border-destructive/40 text-destructive",
                    )}
                  >
                    <span className="hidden sm:inline">
                      {status?.available === true ? `● ${statusLabel}` : statusLabel}
                    </span>
                    <span className="sm:hidden">●</span>
                  </Pill>
                </button>
              </PopoverTrigger>
              <PopoverContent className="w-80" align="start">
                <ModelCard status={chat.modelStatus} error={chat.modelStatusError} compact />
              </PopoverContent>
            </Popover>
          </div>

          <div className="flex items-center gap-2">
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="ghost" size="icon" aria-label="Generation settings">
                  <Settings2 className="size-4" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-72" align="end">
                <div className="space-y-4">
                  <p className="studio-eyebrow">Generation settings</p>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-xs">temperature</Label>
                      <span className="font-mono text-[11px] text-muted-foreground">
                        {chat.settings.temperature.toFixed(2)}
                      </span>
                    </div>
                    <Slider
                      value={[chat.settings.temperature]}
                      min={0}
                      max={2}
                      step={0.05}
                      onValueChange={([value]) =>
                        chat.setSettings({ ...chat.settings, temperature: value ?? 0.8 })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-xs">max new tokens</Label>
                      <span className="font-mono text-[11px] text-muted-foreground">
                        {chat.settings.maxNewTokens}
                      </span>
                    </div>
                    <Slider
                      value={[chat.settings.maxNewTokens]}
                      min={8}
                      max={64}
                      step={4}
                      onValueChange={([value]) =>
                        chat.setSettings({ ...chat.settings, maxNewTokens: value ?? 64 })
                      }
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-xs">deterministic (greedy)</Label>
                    <Switch
                      checked={chat.settings.deterministic}
                      onCheckedChange={(checked) => chat.setSettings({ ...chat.settings, deterministic: checked })}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-xs">recall memory</Label>
                    <Switch
                      checked={chat.settings.useMemory}
                      onCheckedChange={(checked) => chat.setSettings({ ...chat.settings, useMemory: checked })}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label className="text-xs">search ingested corpus</Label>
                    <Switch
                      checked={chat.settings.useRetrieval}
                      onCheckedChange={(checked) => chat.setSettings({ ...chat.settings, useRetrieval: checked })}
                    />
                  </div>
                  <p className="text-[10px] leading-4 text-muted-foreground">
                    Settings travel with the next turn and are stored on the assistant message that used them.
                  </p>
                </div>
              </PopoverContent>
            </Popover>

            <Button
              variant="outline"
              size="sm"
              className="hidden sm:inline-flex"
              onClick={() => navigate("/dashboard")}
            >
              <PanelLeft className="mr-2 size-4" /> Studio
            </Button>

            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                void signOut().then(() => navigate("/"));
              }}
            >
              Sign out
            </Button>
          </div>
        </div>
      </header>

      <div className="mx-auto flex min-h-0 w-full max-w-[110rem] flex-1">
        <aside className="hidden w-72 shrink-0 border-r border-border px-4 py-5 lg:block">
          {sidebar}
        </aside>

        <main className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-2 sm:px-6">
            <div className="min-w-0">
              <p className="truncate text-xs text-foreground">
                {chat.conversation?.title ?? "New conversation"}
              </p>
              <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
                {chat.conversationId ? `stored · ${chat.conversationId}` : "nothing stored yet"}
              </p>
            </div>
            <div className="flex items-center gap-2">
              {canRegenerate ? (
                <Button variant="outline" size="sm" onClick={regenerate}>
                  <RotateCcw className="mr-2 size-3.5" /> Regenerate
                </Button>
              ) : null}
              {chat.conversationId && chat.messages.length > 0 ? (
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={chat.streaming}
                  onClick={() => setConfirmClear(true)}
                >
                  <Trash2 className="mr-2 size-3.5" /> Clear
                </Button>
              ) : null}
            </div>
          </div>

          <ChatThread
            messages={chat.messages}
            stream={chat.stream}
            streaming={chat.streaming}
            waiting={waiting}
            transcriptLoaded={chat.transcriptLoaded}
            onSuggestion={(text) => {
              setDraft(text);
              textareaRef.current?.focus();
            }}
          />

          <div className="border-t border-border px-4 py-4 sm:px-6">
            <div className="mx-auto w-full max-w-3xl">
              <div className="flex items-end gap-2 studio-frame p-2">
                <Textarea
                  ref={textareaRef}
                  value={draft}
                  rows={2}
                  placeholder="Ask Alpha — short prompts fit its 256-token window best."
                  className="min-h-[44px] resize-none border-0 bg-transparent shadow-none focus-visible:ring-0"
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      void submit();
                    }
                  }}
                />
                {chat.streaming ? (
                  <Button variant="outline" onClick={() => void chat.stop()}>
                    <Square className="mr-2 size-3.5" /> Stop
                  </Button>
                ) : (
                  <Button disabled={chat.generating || draft.trim().length === 0} onClick={() => void submit()}>
                    {chat.generating ? (
                      <Loader2 className="mr-2 size-4 animate-spin" />
                    ) : (
                      <SendHorizonal className="mr-2 size-4" />
                    )}
                    Send
                  </Button>
                )}
              </div>
              <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
                <span>enter to send · shift+enter for a new line</span>
                <span>
                  {chat.streaming
                    ? "generating on the server"
                    : "256-token window · answers come from Alpha's own weights"}
                </span>
              </div>
            </div>
          </div>
        </main>

        <aside className="hidden w-80 shrink-0 overflow-y-auto border-l border-border px-5 py-6 xl:block">
          <ModelCard status={chat.modelStatus} error={chat.modelStatusError} />
        </aside>
      </div>

      <AlertDialog open={confirmClear} onOpenChange={setConfirmClear}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear this conversation?</AlertDialogTitle>
            <AlertDialogDescription>
              Every message in this conversation is deleted. The conversation itself stays.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep the messages</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                void chat
                  .clear()
                  .then(() => toast.success("Conversation cleared."))
                  .catch((error) => toast.error(chat.errorText(error)));
              }}
            >
              Clear
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
