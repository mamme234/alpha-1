/**
 * The conversation list: stored transcripts, not browser-local history.
 *
 * Renaming and deletion go through Alpha's own API (session-scoped, ownership
 * checked server-side), and deletion asks first because it removes the
 * transcript's messages with it.
 */

import { cn } from "@/lib/utils";
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { ChatConversation } from "@/hooks/use-chat";
import { MessageSquarePlus, MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import { useState } from "react";

function formatWhen(timestamp: number): string {
  const minutes = Math.max(0, Math.round((Date.now() - timestamp) / 60000));
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

function ConversationItem({
  conversation,
  active,
  onSelect,
  onRename,
  onDelete,
}: {
  conversation: ChatConversation;
  active: boolean;
  onSelect: () => void;
  onRename: (title: string) => Promise<void>;
  onDelete: () => Promise<void>;
}) {
  const [renaming, setRenaming] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [title, setTitle] = useState(conversation.title);
  const [busy, setBusy] = useState(false);

  return (
    <div
      className={cn(
        "group flex items-center gap-1 rounded-sm border border-transparent px-2 py-2 transition-colors",
        active ? "border-border bg-accent/70" : "hover:bg-muted/70",
      )}
    >
      <button type="button" className="min-w-0 flex-1 text-left" onClick={onSelect}>
        <span className="block truncate text-xs text-foreground">{conversation.title}</span>
        <span className="mt-0.5 block text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
          {conversation.messageCount} message{conversation.messageCount === 1 ? "" : "s"} · {formatWhen(conversation.updatedAt)}
        </span>
      </button>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="size-7 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
            aria-label={`Actions for ${conversation.title}`}
          >
            <MoreHorizontal className="size-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem
            onSelect={() => {
              setTitle(conversation.title);
              setRenaming(true);
            }}
          >
            <Pencil className="mr-2 size-3.5" /> Rename
          </DropdownMenuItem>
          <DropdownMenuItem className="text-destructive" onSelect={() => setConfirming(true)}>
            <Trash2 className="mr-2 size-3.5" /> Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog open={renaming} onOpenChange={setRenaming}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle className="studio-serif">Rename conversation</DialogTitle>
            <DialogDescription>The new title is stored with the transcript.</DialogDescription>
          </DialogHeader>
          <Input
            value={title}
            maxLength={120}
            onChange={(event) => setTitle(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && title.trim()) {
                setRenaming(false);
                void onRename(title.trim());
              }
            }}
          />
          <DialogFooter>
            <Button
              disabled={!title.trim() || busy}
              onClick={() => {
                setBusy(true);
                void onRename(title.trim()).finally(() => {
                  setBusy(false);
                  setRenaming(false);
                });
              }}
            >
              Rename
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={confirming} onOpenChange={setConfirming}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete “{conversation.title}”?</AlertDialogTitle>
            <AlertDialogDescription>
              This removes the conversation and every message in it. It cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep it</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                setBusy(true);
                void onDelete().finally(() => setBusy(false));
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

export function ConversationList({
  conversations,
  activeId,
  loaded,
  onSelect,
  onNew,
  onRename,
  onDelete,
}: {
  conversations: ChatConversation[];
  activeId: string | null;
  loaded: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, title: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}) {
  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <Button variant="outline" className="w-full justify-start" onClick={onNew}>
        <MessageSquarePlus className="mr-2 size-4" /> New conversation
      </Button>
      <ScrollArea className="-mx-2 flex-1">
        <div className="space-y-1 px-2 pb-4">
          {!loaded ? (
            <p className="px-2 text-xs text-muted-foreground">Loading conversations…</p>
          ) : conversations.length === 0 ? (
            <p className="px-2 text-xs leading-5 text-muted-foreground">
              No conversations yet. Ask Alpha something and the turn is stored here for good.
            </p>
          ) : (
            conversations.map((conversation) => (
              <ConversationItem
                key={conversation.conversationId}
                conversation={conversation}
                active={conversation.conversationId === activeId}
                onSelect={() => onSelect(conversation.conversationId)}
                onRename={(title) => onRename(conversation.conversationId, title)}
                onDelete={() => onDelete(conversation.conversationId)}
              />
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
