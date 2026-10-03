/**
 * useChat — the browser side of Alpha's chat product.
 *
 * The page never generates anything itself. It calls Alpha's Convex chat API,
 * which runs the serving runtime (the Step 5 artefact) server-side, and it
 * subscribes to the same tables the backend writes: the conversation list, the
 * transcript, and the live stream row. Everything shown here is stored server
 * output — there is no optimistic or fabricated text.
 */

import { api } from "@/convex/_generated/api";
import { useAuth } from "@/hooks/use-auth";
import { useAction, useMutation, useQuery } from "convex/react";
import type { FunctionReturnType } from "convex/server";
import { useCallback, useEffect, useMemo, useState } from "react";

/** The sampling settings the composer sends with a turn. */
export type ChatSettings = {
  temperature: number;
  maxNewTokens: number;
  deterministic: boolean;
  useMemory: boolean;
  useRetrieval: boolean;
};

export const DEFAULT_CHAT_SETTINGS: ChatSettings = {
  temperature: 0.8,
  maxNewTokens: 64,
  deterministic: false,
  useMemory: true,
  useRetrieval: true,
};

export type ChatMessage = NonNullable<FunctionReturnType<typeof api.alpha.conversations.get>>["messages"][number];
export type ChatStream = NonNullable<FunctionReturnType<typeof api.alpha.chat.activeStream>>;
export type ChatConversation = FunctionReturnType<typeof api.alpha.conversations.list>[number];
export type ModelStatus = Awaited<FunctionReturnType<typeof api.alpha.chat.modelStatus>>;
export type ChatTurnOutcome = Awaited<FunctionReturnType<typeof api.alpha.chat.send>>;

const CONVERSATION_KEY = "alpha.chat.conversation";

function storedConversationId(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(CONVERSATION_KEY);
  } catch {
    return null;
  }
}

/** Pull the server's own message out of a Convex error, however it is wrapped. */
export function chatErrorMessage(error: unknown): string {
  const data = (error as { data?: unknown } | null)?.data;
  if (data && typeof data === "object" && "message" in data) {
    const message = (data as { message?: unknown }).message;
    if (typeof message === "string" && message.length > 0) return message;
  }
  if (error instanceof Error && error.message) {
    return (
      error.message
        .replace(/^\[CONVEX[^\]]*\]\s*/i, "")
        .replace(/^Server Error\s*/i, "")
        .replace(/^Uncaught (Convex)?Error:\s*/i, "") || "Alpha could not complete that request."
    );
  }
  return "Alpha could not complete that request.";
}

export function useChat() {
  const { sessionToken } = useAuth();
  const [conversationId, setConversationId] = useState<string | null>(() => storedConversationId());
  const [generating, setGenerating] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>(DEFAULT_CHAT_SETTINGS);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [modelStatusError, setModelStatusError] = useState<string | null>(null);

  const conversations = useQuery(
    api.alpha.conversations.list,
    sessionToken ? { sessionToken, limit: 40 } : "skip",
  );
  const transcript = useQuery(
    api.alpha.conversations.get,
    sessionToken && conversationId ? { sessionToken, conversationId } : "skip",
  );
  const stream = useQuery(
    api.alpha.chat.activeStream,
    sessionToken && conversationId ? { sessionToken, conversationId } : "skip",
  );

  const startConversation = useMutation(api.alpha.conversations.start);
  const renameConversation = useMutation(api.alpha.conversations.rename);
  const removeConversation = useMutation(api.alpha.conversations.remove);
  const clearMessages = useMutation(api.alpha.chat.clearMessages);
  const stopGeneration = useMutation(api.alpha.chat.stop);
  const sendAction = useAction(api.alpha.chat.send);
  const regenerateAction = useAction(api.alpha.chat.regenerate);
  const modelStatusAction = useAction(api.alpha.chat.modelStatus);

  // The model's own description of itself, loaded once per mount.
  useEffect(() => {
    let cancelled = false;
    void modelStatusAction({})
      .then((value) => {
        if (!cancelled) {
          setModelStatus(value);
          setModelStatusError(null);
        }
      })
      .catch((error) => {
        if (!cancelled) setModelStatusError(chatErrorMessage(error));
      });
    return () => {
      cancelled = true;
    };
  }, [modelStatusAction]);

  // Remember the open conversation across reloads.
  useEffect(() => {
    try {
      if (conversationId) window.localStorage.setItem(CONVERSATION_KEY, conversationId);
      else window.localStorage.removeItem(CONVERSATION_KEY);
    } catch {
      // A browser that blocks storage still works; the selection just resets.
    }
  }, [conversationId]);

  // A conversation from another account (or a deleted one) resolves to null.
  // That is treated as "no conversation" without rewriting state during render;
  // the stale id is also dropped from storage so a reload does not retry it.
  const activeConversationId = transcript === null ? null : conversationId;
  useEffect(() => {
    if (conversationId && transcript === null) {
      try {
        window.localStorage.removeItem(CONVERSATION_KEY);
      } catch {
        // Storage that is unavailable changes nothing here.
      }
    }
  }, [conversationId, transcript]);

  const selectConversation = useCallback((id: string | null) => {
    setConversationId(id);
  }, []);

  const newConversation = useCallback(() => {
    setConversationId(null);
  }, []);

  const send = useCallback(
    async (text: string): Promise<ChatTurnOutcome> => {
      if (!sessionToken) throw new Error("Sign in to chat with Alpha.");
      const message = text.trim();
      if (!message) throw new Error("Write a message first.");
      setGenerating(true);
      try {
        let target = activeConversationId;
        if (!target) {
          const started = await startConversation({ sessionToken, kind: "chat" });
          target = started.conversationId;
          setConversationId(target);
        }
        return (await sendAction({
          sessionToken,
          conversationId: target,
          message,
          settings,
        })) as ChatTurnOutcome;
      } finally {
        setGenerating(false);
      }
    },
    [sessionToken, activeConversationId, startConversation, sendAction, settings],
  );

  const regenerate = useCallback(async (): Promise<ChatTurnOutcome> => {
    if (!sessionToken) throw new Error("Sign in to chat with Alpha.");
    if (!activeConversationId) throw new Error("Open a conversation first.");
    setGenerating(true);
    try {
      return (await regenerateAction({
        sessionToken,
        conversationId: activeConversationId,
        settings,
      })) as ChatTurnOutcome;
    } finally {
      setGenerating(false);
    }
  }, [sessionToken, activeConversationId, regenerateAction, settings]);

  const stop = useCallback(async () => {
    if (!sessionToken || !stream || stream.status !== "streaming") return;
    await stopGeneration({ sessionToken, streamId: stream.streamId });
  }, [sessionToken, stream, stopGeneration]);

  const clear = useCallback(async () => {
    if (!sessionToken || !activeConversationId) return;
    await clearMessages({ sessionToken, conversationId: activeConversationId });
  }, [sessionToken, activeConversationId, clearMessages]);

  const rename = useCallback(
    async (id: string, title: string) => {
      if (!sessionToken) return;
      await renameConversation({ sessionToken, conversationId: id, title });
    },
    [sessionToken, renameConversation],
  );

  const remove = useCallback(
    async (id: string) => {
      if (!sessionToken) return;
      await removeConversation({ sessionToken, conversationId: id });
      if (id === activeConversationId) setConversationId(null);
    },
    [sessionToken, removeConversation, activeConversationId],
  );

  const messages = useMemo(() => transcript?.messages ?? [], [transcript]);
  const streaming = stream?.status === "streaming";

  return {
    sessionToken,
    conversationId: activeConversationId,
    conversation: transcript?.conversation ?? null,
    conversations: conversations ?? [],
    conversationsLoaded: conversations !== undefined,
    messages,
    transcriptLoaded: transcript !== undefined,
    stream: stream ?? null,
    streaming,
    generating,
    settings,
    setSettings,
    modelStatus,
    modelStatusError,
    selectConversation,
    newConversation,
    send,
    regenerate,
    stop,
    clear,
    rename,
    remove,
    errorText: chatErrorMessage,
  };
}
