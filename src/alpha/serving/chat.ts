/**
 * Chat turns over the serving runtime.
 *
 * This is the layer between a chat product and `AlphaServingRuntime`. It owns
 * the rules that only make sense for a model this small:
 *
 *   - the context window is 96 tokens, so a request that cannot fit is refused
 *     with a real number ("this message is 74 tokens; 53 are available") rather
 *     than silently truncated;
 *   - history is capped and passed as data; the context engine may still trim
 *     it, and the turn's context report says exactly what survived;
 *   - tools are a fixed allow-list of real built-in tools — the caller can
 *     narrow it, never widen it;
 *   - nothing here invents a response. A failed generation returns the error
 *     from the runtime untouched.
 */

import { AlphaValidationError } from "../core/errors";
import type { SamplingConfig } from "../inference/engine";
import type {
  RespondRequest,
  RespondResult,
  RespondStreamEvent,
  RespondStreamOptions,
} from "../runtime/orchestrator";
import { ALPHA_CHAT_TOOL_ALLOWLIST, type AlphaServingRuntime } from "./runtime";

export type ChatHistoryTurn = { role: "user" | "assistant"; content: string };

export type ChatTurnSettings = Partial<SamplingConfig> & {
  /** Recall approved memory before answering. Defaults to on. */
  useMemory?: boolean;
  /** Search the ingested corpus. Defaults to on when the account has one. */
  useRetrieval?: boolean;
  /** Tools permitted this turn. Defaults to the chat allow-list. */
  allowedTools?: string[];
};

export type ChatTurnInput = {
  actorId: string;
  conversationId: string;
  sessionId?: string | null;
  message: string;
  history?: ChatHistoryTurn[];
  settings?: ChatTurnSettings;
  cancellation?: RespondStreamOptions["cancellation"];
  shouldStop?: RespondStreamOptions["shouldStop"];
  shouldStopEvery?: number;
};

/** Only the most recent turns are offered to the context engine. */
export const CHAT_MAX_HISTORY_TURNS = 6;

/** Build the runtime request for one chat turn, or refuse it with a reason. */
export function buildChatRequest(
  runtime: AlphaServingRuntime,
  input: ChatTurnInput,
): RespondRequest {
  const message = typeof input.message === "string" ? input.message.trim() : "";
  if (message === "") {
    throw new AlphaValidationError("inference", "a chat turn needs a non-empty message");
  }
  const tokens = runtime.context.countTokens(message);
  const maxTokens = runtime.limits.maxRequestTokens;
  if (tokens > maxTokens) {
    throw new AlphaValidationError(
      "inference",
      `this message is ${tokens} tokens; the serving model's ${runtime.limits.contextLength}-token window leaves at most ${maxTokens} tokens for a request once its instruction is counted. Shorten the message and try again.`,
      { tokens, maxRequestTokens: maxTokens, contextLength: runtime.limits.contextLength },
    );
  }

  const history = (input.history ?? [])
    .filter((turn) => typeof turn.content === "string" && turn.content.trim() !== "")
    .slice(-CHAT_MAX_HISTORY_TURNS);

  return {
    message,
    actorId: input.actorId,
    conversationId: input.conversationId,
    sessionId: input.sessionId ?? null,
    history,
    useMemory: input.settings?.useMemory ?? true,
    useRetrieval: input.settings?.useRetrieval ?? runtime.documentVectorCount > 0,
    allowedTools: input.settings?.allowedTools ?? [...ALPHA_CHAT_TOOL_ALLOWLIST],
    sampling: samplingFrom(input.settings),
  };
}

/** One-shot turn. Used by callers that do not stream (and by verification). */
export function runChatTurn(
  runtime: AlphaServingRuntime,
  input: ChatTurnInput,
): Promise<RespondResult> {
  return runtime.respond(buildChatRequest(runtime, input));
}

/** Streamed turn: same request, same pipeline, deltas as they are produced. */
export function runChatTurnStream(
  runtime: AlphaServingRuntime,
  input: ChatTurnInput,
): AsyncGenerator<RespondStreamEvent, RespondResult, void> {
  const request = buildChatRequest(runtime, input);
  return runtime.respondStream(request, {
    cancellation: input.cancellation ?? null,
    shouldStop: input.shouldStop ?? null,
    shouldStopEvery: input.shouldStopEvery ?? 12,
  });
}

/** Pull only the sampling fields out of a settings object. */
function samplingFrom(settings: ChatTurnSettings | undefined): Partial<SamplingConfig> {
  if (!settings) return {};
  const out: Partial<SamplingConfig> = {};
  const {
    temperature,
    topK,
    topP,
    maxNewTokens,
    repetitionPenalty,
    stopSequences,
    stopTokenIds,
    deterministic,
    seed,
  } = settings;
  if (temperature !== undefined) out.temperature = temperature;
  if (topK !== undefined) out.topK = topK;
  if (topP !== undefined) out.topP = topP;
  if (maxNewTokens !== undefined) out.maxNewTokens = maxNewTokens;
  if (repetitionPenalty !== undefined) out.repetitionPenalty = repetitionPenalty;
  if (stopSequences !== undefined) out.stopSequences = stopSequences;
  if (stopTokenIds !== undefined) out.stopTokenIds = stopTokenIds;
  if (deterministic !== undefined) out.deterministic = deterministic;
  if (seed !== undefined) out.seed = seed;
  return out;
}
