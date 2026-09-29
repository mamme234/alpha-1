/**
 * Alpha Tools — built-in tools.
 *
 * Each tool is implemented here and nowhere else: there is no mock handler and
 * no sample response. Tools that need a subsystem (tokenizer, embeddings,
 * vector store, memory) read it from the execution context and fail with a
 * clear message when the host application has not supplied it yet.
 */

import { AlphaToolError, AlphaValidationError } from "../core/errors";
import type { AlphaEmbedder } from "../embeddings/embedder";
import type { AlphaMemoryStore, MemoryScope } from "../memory/store";
import type { AlphaTokenizer } from "../tokenizer/bpe";
import type { AlphaVectorStore } from "../vector/store";
import { evaluateExpression } from "./expression";
import { objectSchema } from "./schema";
import type { AlphaToolRegistry, ToolServices } from "./registry";

function service<T>(services: ToolServices, key: string): T {
  const value = services[key];
  if (!value) {
    throw new AlphaToolError(
      `this tool requires the "${key}" subsystem, which the host application has not provided`,
      { missing: key },
    );
  }
  return value as T;
}

export const BUILTIN_TOOL_NAMES = [
  "alpha.text.stats",
  "alpha.calculator",
  "alpha.tokenizer.analyze",
  "alpha.corpus.search",
  "alpha.memory.search",
  "alpha.memory.write",
  "alpha.admin.clear_vector_store",
] as const;

export function registerBuiltinTools(registry: AlphaToolRegistry): string[] {
  registry.register<{ text: string }, { characters: number; words: number; lines: number; sentences: number; longestWord: string }>({
    name: "alpha.text.stats",
    description:
      "Count characters, words, lines and sentences in a piece of text and report the longest word. Purely local string analysis.",
    module: "tools",
    inputSchema: objectSchema(
      { text: { type: "string", description: "Text to analyse", minLength: 1, maxLength: 20000 } },
      ["text"],
    ),
    permission: "tool.execute",
    source: "builtin",
    tags: ["text", "count", "measure", "analyse"],
    handler: ({ text }) => {
      const words = text.split(/\s+/).filter(Boolean);
      const sentences = text.split(/[.!?]+/).filter((part) => part.trim().length > 0);
      const longestWord = words.reduce((longest, word) => (word.length > longest.length ? word : longest), "");
      return {
        characters: text.length,
        words: words.length,
        lines: text.split("\n").length,
        sentences: sentences.length,
        longestWord,
      };
    },
    verify: (output) => ({
      ok: output.words > 0 || output.characters > 0,
      reason: output.characters > 0 ? "text measured" : "text was empty",
    }),
  });

  registry.register<{ expression: string }, { expression: string; value: number }>({
    name: "alpha.calculator",
    description:
      "Evaluate an arithmetic expression (+ - * / % ^, parentheses, sqrt/abs/min/max/round/floor/ceil/log/exp/pow). No variables, no evaluation of code.",
    module: "tools",
    inputSchema: objectSchema(
      {
        expression: {
          type: "string",
          description: "Arithmetic expression, e.g. (12 + 30) / 6 ^ 2",
          minLength: 1,
          maxLength: 500,
        },
      },
      ["expression"],
    ),
    permission: "tool.execute",
    source: "builtin",
    tags: ["math", "calculate", "arithmetic", "compute"],
    handler: ({ expression }) => ({ expression, value: evaluateExpression(expression) }),
    verify: (output) => ({
      ok: Number.isFinite(output.value),
      reason: Number.isFinite(output.value) ? "value computed" : "value was not finite",
    }),
  });

  registry.register<{ text: string }, {
    tokens: number;
    ids: number[];
    pieces: string[];
    unknown: number;
    truncated: boolean;
    vocabSize: number;
    tokenizerVersion: string;
  }>({
    name: "alpha.tokenizer.analyze",
    description:
      "Tokenise text with Alpha's own tokenizer and report the token ids, the subword pieces, the count of unknown tokens, and the vocabulary version.",
    module: "tokenizer",
    inputSchema: objectSchema(
      { text: { type: "string", description: "Text to tokenise", minLength: 1, maxLength: 8000 } },
      ["text"],
    ),
    permission: "tool.execute",
    source: "builtin",
    tags: ["tokenizer", "tokens", "subword", "analyse"],
    handler: ({ text }, ctx) => {
      const tokenizer = service<AlphaTokenizer>(ctx.services, "tokenizer");
      const encoded = tokenizer.encodeDetailed(text);
      return {
        tokens: encoded.ids.length,
        ids: encoded.ids.slice(0, 64),
        pieces: encoded.tokens.slice(0, 64),
        unknown: encoded.unknown,
        truncated: encoded.truncated,
        vocabSize: tokenizer.vocabSize,
        tokenizerVersion: tokenizer.version,
      };
    },
    verify: (output) => ({
      ok: output.tokens > 0,
      reason: output.tokens > 0 ? "text tokenised" : "tokenizer produced no tokens",
    }),
  });

  registry.register<{ query: string; topK?: number; minScore?: number }, {
    hits: { rank: number; score: number; text: string; title: string; documentId: string }[];
    collection: string;
    searched: number;
  }>({
    name: "alpha.corpus.search",
    description:
      "Semantic search over the documents ingested into Alpha's RAG pipeline. Uses Alpha's own embeddings and vector store; returns the closest chunks with scores.",
    module: "rag",
    inputSchema: objectSchema(
      {
        query: { type: "string", description: "Natural-language query", minLength: 1, maxLength: 2000 },
        topK: { type: "integer", description: "Number of chunks to return", minimum: 1, maximum: 12, default: 4 },
        minScore: { type: "number", description: "Minimum cosine score", minimum: -1, maximum: 1 },
      },
      ["query"],
    ),
    permission: "rag.query",
    source: "builtin",
    tags: ["search", "retrieval", "documents", "rag", "knowledge"],
    handler: ({ query, topK, minScore }, ctx) => {
      const embedder = service<AlphaEmbedder>(ctx.services, "embedder");
      const store = service<AlphaVectorStore>(ctx.services, "vectorStore");
      const collection = "alpha_documents";
      const info = store.getCollection(collection);
      if (!info || info.recordCount === 0) {
        return { hits: [], collection, searched: 0 };
      }
      const vector = embedder.embedQuery(query).vector;
      const hits = store.search({
        collection,
        vector,
        topK: topK ?? 4,
        minScore: minScore ?? 0.05,
      });
      return {
        hits: hits.map((hit) => ({
          rank: hit.rank,
          score: Number(hit.score.toFixed(4)),
          text: hit.record.text,
          title: String(hit.record.metadata.title ?? "untitled"),
          documentId: hit.record.sourceId ?? "unknown",
        })),
        collection,
        searched: info.recordCount,
      };
    },
    verify: (output) => ({
      ok: Array.isArray(output.hits),
      reason: Array.isArray(output.hits)
        ? `${output.hits.length} chunk(s) retrieved from ${output.searched} indexed`
        : "search returned no result list",
    }),
  });

  registry.register<{ query: string; topK?: number; scopes?: string[]; sessionId?: string }, {
    memories: { id: string; scope: string; key: string; content: string; relevance: number; similarity: number }[];
  }>({
    name: "alpha.memory.search",
    description:
      "Retrieve memories relevant to a query from Alpha's memory store, scored by embedding similarity, recency and importance.",
    module: "memory",
    inputSchema: objectSchema(
      {
        query: { type: "string", description: "What to recall", minLength: 1, maxLength: 2000 },
        topK: { type: "integer", minimum: 1, maximum: 12, default: 5 },
        scopes: {
          type: "array",
          description: "Memory scopes to search",
          items: { type: "string", enum: ["conversation", "session", "long-term"] },
        },
        sessionId: { type: "string", description: "Session to scope conversation/session memory to" },
      },
      ["query"],
    ),
    permission: "memory.read",
    source: "builtin",
    tags: ["memory", "recall", "remember", "context"],
    handler: ({ query, topK, scopes, sessionId }, ctx) => {
      const memory = service<AlphaMemoryStore>(ctx.services, "memory");
      const entries = memory.retrieve(query, {
        topK: topK ?? 5,
        scopes: scopes as MemoryScope[] | undefined,
        sessionId: sessionId ?? null,
      });
      return {
        memories: entries.map((entry) => ({
          id: entry.record.id,
          scope: entry.record.scope,
          key: entry.record.key,
          content: entry.record.content,
          relevance: entry.relevance,
          similarity: entry.similarity,
        })),
      };
    },
    verify: (output) => ({
      ok: Array.isArray(output.memories),
      reason: `${output.memories.length} memory record(s) scored`,
    }),
  });

  registry.register<{
    key: string;
    content: string;
    scope: string;
    sessionId?: string;
    importance?: number;
    approved?: boolean;
  }, { id: string; scope: string; key: string; approved: boolean; requiresApproval: boolean }>({
    name: "alpha.memory.write",
    description:
      "Write a memory. Conversation and session scopes are written immediately. Long-term memory needs the memory.write.long-term permission and explicit approval.",
    module: "memory",
    inputSchema: objectSchema(
      {
        key: { type: "string", description: "Short identifier, e.g. user.preference.theme", minLength: 1, maxLength: 120 },
        content: { type: "string", description: "What to remember", minLength: 1, maxLength: 4000 },
        scope: { type: "string", enum: ["conversation", "session", "long-term"] },
        sessionId: { type: "string", description: "Required for conversation and session scopes" },
        importance: { type: "number", minimum: 0, maximum: 1 },
        approved: { type: "boolean", description: "Required true for long-term memory" },
      },
      ["key", "content", "scope"],
    ),
    permission: "memory.write",
    source: "builtin",
    dangerous: true,
    characteristics: { mutates: true },
    tags: ["memory", "remember", "store", "write"],
    handler: ({ key, content, scope, sessionId, importance, approved }, ctx) => {
      const memory = service<AlphaMemoryStore>(ctx.services, "memory");
      if (scope === "long-term") {
        // A second, narrower permission is required for durable memory.
        ctx.policy.assert(ctx.actorId, "memory.write.long-term", key);
      }
      const record = memory.write({
        key,
        content,
        scope: scope as MemoryScope,
        sessionId: sessionId ?? null,
        importance,
        approved: scope === "long-term" ? approved === true : true,
        source: "tool",
      });
      return {
        id: record.id,
        scope: record.scope,
        key: record.key,
        approved: record.approved,
        requiresApproval: scope === "long-term",
      };
    },
    verify: (output) => ({
      ok: Boolean(output.id),
      reason: output.id ? `memory "${output.key}" stored in ${output.scope}` : "memory was not stored",
    }),
  });

  registry.register<{ confirm: boolean }, { cleared: number; collections: number }>({
    name: "alpha.admin.clear_vector_store",
    description:
      "Destructive: delete every vector in Alpha's vector store. Requires an explicit approval recorded by the policy engine.",
    module: "vector",
    inputSchema: objectSchema(
      {
        confirm: {
          type: "boolean",
          description: "Must be true; the call is rejected otherwise",
        },
      },
      ["confirm"],
    ),
    permission: "tool.execute.dangerous",
    source: "builtin",
    dangerous: true,
    requiresApproval: true,
    characteristics: { mutates: true },
    tags: ["admin", "vector", "delete", "destructive"],
    handler: ({ confirm }, ctx) => {
      if (confirm !== true) {
        throw new AlphaValidationError("tools", "clearing the vector store requires confirm = true");
      }
      const store = service<AlphaVectorStore>(ctx.services, "vectorStore");
      const before = store.count();
      const collections = store.listCollections().length;
      store.clear();
      return { cleared: before, collections };
    },
    verify: (output) => ({
      ok: output.cleared >= 0,
      reason: `removed ${output.cleared} vector(s)`,
    }),
  });

  return [...BUILTIN_TOOL_NAMES];
}
