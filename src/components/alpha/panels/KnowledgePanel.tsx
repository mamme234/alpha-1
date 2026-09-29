/**
 * Knowledge — retrieval, vectors and memory.
 *
 * The three subsystems that hold text between turns. Ingestion embeds with
 * Alpha's own encoder, the vector store owns its vectors locally, and memory
 * keeps its scopes distinct with approval required before anything persists.
 */

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { EmptyNote, Eyebrow, Frame, KeyValue, Mono, Stat, StatGrid } from "@/components/alpha/studio";
import type { AlphaRuntime } from "@/hooks/use-alpha";
import { useMemo, useState } from "react";
import { Database, Loader2, Plus, Search, Trash2 } from "lucide-react";
import { toast } from "sonner";

const SAMPLE_DOCUMENT = {
  title: "Alpha security policy",
  license: "CC0-1.0",
  content: `Alpha runs every privileged action through a policy engine. An actor has a role, a role grants permissions, and an agent is additionally confined to a tool allow-list registered with its scope.

Long-term memory is written only with explicit approval. Every approval is recorded in a hash-chained audit log, so a decision can be replayed and a tampered record can be detected.

Retrieved documents are wrapped as data rather than instructions before they reach the model, and prompt-injection patterns are scored so a document cannot quietly become a command.`,
};

export function KnowledgePanel({ alpha }: { alpha: AlphaRuntime }) {
  const snapshot = alpha.snapshot;
  const [title, setTitle] = useState(SAMPLE_DOCUMENT.title);
  const [content, setContent] = useState(SAMPLE_DOCUMENT.content);
  const [license, setLicense] = useState(SAMPLE_DOCUMENT.license);
  const [query, setQuery] = useState("how does alpha handle approvals?");
  const [answer, setAnswer] = useState<string>("");
  const [memKey, setMemKey] = useState("user.preference.pace");
  const [memContent, setMemContent] = useState("Prefers long training runs overnight.");
  const [memScope, setMemScope] = useState<"conversation" | "session" | "long-term">("long-term");
  const [memApproved, setMemApproved] = useState(false);
  const [recallQuery, setRecallQuery] = useState("training pace");

  const memories = snapshot?.memory.entries ?? [];
  const documents = snapshot?.rag.documents ?? [];
  const collections = snapshot?.rag.collections ?? [];

  const memoryCounts = useMemo(
    () => ({
      conversation: memories.filter((memory) => memory.scope === "conversation").length,
      session: memories.filter((memory) => memory.scope === "session").length,
      "long-term": memories.filter((memory) => memory.scope === "long-term").length,
    }),
    [memories],
  );

  const ingest = async () => {
    const document = await alpha.ingest({ title, content, license });
    if (document) toast.success(`Ingested "${document.title}" as ${document.chunks} chunk(s)`);
    else toast.error("Ingestion failed");
  };

  const ask = async () => {
    const result = await alpha.generate(query, { maxNewTokens: 48 }, "rag");
    if (!result) return;
    // Retrieval mode returns a RagAnswer (with citations); raw mode returns the
    // generation itself. Both carry the model stage that produced them.
    setAnswer("answer" in result ? result.answer : result.text);
  };

  const saveMemory = async () => {
    const record = await alpha.writeMemory({
      scope: memScope,
      key: memKey,
      content: memContent,
      importance: 0.7,
      approved: memScope === "long-term" ? memApproved : true,
      sessionId: memScope === "long-term" ? null : "workspace-session",
    });
    if (record) toast.success(`Stored ${record.scope} memory "${record.key}"`);
    else toast.error("Long-term memory requires the approval checkbox");
  };

  const recallNow = async () => {
    const entries = await alpha.recall(recallQuery, { topK: 5 });
    if (entries) {
      toast.success(
        entries.length
          ? `Top memory relevance ${entries[0].relevance.toFixed(4)} (similarity ${entries[0].similarity.toFixed(4)})`
          : "No memories matched",
      );
    }
  };

  return (
    <div className="space-y-6">
      <Frame
        title="Retrieval augmented generation"
        status={snapshot?.statuses.rag ?? "planned"}
        lede="documents → parsing → chunking with overlap → Alpha embeddings → vector storage → retrieval → context assembly → Alpha LLM → answer with citations."
      >
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-3">
            <Eyebrow>Ingest a document</Eyebrow>
            <Input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Document title" />
            <Textarea
              value={content}
              onChange={(event) => setContent(event.target.value)}
              className="min-h-32 resize-y font-mono text-xs"
            />
            <div className="flex items-center gap-3">
              <Input
                value={license}
                onChange={(event) => setLicense(event.target.value)}
                placeholder="Licence"
                className="max-w-40"
              />
              <Button onClick={ingest} disabled={Boolean(alpha.busy)}>
                {alpha.busy === "ingesting" ? (
                  <Loader2 className="mr-2 size-3.5 animate-spin" />
                ) : (
                  <Plus className="mr-2 size-3.5" />
                )}
                Ingest and embed
              </Button>
            </div>
            <p className="text-[11px] leading-4 text-muted-foreground">
              Only text and markdown are parsed. Any other kind fails loudly instead of being mis-parsed, and the licence
              travels with every chunk so training and retrieval data stay auditable.
            </p>
          </div>

          <div className="space-y-3">
            <Eyebrow>Ask the corpus</Eyebrow>
            <Textarea value={query} onChange={(event) => setQuery(event.target.value)} className="min-h-20 font-mono text-xs" />
            <Button variant="outline" onClick={ask} disabled={Boolean(alpha.busy)}>
              <Search className="mr-2 size-3.5" />
              Retrieve and answer
            </Button>
            <div className="rounded-md border border-border bg-muted/25 p-4">
              <Eyebrow>Answer</Eyebrow>
              <pre className="mt-2 whitespace-pre-wrap font-mono text-xs leading-5 text-foreground">
                {answer || "—"}
              </pre>
            </div>
            <p className="text-[11px] leading-4 text-muted-foreground">
              The answer comes from Alpha's own weights over the retrieved context. While the model is untrained this reads
              as noise; the citations below are still real, which is exactly what makes the retrieval half checkable.
            </p>
          </div>
        </div>
      </Frame>

      <div className="grid gap-6 lg:grid-cols-2">
        <Frame
          title="Corpus"
          status="ready"
          lede="Documents ingested into the retrieval pipeline, and the vectors derived from them."
        >
          <StatGrid className="mb-4">
            <Stat label="Documents" value={documents.length} />
            <Stat label="Chunks indexed" value={snapshot?.rag.vectors ?? 0} />
            <Stat label="Collections" value={collections.length} />
          </StatGrid>
          {documents.length === 0 ? (
            <EmptyNote>No documents ingested yet.</EmptyNote>
          ) : (
            <div className="space-y-2">
              {documents.map((document) => (
                <div key={document.id} className="flex items-start justify-between gap-3 rounded-md border border-border px-3 py-2">
                  <div>
                    <p className="text-xs text-foreground">{document.title}</p>
                    <Mono>
                      {document.chunks} chunks · {document.tokens} tokens · {document.license}
                    </Mono>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={async () => {
                      const removed = await alpha.removeDocument(document.id);
                      if (removed !== null) toast.success(`Removed ${removed} vector(s)`);
                    }}
                    aria-label={`Remove ${document.title}`}
                  >
                    <Trash2 className="size-3.5" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </Frame>

        <Frame
          title="Vector store"
          status={snapshot?.statuses.vector ?? "planned"}
          lede="Alpha's own store: collections as namespaces, vectors with metadata, exact cosine search. Nothing is hosted elsewhere."
        >
          {collections.length === 0 ? (
            <EmptyNote>No collections yet — ingest a document to create one.</EmptyNote>
          ) : (
            <div className="space-y-2">
              {collections.map((collection) => (
                <div key={collection.name} className="rounded-md border border-border px-3 py-2">
                  <div className="flex items-center justify-between gap-3">
                    <p className="font-mono text-xs text-foreground">{collection.name}</p>
                    <Mono>{collection.metric}</Mono>
                  </div>
                  <div className="mt-1 flex items-center gap-4">
                    <Mono>{collection.recordCount} records</Mono>
                    <Mono>{collection.dimension} dimensions</Mono>
                    <Mono>created {new Date(collection.createdAt).toLocaleTimeString()}</Mono>
                  </div>
                </div>
              ))}
              <div className="flex items-center gap-2 pt-2 text-[11px] text-muted-foreground">
                <Database className="size-3.5" />
                Search is an exact scan: O(records × dimensions), stated rather than replaced by an approximate index that
                does not exist.
              </div>
            </div>
          )}
        </Frame>
      </div>

      <Frame
        title="Memory"
        status={snapshot?.statuses.memory ?? "planned"}
        lede="Three scopes with different lifetimes. Long-term memory needs approval and can be deleted, one record or one scope at a time."
      >
        <StatGrid className="mb-5">
          <Stat label="Conversation" value={memoryCounts.conversation} hint="one exchange" />
          <Stat label="Session" value={memoryCounts.session} hint="this workspace" />
          <Stat label="Long-term" value={memoryCounts["long-term"]} hint="persisted with approval" />
          <Stat
            label="Relevance model"
            value="local"
            hint="similarity × recency × importance × usage"
          />
        </StatGrid>

        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-3">
            <Eyebrow>Write a memory</Eyebrow>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label className="text-[11px] text-muted-foreground">Key</Label>
                <Input value={memKey} onChange={(event) => setMemKey(event.target.value)} />
              </div>
              <div className="space-y-1">
                <Label className="text-[11px] text-muted-foreground">Scope</Label>
                <select
                  value={memScope}
                  onChange={(event) => setMemScope(event.target.value as "conversation" | "session" | "long-term")}
                  className="h-9 w-full rounded-md border border-input bg-transparent px-2 text-xs"
                >
                  <option value="conversation">conversation</option>
                  <option value="session">session</option>
                  <option value="long-term">long-term</option>
                </select>
              </div>
            </div>
            <Textarea value={memContent} onChange={(event) => setMemContent(event.target.value)} className="min-h-20" />
            {memScope === "long-term" ? (
              <label className="flex items-center gap-2 text-[11px] text-muted-foreground">
                <input
                  type="checkbox"
                  checked={memApproved}
                  onChange={(event) => setMemApproved(event.target.checked)}
                  className="size-3.5"
                />
                I approve storing this as long-term memory
              </label>
            ) : null}
            <Button onClick={saveMemory} disabled={Boolean(alpha.busy)}>
              Store memory
            </Button>
          </div>

          <div className="space-y-3">
            <Eyebrow>Recall</Eyebrow>
            <div className="flex gap-2">
              <Input value={recallQuery} onChange={(event) => setRecallQuery(event.target.value)} />
              <Button variant="outline" onClick={recallNow} disabled={Boolean(alpha.busy)}>
                Score
              </Button>
            </div>
            <div className="max-h-72 space-y-2 overflow-auto pr-1">
              {memories.length === 0 ? (
                <EmptyNote>No memories stored yet.</EmptyNote>
              ) : (
                memories.map((memory) => (
                  <div key={memory.id} className="rounded-md border border-border px-3 py-2">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <p className="font-mono text-[11px] text-foreground">{memory.key}</p>
                        <Mono>
                          {memory.scope}
                          {memory.approved ? " · approved" : " · unapproved"}
                          {memory.sessionId ? ` · ${memory.sessionId}` : ""}
                        </Mono>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        aria-label={`Delete ${memory.key}`}
                        onClick={async () => {
                          const removed = await alpha.forgetMemory(memory.id);
                          if (removed) toast.success("Memory deleted");
                        }}
                      >
                        <Trash2 className="size-3.5" />
                      </Button>
                    </div>
                    <p className="mt-1 text-[11px] leading-4 text-muted-foreground">{memory.content}</p>
                  </div>
                ))
              )}
            </div>
            <div className="rounded-md border border-border p-3">
              <KeyValue label="scoring">{snapshot?.statuses.memory === "ready" ? "cosine × decay × importance × usage" : "—"}</KeyValue>
              <KeyValue label="store location">Convex (Alpha's own tables)</KeyValue>
            </div>
          </div>
        </div>
      </Frame>
    </div>
  );
}
