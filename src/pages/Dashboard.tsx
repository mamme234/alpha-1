/**
 * Alpha workspace.
 *
 * A studio for the model: train it, talk to it, give it documents and memory,
 * run agents and workflows, and watch what it recorded. The status strip reads
 * from the live workspace, so the label at the top is the truth of the current
 * session rather than a design decision.
 */

import { Button } from "@/components/ui/button";
import { StageBadge } from "@/components/alpha/studio";
import { GovernancePanel } from "@/components/alpha/panels/GovernancePanel";
import { InferencePanel } from "@/components/alpha/panels/InferencePanel";
import { KnowledgePanel } from "@/components/alpha/panels/KnowledgePanel";
import { OverviewPanel } from "@/components/alpha/panels/OverviewPanel";
import { TrainingPanel } from "@/components/alpha/panels/TrainingPanel";
import { WorkPanel } from "@/components/alpha/panels/WorkPanel";
import { useAlpha } from "@/hooks/use-alpha";
import { useAuth } from "@/hooks/use-auth";
import { cn } from "@/lib/utils";
import { AlertTriangle, Home, Loader2, LogOut, Sparkles } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router";

const TABS = [
  { id: "overview", label: "Overview", hint: "model · architecture" },
  { id: "training", label: "Training", hint: "corpus · loss · checkpoints" },
  { id: "inference", label: "Inference", hint: "generate · sample" },
  { id: "knowledge", label: "Knowledge", hint: "rag · vectors · memory" },
  { id: "work", label: "Work", hint: "agents · tools · automation" },
  { id: "governance", label: "Governance", hint: "security · traces" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function Dashboard() {
  const { user, signOut } = useAuth();
  const alpha = useAlpha();
  const navigate = useNavigate();
  const [tab, setTab] = useState<TabId>("overview");

  const handleSignOut = async () => {
    await signOut();
    navigate("/");
  };

  const snapshot = alpha.snapshot;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border bg-background/90 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-4 px-6 py-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-baseline gap-4">
              <button type="button" onClick={() => navigate("/")} className="studio-serif text-lg tracking-tight">
                Alpha
              </button>
              <span className="studio-eyebrow hidden sm:inline">workspace</span>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              {snapshot ? <StageBadge stage={snapshot.model.stage} /> : null}
              <span className="text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
                {user?.email ?? user?.displayName ?? "signed in"}
              </span>
              <Button variant="ghost" size="sm" onClick={() => navigate("/")}>
                <Home className="mr-2 size-3.5" />
                Home
              </Button>
              <Button variant="outline" size="sm" onClick={handleSignOut}>
                <LogOut className="mr-2 size-3.5" />
                Sign out
              </Button>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-border pt-3 text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
            <span>
              model <span className="text-foreground">{snapshot?.model.name ?? "—"}</span>
            </span>
            <span>
              parameters <span className="text-foreground">{snapshot?.model.parameterCount.toLocaleString() ?? "—"}</span>
            </span>
            <span>
              vocabulary <span className="text-foreground">{snapshot?.tokenizer.vocabSize.toLocaleString() ?? "—"}</span>
            </span>
            <span>
              checkpoint{" "}
              <span className="text-foreground">
                {snapshot?.training.checkpoint ? `step ${snapshot.training.checkpoint.step}` : "none"}
              </span>
            </span>
            <span>
              vectors <span className="text-foreground">{snapshot?.rag.vectors ?? 0}</span>
            </span>
            <span className="ml-auto flex items-center gap-2">
              {alpha.busy ? (
                <>
                  <Loader2 className="size-3 animate-spin" />
                  <span className="text-foreground">{alpha.busy}</span>
                </>
              ) : (
                <>
                  <span className="size-1.5 rounded-full bg-chart-2" />
                  idle · artifacts persisted to Convex
                </>
              )}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl gap-8 px-6 py-8 lg:flex">
        <nav className="mb-6 lg:mb-0 lg:w-56 lg:shrink-0">
          <ul className="flex gap-2 overflow-x-auto pb-1 lg:sticky lg:top-8 lg:flex-col lg:overflow-visible lg:pb-0">
            {TABS.map((item) => (
              <li key={item.id}>
                <button
                  type="button"
                  onClick={() => setTab(item.id)}
                  className={cn(
                    "flex w-full min-w-max flex-col items-start rounded-md border px-3 py-2 text-left transition-colors",
                    tab === item.id
                      ? "border-primary/40 bg-primary/5 text-foreground"
                      : "border-transparent text-muted-foreground hover:border-border hover:text-foreground",
                  )}
                >
                  <span className="text-xs uppercase tracking-[0.16em]">{item.label}</span>
                  <span className="text-[10px] text-muted-foreground">{item.hint}</span>
                </button>
              </li>
            ))}
          </ul>
        </nav>

        <div className="min-w-0 flex-1">
          {alpha.bootError ? (
            <div className="studio-frame flex items-start gap-3 p-6">
              <AlertTriangle className="mt-0.5 size-4 text-destructive" />
              <div>
                <p className="text-sm text-foreground">Alpha could not initialise</p>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">{alpha.bootError}</p>
              </div>
            </div>
          ) : !alpha.ready ? (
            <div className="studio-frame flex items-center gap-3 p-6">
              <Loader2 className="size-4 animate-spin text-muted-foreground" />
              <div>
                <p className="text-sm text-foreground">Training Alpha's tokenizer on the bundled corpus…</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Vocabulary, model and every subsystem are constructed in the browser. This takes a moment and happens once
                  per session.
                </p>
              </div>
            </div>
          ) : (
            <>
              {tab === "overview" ? <OverviewPanel alpha={alpha} /> : null}
              {tab === "training" ? <TrainingPanel alpha={alpha} /> : null}
              {tab === "inference" ? <InferencePanel alpha={alpha} /> : null}
              {tab === "knowledge" ? <KnowledgePanel alpha={alpha} /> : null}
              {tab === "work" ? <WorkPanel alpha={alpha} /> : null}
              {tab === "governance" ? <GovernancePanel alpha={alpha} /> : null}
            </>
          )}
        </div>
      </main>

      <footer className="mx-auto w-full max-w-7xl px-6 pb-10">
        <div className="flex flex-col gap-2 border-t border-border pt-4 text-[11px] leading-5 text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
          <span className="flex items-center gap-2">
            <Sparkles className="size-3.5" />
            Alpha runs in this tab. Model, tokenizer, vectors and memories persist to Alpha's own Convex tables.
          </span>
          <span>No external AI provider is contacted anywhere in this application.</span>
        </div>
      </footer>
    </div>
  );
}
