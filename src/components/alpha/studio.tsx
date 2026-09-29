/**
 * Studio primitives.
 *
 * Thin framing, warm off-whites, muted neutrals and small-caps labels. These
 * components are presentation only — every number they render comes from
 * `workspace.snapshot()`, and every status comes from Alpha's own manifest.
 */

import { cn } from "@/lib/utils";
import {
  modelStageLabel,
  statusLabel,
  type AlphaModelStage,
  type AlphaStatus,
} from "@/alpha";
import type { ReactNode } from "react";

const STATUS_STYLES: Record<AlphaStatus, string> = {
  ready: "border-chart-2/35 bg-chart-2/10 text-chart-2",
  "in-development": "border-chart-4/40 bg-chart-4/12 text-chart-4",
  untrained: "border-chart-5/35 bg-chart-5/10 text-chart-5",
  "not-configured": "border-chart-3/30 bg-chart-3/8 text-chart-3",
  planned: "border-border bg-muted text-muted-foreground",
};

const STAGE_STYLES: Record<AlphaModelStage, string> = {
  architecture: "border-border bg-muted text-muted-foreground",
  untrained: "border-chart-5/35 bg-chart-5/10 text-chart-5",
  trained: "border-chart-2/35 bg-chart-2/10 text-chart-2",
  "fine-tuned": "border-chart-1/35 bg-chart-1/10 text-chart-1",
  production: "border-primary/40 bg-primary/10 text-primary",
};

export function StatusBadge({ status, className }: { status: AlphaStatus; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-[2px] text-[10px] font-medium uppercase tracking-[0.14em]",
        STATUS_STYLES[status],
        className,
      )}
    >
      {statusLabel(status)}
    </span>
  );
}

export function StageBadge({ stage, className }: { stage: AlphaModelStage; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-[2px] text-[10px] font-medium uppercase tracking-[0.14em]",
        STAGE_STYLES[stage],
        className,
      )}
    >
      {modelStageLabel(stage)}
    </span>
  );
}

export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cn("studio-eyebrow", className)}>{children}</p>;
}

export function Rule({ className }: { className?: string }) {
  return <div className={cn("h-px w-full bg-border", className)} />;
}

/** A mounted section: hairline frame, editorial heading, optional lede. */
export function Frame({
  title,
  lede,
  status,
  actions,
  children,
  className,
}: {
  title: string;
  lede?: string;
  status?: AlphaStatus;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("studio-frame", className)}>
      <header className="flex flex-col gap-3 border-b border-border px-5 py-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <h2 className="studio-serif text-lg text-foreground">{title}</h2>
            {status ? <StatusBadge status={status} /> : null}
          </div>
          {lede ? <p className="max-w-2xl text-xs leading-5 text-muted-foreground">{lede}</p> : null}
        </div>
        {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
      </header>
      <div className="px-5 py-5">{children}</div>
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
  className,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  className?: string;
}) {
  return (
    <div className={cn("space-y-1", className)}>
      <Eyebrow>{label}</Eyebrow>
      <p className="studio-serif text-xl leading-none text-foreground">{value}</p>
      {hint ? <p className="text-[11px] leading-4 text-muted-foreground">{hint}</p> : null}
    </div>
  );
}

export function StatGrid({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn("grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-3 lg:grid-cols-4", className)}>
      {children}
    </div>
  );
}

/** Key/value line used inside dense tables and definition lists. */
export function KeyValue({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-border/60 py-2 last:border-b-0">
      <span className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">{label}</span>
      <span className="text-right font-mono text-xs text-foreground">{children}</span>
    </div>
  );
}

export function EmptyNote({ children }: { children: ReactNode }) {
  return (
    <p className="studio-muted-surface px-4 py-3 text-xs leading-5 text-muted-foreground">{children}</p>
  );
}

export function Mono({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cn("font-mono text-[11px] text-muted-foreground", className)}>{children}</span>;
}

/** Honest provenance line shown wherever generated text appears. */
export function WarningNote({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-md border border-chart-4/40 bg-chart-4/10 px-4 py-3 text-xs leading-5 text-foreground">
      {children}
    </p>
  );
}

export function Pill({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-sm border border-border px-2 py-[2px] font-mono text-[10px] text-muted-foreground",
        className,
      )}
    >
      {children}
    </span>
  );
}
