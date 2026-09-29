/**
 * Governance — security and observability.
 *
 * The security half shows who may do what and every decision the policy engine
 * made. The observability half shows the metrics, traces and logs Alpha
 * recorded while doing real work.
 */

import { EmptyNote, Eyebrow, Frame, KeyValue, Mono, Pill, Stat, StatGrid, WarningNote } from "@/components/alpha/studio";
import { INJECTION_PATTERNS } from "@/alpha";
import type { AlphaRuntime } from "@/hooks/use-alpha";
import { AlertTriangle, CheckCircle2, ShieldCheck } from "lucide-react";

export function GovernancePanel({ alpha }: { alpha: AlphaRuntime }) {
  const snapshot = alpha.snapshot;
  const security = snapshot?.security;
  const observability = snapshot?.observability;
  const counters = observability?.metrics.filter((metric) => metric.kind !== "histogram") ?? [];
  const histograms = observability?.metrics.filter((metric) => metric.kind === "histogram") ?? [];

  return (
    <div className="space-y-6">
      <Frame
        title="Security"
        status={snapshot?.statuses.security ?? "planned"}
        lede="Roles grant permissions; agent scopes narrow them further; gated tools need a recorded approval; every decision is audited in a hash-chained log."
      >
        {security ? (
          <>
            <StatGrid className="mb-6">
              <Stat label="Audit records" value={security.auditSize} hint="append-only" />
              <Stat
                label="Chain"
                value={
                  <span className="inline-flex items-center gap-2">
                    {security.chainIntact ? (
                      <CheckCircle2 className="size-4 text-chart-2" />
                    ) : (
                      <AlertTriangle className="size-4 text-destructive" />
                    )}
                    {security.chainIntact ? "intact" : "broken"}
                  </span>
                }
                hint="replayable hash chain"
              />
              <Stat
                label="Policy decisions"
                value={security.policyEvents.length}
                hint={`${security.policyEvents.filter((event) => !event.allowed).length} denied`}
              />
              <Stat label="Injection patterns" value={INJECTION_PATTERNS.length} hint="heuristic defence" />
            </StatGrid>

            <div className="grid gap-6 lg:grid-cols-2">
              <div>
                <Eyebrow>Actors</Eyebrow>
                <div className="mt-2">
                  {security.roles.map((actor) => (
                    <KeyValue key={actor.actorId} label={actor.actorId}>
                      {actor.role ?? "no role assigned"}
                    </KeyValue>
                  ))}
                </div>
                <div className="mt-4 rounded-md border border-border p-4">
                  <Eyebrow>Agent scope</Eyebrow>
                  <p className="mt-2 text-[11px] leading-4 text-muted-foreground">
                    The agent actor holds six tools and none of the destructive ones. It cannot write long-term memory,
                    cannot reach the network, and cannot touch the filesystem — and the attempts are recorded when it tries.
                  </p>
                </div>
              </div>

              <div>
                <Eyebrow>Recent policy decisions</Eyebrow>
                {security.policyEvents.length === 0 ? (
                  <EmptyNote>No policy decision recorded yet.</EmptyNote>
                ) : (
                  <div className="mt-2 max-h-80 space-y-2 overflow-auto pr-1">
                    {security.policyEvents.map((event, index) => (
                      <div key={`${event.at}-${index}`} className="rounded-md border border-border px-3 py-2">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono text-[11px] text-foreground">{event.permission}</span>
                          <Pill className={event.allowed ? "border-chart-2/40 text-chart-2" : "border-destructive/40 text-destructive"}>
                            {event.allowed ? "allowed" : "denied"}
                          </Pill>
                        </div>
                        <Mono>{event.actorId}</Mono>
                        <p className="mt-1 text-[11px] leading-4 text-muted-foreground">{event.reason}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </>
        ) : (
          <EmptyNote>Alpha is still initialising.</EmptyNote>
        )}

        <div className="mt-6">
          <WarningNote>
            <span className="inline-flex items-center gap-2">
              <ShieldCheck className="size-3.5" />
              Prompt-injection detection is heuristic. It reduces risk by scoring instruction-override, role-hijack,
              delimiter and exfiltration patterns and wrapping retrieved text as data — it does not prove safety, and this
              panel never claims it does.
            </span>
          </WarningNote>
        </div>
      </Frame>

      <Frame
        title="Observability"
        status={snapshot?.statuses.observability ?? "planned"}
        lede="Counters, histograms with real percentiles, traces with span ids, and structured logs — recorded by the modules as they work."
      >
        {observability ? (
          <>
            <StatGrid className="mb-6">
              <Stat label="Spans" value={observability.spans} hint={`${observability.openSpans} open`} />
              <Stat label="Traces" value={observability.traces.length} />
              <Stat label="Errors" value={observability.errorCount} />
              <Stat
                label="Log levels"
                value={`${observability.logCounts.info}/${observability.logCounts.warn}/${observability.logCounts.error}`}
                hint="info / warn / error"
              />
            </StatGrid>

            <div className="grid gap-6 lg:grid-cols-2">
              <div>
                <Eyebrow>Counters</Eyebrow>
                <div className="mt-2 max-h-64 overflow-auto pr-1">
                  {counters.length === 0 ? (
                    <EmptyNote>Nothing measured yet.</EmptyNote>
                  ) : (
                    counters.map((metric) => (
                      <KeyValue key={metric.name} label={metric.name.replace("alpha.", "")}>
                        {metric.value.toLocaleString()} {metric.unit === "count" ? "" : metric.unit}
                      </KeyValue>
                    ))
                  )}
                </div>
              </div>
              <div>
                <Eyebrow>Latency distributions</Eyebrow>
                <div className="mt-2 max-h-64 space-y-2 overflow-auto pr-1">
                  {histograms.length === 0 ? (
                    <EmptyNote>No samples yet.</EmptyNote>
                  ) : (
                    histograms.map((metric) => (
                      <div key={metric.name} className="rounded-md border border-border px-3 py-2">
                        <p className="font-mono text-[11px] text-foreground">{metric.name.replace("alpha.", "")}</p>
                        {metric.histogram ? (
                          <Mono>
                            n={metric.histogram.count} · p50 {metric.histogram.p50.toFixed(1)} · p95{" "}
                            {metric.histogram.p95.toFixed(1)} · max {metric.histogram.max.toFixed(1)} {metric.unit}
                          </Mono>
                        ) : null}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>

            <div className="mt-6 grid gap-6 lg:grid-cols-2">
              <div>
                <Eyebrow>Traces</Eyebrow>
                <div className="mt-2 max-h-64 space-y-2 overflow-auto pr-1">
                  {observability.traces.length === 0 ? (
                    <EmptyNote>No trace recorded yet.</EmptyNote>
                  ) : (
                    observability.traces.map((trace) => (
                      <div key={trace.traceId} className="rounded-md border border-border px-3 py-2">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono text-[11px] text-foreground">{trace.rootSpan}</span>
                          <Pill className={trace.status === "error" ? "border-destructive/40 text-destructive" : ""}>
                            {trace.status}
                          </Pill>
                        </div>
                        <Mono>
                          {trace.traceId.slice(0, 12)}… · {trace.spans} span(s) · {trace.durationMs} ms ·{" "}
                          {trace.modules.join(", ")}
                        </Mono>
                      </div>
                    ))
                  )}
                </div>
              </div>
              <div>
                <Eyebrow>Structured logs</Eyebrow>
                <div className="mt-2 max-h-64 space-y-1 overflow-auto pr-1">
                  {observability.logs.length === 0 ? (
                    <EmptyNote>No log entries yet.</EmptyNote>
                  ) : (
                    observability.logs.slice(0, 24).map((entry) => (
                      <div key={entry.id} className="flex items-start gap-2 border-b border-border/50 py-1">
                        <Pill className={entry.level === "error" ? "border-destructive/40 text-destructive" : ""}>
                          {entry.level}
                        </Pill>
                        <span className="font-mono text-[10px] text-muted-foreground">{entry.module}</span>
                        <span className="flex-1 font-mono text-[11px] leading-4 text-foreground">{entry.message}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </>
        ) : (
          <EmptyNote>Alpha is still initialising.</EmptyNote>
        )}
      </Frame>
    </div>
  );
}
